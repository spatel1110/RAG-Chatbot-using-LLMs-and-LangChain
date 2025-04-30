import streamlit as st
import requests
import time
import logging

BACKEND_URL = "http://127.0.0.1:8000"
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot")

if "messages" not in st.session_state: st.session_state.messages = []
if "selected_model" not in st.session_state: st.session_state.selected_model = None
if "current_stats" not in st.session_state: st.session_state.current_stats = None
if "file_uploader_key" not in st.session_state: st.session_state.file_uploader_key = 0
if "upload_status_msg" not in st.session_state: st.session_state.upload_status_msg = ""


def safe_request(method, url, **kwargs):
    """Wrapper for requests calls with basic error handling."""
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend at {url}. Is it running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The backend might be busy or unresponsive.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", e.response.text)
        st.error(f"HTTP Error {e.response.status_code}: {detail}")
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected error occurred: {e}")
    return None

def fetch_stats():
    """Fetches and updates stats in session state."""
    stats_data = safe_request("get", f"{BACKEND_URL}/stats/")
    if stats_data:
        st.session_state.current_stats = stats_data
    else:
        st.session_state.current_stats = None
    if st.session_state.current_stats and "llm_models_available" in st.session_state.current_stats:
        models = st.session_state.current_stats["llm_models_available"]
        if st.session_state.selected_model is None or st.session_state.selected_model not in models:
            if models: st.session_state.selected_model = models[0]
    elif st.session_state.selected_model is None:
        st.session_state.selected_model = "N/A"


def fetch_history():
    """Fetches and updates chat history."""
    history_data = safe_request("get", f"{BACKEND_URL}/history/")
    if history_data is not None:
        st.session_state.messages = history_data

def clear_chat_and_data():
    """Clears backend data and resets frontend state."""
    response_data = safe_request("post", f"{BACKEND_URL}/clear/")
    if response_data:
        st.session_state.messages = []
        st.session_state.current_stats = None
        st.session_state.file_uploader_key += 1
        st.session_state.upload_status_msg = "Chat history and indexed data cleared."
        fetch_stats()
        st.rerun()
    else:
        st.error("Failed to clear chat data on the backend.")


if st.session_state.current_stats is None:
    fetch_stats()


with st.sidebar:
    st.header("⚙️ Config")

    available_models = ["N/A"]
    current_selection_index = 0
    if st.session_state.current_stats and "llm_models_available" in st.session_state.current_stats:
        available_models = st.session_state.current_stats.get("llm_models_available", ["N/A"])
        if st.session_state.selected_model in available_models:
            current_selection_index = available_models.index(st.session_state.selected_model)
        elif available_models:
            st.session_state.selected_model = available_models[0]

    st.session_state.selected_model = st.selectbox(
        "Select LLM Model:",
        options=available_models,
        index=current_selection_index,
        key="model_select"
    )

    st.header("📄 Upload Context")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    if uploaded_files:
        if st.button("Process Files", key="process_button"):
            api_files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            if api_files:
                with st.spinner("Processing files... (Large files may take time)"):
                    upload_response = safe_request("post", f"{BACKEND_URL}/upload/", files=api_files)
                    if upload_response:
                        st.session_state.upload_status_msg = upload_response.get("message", "Processing complete.")
                        if "file_statuses" in upload_response:
                            st.expander("File Processing Details").json(upload_response["file_statuses"])
                        fetch_stats()
                        st.success(st.session_state.upload_status_msg)
                    else:
                        st.session_state.upload_status_msg = "File processing failed. Check logs or backend status."
                        st.error(st.session_state.upload_status_msg)
            else:
                st.warning("No valid files selected.")

    if st.session_state.upload_status_msg:
        if "failed" in st.session_state.upload_status_msg.lower() or "error" in st.session_state.upload_status_msg.lower():
            st.warning(st.session_state.upload_status_msg)
        else:
            st.info(st.session_state.upload_status_msg)


    st.header("📊 Stats")
    if st.button("Refresh Stats", key="refresh_stats"):
        fetch_stats()

    if st.session_state.current_stats:
        stats = st.session_state.current_stats
        if "error" in stats:
            st.error(stats["error"])
        else:
            col1, col2 = st.columns(2)
            col1.metric("Query Count", stats.get("query_count", "N/A"))
            col2.metric("Avg Proc. Time (s)", stats.get("avg_processing_time_per_query", "N/A"))
            with st.expander("More Stats"):
                st.metric("Avg Query Chars", stats.get("avg_query_length", "N/A"))
                st.metric("Avg Response Chars", stats.get("avg_response_length", "N/A"))
                st.metric("Total Proc. Time (s)", stats.get("total_processing_time", "N/A"))
                st.caption(f"Embedding: {stats.get('embedding_model', 'N/A')}")

            st.subheader("Context Files Status")
            uploaded_files_backend = stats.get("uploaded_files", [])
            if uploaded_files_backend:
                status_lines = [f"- {f['name']}: {f['status']}" for f in uploaded_files_backend]
                st.caption("\n".join(status_lines))
            else:
                st.caption("No documents processed by backend yet.")
    else:
        st.caption("Stats unavailable. Fetching or backend issue.")


    st.header("🗑️ Manage")
    if st.button("Clear Chat & Context", key="clear_button"):
        clear_chat_and_data()


st.header("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about the documents..."):
    st.chat_message("user").markdown(prompt)

    if not st.session_state.selected_model or st.session_state.selected_model == "N/A":
        st.error("Please select a valid LLM model from the sidebar.")
    else:
        payload = {"query": prompt, "model_name": st.session_state.selected_model}
        with st.spinner(f"Asking {st.session_state.selected_model}..."):
            start_time = time.time()
            chat_response = safe_request("post", f"{BACKEND_URL}/chat/", params=payload)
            frontend_time = time.time() - start_time

        if chat_response and "answer" in chat_response:
            with st.chat_message("assistant"):
                st.markdown(chat_response["answer"])
                st.caption(f"Model: {chat_response.get('model_used', 'N/A')} | Backend Time: {chat_response.get('processing_time', 0.0):.2f}s | Network+FE: {frontend_time:.2f}s")

            fetch_history()
            fetch_stats()
        else:
            st.warning("Could not get response from backend.")
