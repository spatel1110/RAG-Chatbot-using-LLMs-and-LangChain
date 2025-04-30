import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from core.rag_service import RAGService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="RAG Chatbot API")
origins = ["http://localhost", "http://localhost:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    rag_service = RAGService()
except ConnectionError as e:
    logging.fatal(f"Failed to initialize RAGService: {e}. API cannot start.")
    rag_service = None


@app.post("/upload/", summary="Upload files for RAG context")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accepts files, passes them to the RAG service for processing and indexing.
    Uses tempfile internally via RAGService.
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service initialization failed. Check backend logs.")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    try:
        chunks_added = rag_service.process_uploaded_files(files)
        file_statuses = rag_service.uploaded_files_info
        return {
            "message": f"Processed {len(files)} files. Added {chunks_added} chunks to vector store.",
            "file_statuses": file_statuses
        }
    except Exception as e:
        logging.error(f"Error during file upload processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during file processing: {str(e)}")


@app.post("/chat/", summary="Send a query to the chatbot")
async def chat_endpoint(
    query: str = Query(..., min_length=1, description="User's query"),
    model_name: str = Query(..., description="LLM to use (e.g., 'mistral', 'llama3')")
):
    """
    Forwards the query and selected model to the RAG service. Handles expected
    service errors and returns appropriate HTTP status codes.
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service unavailable.")

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = rag_service.process_query(query, model_name)

        if "error" in result:
            error_message = result["error"]
            status_code = 500

            if "Unsupported model" in error_message or "Vector store not available" in error_message:
                status_code = 400
            elif "Could not connect" in error_message or "Service initialization failed" in error_message:
                status_code = 503

            raise HTTPException(status_code=status_code, detail=error_message)

        return result

    except HTTPException as http_exc:
        raise http_exc
    except ConnectionError as conn_err:
        logging.error(f"Connection error during chat processing: {conn_err}", exc_info=False)
        raise HTTPException(status_code=503, detail=str(conn_err))
    except ValueError as val_err:
        logging.warning(f"Value error during chat processing: {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logging.error(f"Unexpected error in /chat endpoint processing query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")


@app.get("/history/", summary="Get the current chat history")
async def get_history_endpoint() -> List[Dict[str, str]]:
    """Retrieves chat history from the RAG service."""
    if rag_service is None: return []
    return rag_service.get_chat_history()

@app.get("/stats/", summary="Get usage statistics")
async def get_stats_endpoint() -> Dict[str, Any]:
    """Retrieves current stats from the RAG service."""
    if rag_service is None:
        return { "query_count": "N/A", "error": "RAG Service unavailable."}
    return rag_service.get_current_stats()

@app.post("/clear/", summary="Clear chat history and vector data")
async def clear_endpoint():
    """Triggers data clearing in the RAG service."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service unavailable.")

    try:
        rag_service.clear_chat_data()
        return {"message": "Chat history, vector store, and related data cleared."}
    except Exception as e:
        logging.error(f"Error during data clearing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")


if __name__ == "__main__":
    logging.info("Starting FastAPI server via script (use 'uvicorn main:app --reload' for development)...")
    import uvicorn
    if rag_service:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        logging.fatal("Cannot start server: RAGService failed to initialize.")
