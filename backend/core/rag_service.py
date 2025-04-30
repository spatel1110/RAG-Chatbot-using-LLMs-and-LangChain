import os
import shutil
import logging
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class RAGService:
    """
    Manages the RAG pipeline including document processing,
    vector store interaction, and chat generation.
    """
    def __init__(self):
        self.vector_store_path = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
        self.model_1 = os.getenv("MODEL_1", "mistral")
        self.model_2 = os.getenv("MODEL_2", "llama3")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "mistral")

        self.vector_store: Optional[FAISS] = None
        self.chat_history: List[Dict[str, str]] = []
        self.uploaded_files_info: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = self._reset_stats()

        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self._initialize_embeddings()
        self._load_existing_vectorstore()

    def _reset_stats(self) -> Dict[str, Any]:
        return {"query_count": 0, "total_query_length": 0, "total_response_length": 0, "total_processing_time": 0.0}

    def _initialize_embeddings(self):
        try:
            self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
            logging.info(f"Ollama embeddings initialized with model: {self.embedding_model_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize embeddings: {e}")

    def _get_llm(self, model_name: str) -> ChatOllama:
        try:
            llm = ChatOllama(model=model_name)
            logging.info(f"Loaded LLM: {model_name}")
            return llm
        except Exception as e:
            raise ConnectionError(f"Could not connect to Ollama or load model '{model_name}'. Ensure Ollama is running and the model is pulled.")

    def _load_existing_vectorstore(self):
        if os.path.exists(self.vector_store_path):
            try:
                logging.info(f"Loading existing vector store from {self.vector_store_path}...")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logging.info("Existing vector store loaded.")
            except Exception as e:
                logging.warning(f"Failed to load existing vector store: {e}. Will create a new one if documents are uploaded.")
                self.vector_store = None
        else:
            logging.info("No existing vector store found.")
            self.vector_store = None

    def process_uploaded_files(self, files: List[Any]):
        temp_files_paths = []
        docs = []
        self.uploaded_files_info.clear()

        for file in files:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                    shutil.copyfileobj(file.file, tmp_file)
                    temp_files_paths.append(tmp_file.name)
                    logging.info(f"Temporarily saved {file.filename} to {tmp_file.name}")
            except Exception as e:
                logging.error(f"Error saving file {file.filename}: {e}")
                self.uploaded_files_info.append({"name": file.filename, "status": f"Error saving: {e}"})
            finally:
                if hasattr(file, 'file') and not file.file.closed:
                    file.file.close()

        for file_path in temp_files_paths:
            file_name = os.path.basename(file_path.split('_', 1)[1])
            try:
                if file_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                elif file_path.lower().endswith(".docx"):
                    try:
                        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                        loader = UnstructuredWordDocumentLoader(file_path, mode="single", strategy="fast")
                        loaded_docs = loader.load()
                    except ImportError:
                        logging.warning(f"Cannot load DOCX {file_name}: UnstructuredWordDocumentLoader not available. Install 'unstructured' and 'python-docx'.")
                        self.uploaded_files_info.append({"name": file_name, "status": "Skipped (DOCX loader missing)"})
                        continue
                    except Exception as docx_e:
                        logging.error(f"Error loading DOCX file {file_name} with Unstructured: {docx_e}")
                        self.uploaded_files_info.append({"name": file_name, "status": f"Error loading DOCX: {docx_e}"})
                        continue

                elif file_path.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                    loaded_docs = loader.load()
                else:
                    logging.warning(f"Unsupported file type skipped: {file_name}")
                    self.uploaded_files_info.append({"name": file_name, "status": "Unsupported Type"})
                    continue

                docs.extend(loaded_docs)
                self.uploaded_files_info.append({"name": file_name, "status": "Loaded"})
                logging.info(f"Successfully loaded {len(loaded_docs)} document(s) from {file_name}")

            except Exception as e:
                logging.error(f"Error loading file {file_name} from {file_path}: {e}")
                self.uploaded_files_info.append({"name": file_name, "status": f"Error loading: {e}"})

        for path in temp_files_paths:
            try:
                os.remove(path)
            except OSError as e:
                logging.error(f"Error removing temp file {path}: {e}")

        if not docs:
            logging.warning("No documents were successfully loaded from the uploaded files.")
            return 0

        logging.info(f"Splitting {len(docs)} loaded document sections...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        logging.info(f"Created {len(split_docs)} document chunks.")

        if not split_docs:
            logging.warning("No chunks were generated after splitting.")
            return 0

        self._create_or_update_vectorstore(split_docs)
        return len(split_docs)

    def _create_or_update_vectorstore(self, docs_to_add: List[Document]):
        if not docs_to_add: return
        logging.info(f"Updating vector store with {len(docs_to_add)} chunks.")
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs_to_add, self.embeddings)
            logging.info("Created a new vector store.")
        else:
            self.vector_store.add_documents(docs_to_add)
            logging.info(f"Added {len(docs_to_add)} chunks to the existing vector store.")
        try:
            if self.vector_store:
                self.vector_store.save_local(self.vector_store_path)
                logging.info(f"Vector store saved to {self.vector_store_path}")
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")

    def _setup_rag_chain(self, llm, retriever):
        system_prompt = (
            "Use the following context to answer the question concisely. "
            "If the answer isn't in the context, say so."
            "\n\nContext:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [ ("system", system_prompt), ("human", "{input}"), ]
        )
        Youtube_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, Youtube_chain)
        return rag_chain

    def process_query(self, query: str, model_name: str) -> Dict[str, Any]:
        start_time = time.time()
        if self.vector_store is None:
            self._load_existing_vectorstore()
            if self.vector_store is None:
                logging.warning("Query processed but vector store is not initialized. Upload documents first.")
                return {"error": "Vector store not available. Please upload documents."}
        try:
            llm = self._get_llm(model_name)
            retriever = self.vector_store.as_retriever(search_kwargs={'k': 4})
            rag_chain = self._setup_rag_chain(llm, retriever)
            response = rag_chain.invoke({"input": query})
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"Query processed in {processing_time:.2f}s using {model_name}")
            self.stats["query_count"] += 1
            self.stats["total_query_length"] += len(query)
            self.stats["total_response_length"] += len(answer)
            self.stats["total_processing_time"] += processing_time
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            return { "answer": answer, "processing_time": processing_time, "model_used": model_name, }
        except Exception as e:
            logging.error(f"Error during query processing: {e}")
            if isinstance(e, ConnectionError): return {"error": str(e)}
            return {"error": f"An internal error occurred while processing the query."}

    def clear_chat_data(self):
        logging.info("Clearing chat data...")
        self.vector_store = None
        self.chat_history = []
        self.uploaded_files_info = []
        self.stats = self._reset_stats()
        if os.path.exists(self.vector_store_path):
            try:
                shutil.rmtree(self.vector_store_path)
                logging.info(f"Removed persistent vector store at {self.vector_store_path}")
            except Exception as e:
                logging.error(f"Error removing vector store directory: {e}")
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)

    def get_current_stats(self) -> Dict[str, Any]:
        count = self.stats["query_count"]
        if count == 0: return {**self.stats, "avg_query_length": 0, "avg_response_length": 0, "avg_processing_time_per_query": 0, "llm_models_available": [self.model_1, self.model_2], "embedding_model": self.embedding_model_name, "uploaded_files": self.uploaded_files_info}
        avg_proc_time = self.stats["total_processing_time"] / count
        avg_query_len = self.stats["total_query_length"] / count
        avg_resp_len = self.stats["total_response_length"] / count
        return { "query_count": count, "avg_query_length": round(avg_query_len, 2), "avg_response_length": round(avg_resp_len, 2), "total_processing_time": round(self.stats["total_processing_time"], 2), "avg_processing_time_per_query": round(avg_proc_time, 2), "llm_models_available": [self.model_1, self.model_2], "embedding_model": self.embedding_model_name, "uploaded_files": self.uploaded_files_info }

    def get_chat_history(self) -> List[Dict[str, str]]:
        return self.chat_history
