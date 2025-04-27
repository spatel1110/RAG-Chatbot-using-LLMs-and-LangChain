# RAG Chatbot with Ollama LangChain FastAPI & Streamlit

This project implements a Retrieval-Augmented Generation (RAG) chatbot that runs locally using open-source LLMs managed by Ollama. It features a Python backend built with FastAPI and LangChain, and a user-friendly frontend using Streamlit.

<img width="1439" alt="Screenshot 2025-04-26 at 9 46 26 PM" src="https://github.com/user-attachments/assets/718a5b74-0454-43ec-970a-68d16393bd2d" />

## Overview

The goal of this project is to provide a context-aware chat experience by leveraging documents provided by the user. The chatbot can ingest information from various file types (PDF, DOCX, TXT), store it efficiently using a vector database (FAISS), and use this knowledge to answer user queries accurately. Users can switch between different locally hosted language models via Ollama.

## Features

* **Retrieval-Augmented Generation (RAG):** Answers questions based on the content of uploaded documents.
* **Local Open-Source LLMs:** Integrates with models running locally via Ollama (e.g., Mistral 7B, Llama 3 8B, Phi-3 Mini).
* **Multiple Model Support:** Allows switching between configured LLMs during a chat session.
* **Multi-File Upload:** Supports uploading PDF, DOCX, and TXT files. (Note: PDF uses PyPDFLoader due to previous dependency compatibility issues with UnstructuredLoader).
* **Chat History:** Stores and displays the conversation history for the current session.
* **Vector Store:** Uses FAISS (CPU) for efficient document embedding storage and retrieval.
* **Usage Statistics:** Displays basic statistics like query count, average processing time, etc.
* **Web Interface:** Simple and clean UI built with Streamlit for interaction.

## Tech Stack

* **Backend:** Python, FastAPI, LangChain, Uvicorn
* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain
* **LLM Serving:** Ollama
* **Models Used (Example):** Mistral 7B, Llama 3 8B (configurable via `.env`)
* **Vector Store:** FAISS (CPU)

## Architecture

The application follows a simple client-server architecture:

1.  **Frontend (Streamlit):** Provides the user interface for uploading files, chatting, selecting models, and viewing stats. It communicates with the backend API.
2.  **Backend (FastAPI):** Exposes API endpoints for file processing, chat interaction, and fetching data (history, stats).
3.  **Ollama:** Runs the open-source LLMs locally, serving requests from the backend.
4.  **FAISS:** Stores vector embeddings of document chunks locally for fast retrieval.

## Getting Started

### Prerequisites

* **Python:** Version 3.10 or higher recommended.
* **Ollama:** Install the Ollama macOS application from [https://ollama.com/](https://ollama.com/). Ensure the Ollama application is running.
* **Git:** For cloning the repository.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/spatel1110/RAG-Chatbot-using-LLMs-and-LangChain.git
    cd rag-chatbot-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt

### Ollama Setup

1.  **Ensure Ollama is running.** (Check for the menu bar icon on macOS).
2.  **Pull the desired LLMs:** Open your terminal and run:
    ```bash
    ollama pull mistral
    ollama pull llama3
    ```
3.  **Verify models are available:**
    ```bash
    ollama list
    ```

### Configuration

1.  Navigate to the `backend` directory: `cd backend`
2.  Create a `.env` file by copying the example or creating it manually:
    ```dotenv
    MODEL_1="mistral"
    MODEL_2="llama3"
    EMBEDDING_MODEL="mistral"
    VECTORSTORE_PATH="../vectorstore/faiss_index"
    ```
3.  Adjust the model names and paths as needed. Ensure the models listed are pulled via Ollama.
4.  Go back to the project root directory: `cd ..`

### Running the Application

1.  **Start the Backend (FastAPI):**
    Open a terminal in the project root directory.
    ```bash
    # Ensure venv is active
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  **Start the Frontend (Streamlit):**
    Open a *new* terminal in the project root directory.
    ```bash
    # Ensure venv is active
    cd frontend
    streamlit run app.py
    ```
