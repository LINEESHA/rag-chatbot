# RAG Chatbot using Local LLM, FAISS, and Streamlit

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions from uploaded PDF documents using semantic search and a locally hosted Large Language Model (LLM).

The system combines document retrieval with generative AI to provide context-aware responses without relying on external paid APIs. It leverages vector embeddings, FAISS indexing, and local inference to enable secure and cost-effective document question answering.

---

## Key Features

* PDF document upload and processing
* Automatic text extraction and chunking
* Semantic vector embedding generation
* FAISS-based vector database for efficient retrieval
* Retrieval-Augmented Generation (RAG) pipeline
* Local LLM inference (Mistral/Llama compatible)
* Interactive Streamlit user interface
* Fully local deployment without API dependency
* Scalable architecture for enterprise knowledge-base applications

---

## System Architecture

```text
PDF Upload
     │
     ▼
Text Extraction
     │
     ▼
Document Chunking
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Store
     │
     ▼
User Query
     │
     ▼
Similarity Search
     │
     ▼
Relevant Context Retrieval
     │
     ▼
Local LLM
     │
     ▼
Generated Response
```

---

## Project Structure

```text
rag-chatbot/
│
├── rag_chatbot_app.py          # Main Streamlit application
├── inference_api.py           # Local inference service
├── llm_finetuner.py           # Fine-tuning pipeline
├── train.py                   # Model training script
├── requirements.txt           # Project dependencies
├── FINETUNED_LLM_README.md    # Fine-tuning documentation
└── README.md                  # Project documentation
```

---

## Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### AI/ML Components

* Hugging Face Transformers
* Sentence Transformers
* Local LLM (Mistral / Llama)
* Retrieval-Augmented Generation (RAG)

### Vector Database

* FAISS

### Document Processing

* PyPDF2 / PDF Processing Libraries

---

## Installation

### Clone Repository

```bash
git clone https://github.com/LINEESHA/rag-chatbot.git
cd rag-chatbot
```

### Create Virtual Environment

```bash
python -m venv env
```

Activate environment:

**Windows**

```bash
env\Scripts\activate
```

**Linux/Mac**

```bash
source env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit interface:

```bash
streamlit run rag_chatbot_app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## Workflow

### Document Ingestion

1. Upload PDF documents.
2. Extract document text.
3. Split text into manageable chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in FAISS index.

### Query Processing

1. User submits a question.
2. Query embedding is generated.
3. Similar document chunks are retrieved from FAISS.
4. Retrieved context is passed to the Local LLM.
5. LLM generates a context-aware response.

---

## Use Cases

* Enterprise Knowledge Assistants
* Internal Documentation Search
* Research Paper Analysis
* Legal Document Q&A
* Educational Content Retrieval
* Customer Support Knowledge Bases

---

## Future Enhancements

* Multi-document knowledge base support
* Chat history memory
* Hybrid search (Keyword + Semantic Search)
* Role-based authentication
* REST API deployment
* Docker containerization
* Cloud deployment support
* Advanced evaluation metrics

---

## Author

**Lineesha S**

AI/ML Engineer | NLP | Generative AI | Computer Vision

LinkedIn: https://linkedin.com/in/lineesha-s

---

## License

This project is intended for educational, research, and internal enterprise use.
