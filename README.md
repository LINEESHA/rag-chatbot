# RAG Document Q&A Chatbot

## Overview

RAG Document Q&A Chatbot is a Streamlit-based application that allows users to upload TXT and PDF documents and ask questions based on the uploaded content. The application uses a Retrieval-Augmented Generation workflow with a local LLM powered by Ollama.

This project is designed to run locally without depending on paid external APIs.

## Features

* Upload TXT and PDF documents
* Extract text from uploaded files
* Split documents into configurable chunks
* Ask questions through an interactive chat interface
* Generate answers using a local Ollama LLM
* Supports models such as Mistral, Llama2, and Neural Chat
* Adjustable chunk size and Top-K retrieval settings
* Chat history using Streamlit session state
* Simple local deployment

## Tech Stack

* Python
* Streamlit
* Ollama
* Local LLM
* PyPDF
* Requests

## Project Structure

```text
rag-chatbot/
│
├── rag_chatbot_app.py      # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── other scripts           # Training or inference-related files
```

## How It Works

```text
User uploads TXT/PDF files
        ↓
Text is extracted from documents
        ↓
Text is divided into chunks
        ↓
Top-K chunks are selected as context
        ↓
User asks a question
        ↓
Context + question are sent to Ollama
        ↓
Local LLM generates the final answer
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/LINEESHA/rag-chatbot.git
cd rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv env
```

### 3. Activate Virtual Environment

For Windows:

```bash
env\Scripts\activate
```

For Linux/Mac:

```bash
source env/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Requirements

Make sure Ollama is installed and running locally.

Default Ollama URL:

```text
http://localhost:11434
```

Pull a supported model before running the app:

```bash
ollama pull mistral
```

Optional models:

```bash
ollama pull llama2
ollama pull neural-chat
```

## Run the Application

```bash
streamlit run rag_chatbot_app.py
```

After running, open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Application Flow

1. Start the Streamlit application.
2. Configure the Ollama URL from the sidebar.
3. Select the required LLM model.
4. Upload TXT or PDF documents.
5. Adjust chunk size and Top-K retrieval value if needed.
6. Ask questions in the chat box.
7. The app sends document context and user query to Ollama.
8. The local LLM generates a response based on the uploaded document content.

## Current Retrieval Logic

The current implementation uses a simple chunk-based retrieval flow. Uploaded documents are split into chunks, and the top chunks are passed as context to the LLM.

This can be upgraded further using:

* FAISS vector search
* Sentence embeddings
* Similarity score thresholding
* Hybrid search
* Reranking
* Source citation with page numbers

## Future Enhancements

* Add FAISS vector database integration
* Add semantic embedding-based retrieval
* Add similarity score filtering to reduce hallucination
* Display source page numbers
* Support multiple document indexing
* Add conversation memory
* Add better error handling for missing Ollama models
* Add Docker support
* Deploy as an internal document assistant

## Use Cases

* PDF question answering
* Internal document search
* Research document assistant
* Company knowledge-base chatbot
* Local AI assistant without API cost

## Author

Lineesha S

AI/ML Engineer | NLP | Generative AI | Computer Vision

## License

This project is for educational, research, and internal development purposes.
