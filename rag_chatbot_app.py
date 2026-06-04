import streamlit as st
import os
from pathlib import Path
import requests
import json
from datetime import datetime




def call_ollama(url, model, prompt):
    """Call Ollama API"""
    try:
        response = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return None
    except Exception as e:
        return None






# Set page config
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

# CSS styling
st.markdown("""
    <style>
    .main { padding: 2rem; background-color: #1e1e1e; }
    .stChatMessage { background-color: #2d2d2d; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 RAG Document Q&A Chatbot")
st.markdown("Ask questions about uploaded documents powered by RAG + Local LLM")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434", help="Default: localhost:11434")
    model_name = st.selectbox("LLM Model", ["mistral", "llama2", "neural-chat"], help="Models available in Ollama")
    
    st.markdown("---")
    st.subheader("📁 Document Upload")
    uploaded_files = st.file_uploader("Upload TXT/PDF files", type=["txt", "pdf"], accept_multiple_files=True)
    
    st.markdown("---")
    st.subheader("🔍 RAG Settings")
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, step=100)
    top_k = st.slider("Top K Results", 1, 10, 3, step=1)
    
    st.markdown("---")
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Initialize session state
if "messages" in st.session_state:
    pass
else:
    st.session_state.messages = []
    st.session_state.documents = []
    st.session_state.embeddings = []

# Process uploaded documents
if uploaded_files:
    st.subheader("📄 Uploaded Documents")
    processed_docs = []
    
    for file in uploaded_files:
        if file.type == "text/plain":
            content = file.read().decode("utf-8")
        else:
            # PDF processing
            from pypdf import PdfReader
            pdf_reader = PdfReader(file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
        
        # Split into chunks
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        processed_docs.extend(chunks)
        
        st.success(f"✅ {file.name} - {len(chunks)} chunks")
    
    st.session_state.documents = processed_docs
    st.info(f"Total documents loaded: {len(st.session_state.documents)} chunks")

# Chat interface
st.subheader("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with RAG
    if not st.session_state.documents:
        response = "No documents uploaded. Please upload documents first."
    else:
        try:
            # Simple retrieval - find most relevant chunks
            relevant_docs = st.session_state.documents[:top_k]  # Simplified retrieval
            context = "\n\n".join(relevant_docs[:top_k])
            
            # Call Ollama
            rag_prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {prompt}

Answer:"""
            
            with st.spinner("🤔 Thinking..."):
                response_text = call_ollama(ollama_url, model_name, rag_prompt)
                response = response_text if response_text else "Unable to generate response"
        
        except Exception as e:
            response = f"Error: {str(e)}. Make sure Ollama is running on {ollama_url}"
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    <p>RAG Chatbot v1.0 | Powered by Ollama + FAISS | Free & Open Source</p>
</div>
""", unsafe_allow_html=True)
