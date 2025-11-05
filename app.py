"""
Streamlit UI for RAG PDF system.
Handles file uploads, displays ingestion progress, and provides chat interface.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import streamlit as st
import requests
import time
import json
from typing import List, Dict
import os

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="WiM RAG PDF Chat",
    page_icon="",
    layout="wide"
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None


def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_file(file):
    """Upload PDF file to backend"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None


def get_ingestion_status(job_id: str):
    """Get ingestion job status"""
    try:
        response = requests.get(f"{API_BASE_URL}/ingestion/status/{job_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None


def query_rag(query: str, file_id: str = None, use_coref: bool = True, use_intent: bool = True):
    """Query the RAG system"""
    try:
        payload = {
            "query": query,
            "file_id": file_id,
            "use_coref": use_coref,
            "use_intent": use_intent
        }
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error querying: {str(e)}")
        return None


def display_source(source: Dict, index: int):
    """Display a source chunk with metadata"""
    with st.expander(f"Source {index + 1} (Page {source.get('metadata', {}).get('page_no', '?')}, Score: {source.get('score', 0):.2f})"):
        st.text(source.get("text", ""))
        metadata = source.get("metadata", {})
        if metadata:
            st.caption(f"Chunk ID: {metadata.get('chunk_id', 'N/A')}")
            st.caption(f"File: {metadata.get('filename', 'N/A')}")


def main():
    """Main Streamlit app"""
    st.title("WiM RAG PDF Chat System")
    st.markdown("Upload PDFs, ingest them, and chat with your documents using AI")
    
    # Check API health
    if not check_api_health():
        st.error(f"⚠️ Cannot connect to API at {API_BASE_URL}. Please ensure the FastAPI server is running.")
        st.info("Start the server with: `python fastapi_server.py`")
        return
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload Your Files")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to ingest and query"
        )
        
        if uploaded_file is not None:
            if st.button("Upload & Ingest", type="primary"):
                with st.spinner("Uploading file..."):
                    result = upload_file(uploaded_file)
                    if result:
                        job_id = result.get("job_id")
                        st.session_state.uploaded_files.append({
                            "filename": result.get("filename"),
                            "job_id": job_id,
                            "status": "processing"
                        })
                        st.success(f"File uploaded successfully!")
                        st.rerun()
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.header("Uploaded Files")
            for idx, file_info in enumerate(st.session_state.uploaded_files):
                with st.container():
                    st.text(file_info["filename"])
                    job_id = file_info.get("job_id")
                    if job_id:
                        status_info = get_ingestion_status(job_id)
                        if status_info:
                            status = status_info.get("status", "unknown")
                            progress = status_info.get("progress", 0.0)
                            
                            if status == "processing":
                                st.progress(progress)
                                st.caption(status_info.get("message", "Processing..."))
                            elif status == "completed":
                                st.success("✓ Ingested")
                                file_id = status_info.get("file_id")
                                if file_id:
                                    file_info["file_id"] = file_id
                                    file_info["status"] = "completed"
                                    if st.button(f"Select", key=f"select_{idx}"):
                                        st.session_state.current_file_id = file_id
                                        st.rerun()
                            elif status == "error":
                                st.error(f"✗ Error: {status_info.get('message', 'Unknown error')}")
        
        # Settings
        st.header("⚙️ Settings")
        use_coref = st.checkbox("Use Coreference Resolution", value=True)
        use_intent = st.checkbox("Use Intent Classification", value=True)
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with Your Documents")
    
    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for turn in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.write(turn["query"])
            
            with st.chat_message("assistant"):
                st.write(turn["response"])
                
                # Display sources if available
                if turn.get("sources"):
                    st.markdown("**Sources:**")
                    for idx, source in enumerate(turn["sources"]):
                        display_source(source, idx)
                
                # Display metadata
                if turn.get("intent"):
                    st.caption(f"Intent: {turn['intent']}")
                if turn.get("resolved_query") and turn["resolved_query"] != turn["query"]:
                    st.caption(f"Resolved query: {turn['resolved_query']}")
    
    # Query input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user message to history
        st.session_state.conversation_history.append({
            "query": query,
            "response": "",
            "sources": [],
            "intent": "",
            "resolved_query": ""
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Query backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_rag(
                    query,
                    file_id=st.session_state.current_file_id,
                    use_coref=use_coref,
                    use_intent=use_intent
                )
                
                if result:
                    response = result.get("response", "")
                    sources = result.get("sources", [])
                    intent = result.get("intent", "")
                    resolved_query = result.get("resolved_query", "")
                    
                    # Update conversation history
                    st.session_state.conversation_history[-1].update({
                        "response": response,
                        "sources": sources,
                        "intent": intent,
                        "resolved_query": resolved_query
                    })
                    
                    # Display response
                    st.write(response)
                    
                    # Display sources
                    if sources:
                        st.markdown("**Sources:**")
                        for idx, source in enumerate(sources):
                            display_source(source, idx)
                    
                    # Display metadata
                    if intent:
                        st.caption(f"Intent: {intent}")
                    if resolved_query and resolved_query != query:
                        st.caption(f"Resolved query: {resolved_query}")
                else:
                    st.error("Failed to get response from API")
                    st.session_state.conversation_history.pop()
        
        st.rerun()


if __name__ == "__main__":
    main()

