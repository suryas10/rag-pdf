"""Sidebar layout and controls."""

from typing import Dict, Optional
import streamlit as st
from ui.api_client import APIClient


def render_sidebar(api: APIClient) -> Dict:
    with st.sidebar:
        st.header("Control Panel")

        health = api.get_health()
        st.subheader("System Status")
        if health and health.get("components"):
            components = health["components"]
            st.success("API: online")
            st.caption(f"Vector DB: {'ok' if components.get('vector_store') else 'down'}")
            st.caption(f"Text Embeddings: {'ok' if components.get('text_embedder') else 'down'}")
            st.caption(f"Image Embeddings: {'ok' if components.get('vision_embedder') else 'down'}")
        else:
            st.error("API: offline")

        st.divider()

        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_uploader")

        chunk_size = st.number_input("Chunk size", min_value=200, max_value=1200, value=500, step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=300, value=50, step=10)

        use_multimodal = st.toggle("Multimodal retrieval", value=True)

        top_k = st.slider("Retrieval top-k", min_value=1, max_value=10, value=5)

        image_query_file = None
        if use_multimodal:
            image_query_file = st.file_uploader("Optional query image", type=["png", "jpg", "jpeg"], key="image_query")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload & Ingest", type="primary", use_container_width=True) and uploaded_file is not None:
                st.session_state.last_uploaded_file = {
                    "name": uploaded_file.name,
                    "bytes": uploaded_file.getvalue()
                }
                with st.spinner("Uploading and starting ingestion..."):
                    result = api.upload_file(
                        filename=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        include_images=use_multimodal
                    )
                    if result:
                        st.session_state.uploaded_files.append({
                            "filename": result.get("filename"),
                            "job_id": result.get("job_id"),
                            "status": "processing"
                        })
                        st.success("Ingestion started.")
                        st.rerun()
                    else:
                        st.error("Upload failed. Check API logs.")

        with col2:
            if st.button("Re-ingest", use_container_width=True) and st.session_state.last_uploaded_file:
                last = st.session_state.last_uploaded_file
                with st.spinner("Re-ingesting last file..."):
                    result = api.upload_file(
                        filename=last["name"],
                        file_bytes=last["bytes"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        include_images=use_multimodal
                    )
                    if result:
                        st.session_state.uploaded_files.append({
                            "filename": result.get("filename"),
                            "job_id": result.get("job_id"),
                            "status": "processing"
                        })
                        st.success("Re-ingestion started.")
                        st.rerun()
                    else:
                        st.error("Re-ingest failed.")

        st.divider()
        st.subheader("Index Management")
        if st.button("Clear index", use_container_width=True):
            if api.clear_index():
                st.session_state.uploaded_files = []
                st.session_state.current_file_id = None
                st.success("Index cleared.")
                st.rerun()
            else:
                st.error("Failed to clear index.")

        if st.button("Clear DB + memory", use_container_width=True):
            if api.reset():
                st.session_state.uploaded_files = []
                st.session_state.current_file_id = None
                st.session_state.conversation_history = []
                st.session_state.retrieved_items = []
                st.session_state.last_query_stats = {}
                st.success("Database and memory cleared.")
                st.rerun()
            else:
                st.error("Failed to clear database and memory.")

        if st.session_state.current_file_id:
            if st.button("Delete selected file", use_container_width=True):
                if api.clear_conversation(st.session_state.current_file_id):
                    st.session_state.uploaded_files = [
                        f for f in st.session_state.uploaded_files
                        if f.get("file_id") != st.session_state.current_file_id
                    ]
                    st.session_state.current_file_id = None
                    st.session_state.conversation_history = []
                    st.session_state.retrieved_items = []
                    st.session_state.last_query_stats = {}
                    st.success("Selected file deleted. Memory and vectors cleared.")
                    st.rerun()
                else:
                    st.error("Failed to delete selected file.")

        st.divider()
        st.subheader("Uploaded Files")
        for idx, file_info in enumerate(st.session_state.uploaded_files):
            st.text(file_info.get("filename", ""))
            job_id = file_info.get("job_id")
            if job_id:
                status_info = api.get_ingestion_status(job_id)
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
                            if st.button("Select", key=f"select_{idx}"):
                                st.session_state.current_file_id = file_id
                                st.rerun()
                    elif status == "error":
                        st.error(status_info.get("message", "Error"))

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "use_multimodal": use_multimodal,
            "top_k": top_k,
            "image_query_file": image_query_file
        }
