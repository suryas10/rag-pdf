"""
Streamlit UI for RAG PDF system.
Control-panel layout with streaming chat, retrieval inspection, and observability.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from ui.state import init_state
from ui.api_client import APIClient
from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat_history, render_chat_input
from ui.components.panels import render_retrieved_context, render_sources, render_debug_tabs, render_image_panel


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG PDF Control Panel", layout="wide")


def main():
    init_state()
    api = APIClient(API_BASE_URL)

    st.title("WiM RAG PDF Chat System")
    st.caption("Upload PDFs, ingest them, and chat with your documents using AI.")

    if not api.check_health():
        st.error(f"Cannot connect to API at {API_BASE_URL}. Start FastAPI with: python fastapi_server.py")
        return

    settings = render_sidebar(api)

    # Main layout
    col_chat, col_context = st.columns([2, 1])
    with col_chat:
        st.subheader("Chat")
        chat_box = st.container(height=520, border=True)
        with chat_box:
            render_chat_history(st.session_state.conversation_history)

        if st.button("Clear chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.retrieved_items = []
            st.session_state.last_query_stats = {}
            st.rerun()

        query = render_chat_input()

        if query:
            payload = {
                "query": query,
                "file_id": st.session_state.current_file_id,
                "use_coref": True,
                "use_intent": True,
                "use_history": True,
                "use_multimodal": settings["use_multimodal"],
                "top_k": settings["top_k"]
            }

            if settings.get("image_query_file") is not None:
                payload["image_query_base64"] = api.encode_image_to_base64(settings["image_query_file"].getvalue())

            st.session_state.conversation_history.append({
                "query": query,
                "response": "",
                "sources": [],
                "intent": "",
                "resolved_query": ""
            })

            with chat_box:
                with st.chat_message("user"):
                    st.write(query)

                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response_text = ""

                try:
                    for event in api.query_stream(payload):
                        if event.get("type") == "token":
                            response_text += event.get("data", "")
                            response_placeholder.write(response_text)
                        if event.get("type") == "error":
                            response_placeholder.error(event.get("message", "Streaming error"))
                            break
                        if event.get("type") == "final":
                            st.session_state.retrieved_items = event.get("sources", [])
                            stats = event.get("stats", {})
                            stats["response_chars"] = len(response_text)
                            stats["response_tokens_est"] = len(response_text) // 4
                            st.session_state.last_query_stats = stats
                            st.session_state.conversation_history[-1].update({
                                "response": response_text,
                                "sources": event.get("sources", []),
                                "intent": event.get("intent", ""),
                                "resolved_query": event.get("resolved_query", "")
                            })
                            st.session_state.debug_events.append(
                                f"Query: {query} | chunks={stats.get('chunks_count')} | images={stats.get('images_count')} | ms={stats.get('retrieval_time_ms')}"
                            )
                            break
                except Exception:
                    result = api.query(payload)
                    if result:
                        response_text = result.get("response", "")
                        response_placeholder.write(response_text)
                        st.session_state.retrieved_items = result.get("sources", [])
                        stats = result.get("stats", {})
                        stats["response_chars"] = len(response_text)
                        stats["response_tokens_est"] = len(response_text) // 4
                        st.session_state.last_query_stats = stats
                        st.session_state.conversation_history[-1].update({
                            "response": response_text,
                            "sources": result.get("sources", []),
                            "intent": result.get("intent", ""),
                            "resolved_query": result.get("resolved_query", "")
                        })
                        st.session_state.debug_events.append(
                            f"Query: {query} | chunks={stats.get('chunks_count')} | images={stats.get('images_count')} | ms={stats.get('retrieval_time_ms')}"
                        )
                    else:
                        response_placeholder.error("Failed to get response from API.")

    with col_context:
        st.subheader("Retrieved Context")
        render_retrieved_context(st.session_state.retrieved_items)
        render_sources(st.session_state.retrieved_items)
        render_image_panel(st.session_state.retrieved_items)

        if st.session_state.last_query_stats:
            st.subheader("Retrieval Stats")
            st.json(st.session_state.last_query_stats)

    st.divider()
    render_debug_tabs(
        st.session_state.retrieved_items,
        st.session_state.last_query_stats,
        st.session_state.debug_events
    )

    # Auto-refresh if any files are processing
    has_processing = any(f.get("status") == "processing" for f in st.session_state.uploaded_files)
    if has_processing:
        st.autorefresh(interval=2000, key="ingest_refresh")


if __name__ == "__main__":
    main()

