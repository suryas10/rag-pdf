"""Session state initialization helpers."""

import streamlit as st


def init_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "current_file_id" not in st.session_state:
        st.session_state.current_file_id = None
    if "debug_events" not in st.session_state:
        st.session_state.debug_events = []
    if "retrieved_items" not in st.session_state:
        st.session_state.retrieved_items = []
    if "last_query_stats" not in st.session_state:
        st.session_state.last_query_stats = {}
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
