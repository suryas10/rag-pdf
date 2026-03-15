"""Chat UI components."""

import streamlit as st
from typing import List, Dict


def render_chat_history(history: List[Dict]):
    for turn in history:
        with st.chat_message("user"):
            st.write(turn.get("query", ""))
        with st.chat_message("assistant"):
            st.write(turn.get("response", ""))


def render_chat_input() -> str:
    return st.chat_input("Ask a question about your documents...")
