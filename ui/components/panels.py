"""Panels for retrieved context, sources, and metadata."""

from typing import List, Dict
import streamlit as st


def render_retrieved_context(items: List[Dict]):
    with st.expander("Retrieved Context", expanded=False):
        for idx, item in enumerate(items):
            metadata = item.get("metadata", {})
            item_type = metadata.get("type", "text")
            score = item.get("score", 0)
            page = metadata.get("page_no", "?")
            st.markdown(f"**{idx + 1}. {item_type.upper()} — Page {page} — Score {score:.2f}**")
            if item_type == "text":
                st.write(item.get("text", ""))
            if item_type == "image" and metadata.get("image_path"):
                st.image(metadata["image_path"], caption=metadata.get("image_path"))


def render_sources(items: List[Dict]):
    st.subheader("Sources")
    for idx, item in enumerate(items):
        metadata = item.get("metadata", {})
        item_type = metadata.get("type", "text")
        with st.expander(f"Source {idx + 1} ({item_type})"):
            filename = metadata.get("filename") or metadata.get("source_file") or "unknown"
            page_no = metadata.get("page_no", "?")
            st.caption(f"File: {filename} | Page: {page_no}")
            st.json(metadata)
            if item_type == "text":
                st.write(item.get("text", ""))
            if item_type == "image" and metadata.get("image_path"):
                st.image(metadata["image_path"], caption=metadata.get("image_path"))


def render_image_panel(items: List[Dict]):
    images = [i for i in items if i.get("metadata", {}).get("type") == "image"]
    if not images:
        return
    st.subheader("Image Preview")
    for item in images:
        metadata = item.get("metadata", {})
        if metadata.get("image_path"):
            st.image(metadata["image_path"], caption=f"Page {metadata.get('page_no', '?')} — {metadata.get('image_path')}")


def render_debug_tabs(items: List[Dict], stats: Dict, debug_events: List[str]):
    tabs = st.tabs(["Logs / Debug", "Raw Retrieved Chunks", "Metadata Inspector"])
    with tabs[0]:
        for event in debug_events[-100:]:
            st.text(event)
        if stats:
            st.json(stats)

    with tabs[1]:
        st.json(items)

    with tabs[2]:
        for idx, item in enumerate(items):
            st.markdown(f"**Item {idx + 1}**")
            st.json(item.get("metadata", {}))
