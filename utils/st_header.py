import pandas as pd
import streamlit as st

from app import ABOUT_THIS_APP, CHUNK_EXPLAINER, EMBEDDING_MODEL_LIST, LLM_LIST


def st_header():
    """
    Creates a header UI component for the Streamlit application.

    This function uses Streamlit's built-in functions to display a header and a markdown text on the app's page.
    It's a simple way to add a consistent header to your Streamlit application.
    """
    col1, _, col2 = st.columns([0.6, 0.1, 0.3])

    col1.markdown("### 1. Upload your PDF 📄 \n For this demo, a 10-20 page PDF is recommended")
    uploaded_file = col1.file_uploader("Upload your PDF", label_visibility="collapsed", type='pdf')

    col1.markdown("### 2. Configure your RAG (Optional) 🔧")
    col1a, col1b, col1c = col1.columns(3)
    st.session_state["chunk_size"] = col1a.number_input("Chunk Size", value=1000, step=50)
    st.session_state["chunk_overlap"] = col1b.number_input("Chunk Overlap", step=50)
    st.session_state["embedding_model"] = col1a.selectbox("Select your embedding model", EMBEDDING_MODEL_LIST)
    st.session_state["llm_model"] = col1b.selectbox("Select your LLM (Coming Soon)", LLM_LIST)

    col1.markdown("### 3. Build the VectorDB ⚡️")
    if col1.button("Build"):
        st.session_state['document'] = uploaded_file
        st.rerun()

    with col2.expander("**About this application**"):
        st.success(ABOUT_THIS_APP)
    with col2.expander("**EXPLAINER:** What does chunk size/overlap mean?"):
        st.info(CHUNK_EXPLAINER)
