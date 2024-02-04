"""
Streamlit app
"""
# Line 6 to 8 is for streamlit commmunity deployment
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

import os
import streamlit as st
import plotly.graph_objs as go
from ragxplorer import RAGxplorer

st.set_page_config(
    page_title="RAGxplorer Demo",
    page_icon="🦙",
    layout="wide"
)

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['HF_API_KEY'] = st.secrets["HF_API_KEY"]

if "chart" not in st.session_state:
    st.session_state['chart'] = None

if "loaded" not in st.session_state:
    st.session_state['loaded'] = False

st.title("RAGxplorer Demo 🦙🦺")
st.markdown("📦 More details can be found at the GitHub repo [here](https://github.com/gabrielchua/RAGxplorer)")

if not st.session_state['loaded']:
    main_page = st.empty()
    main_button = st.empty()
    with main_page.container():
        uploaded_file = st.file_uploader("Upload your PDF", label_visibility="collapsed", type='pdf')
        st.session_state["embedding_model_type"] = st.radio("Select type of embedding model", ["all-MiniLM-L6-v2", "OpenAI", "HuggingFace"], horizontal=True)

        if st.session_state["embedding_model_type"] == "OpenAI":
            st.session_state["chosen_embedding_model"] = st.selectbox("Select embedding model", ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"])
        elif st.session_state["embedding_model_type"] == "HuggingFace":
            st.session_state["chosen_embedding_model"] = st.text_input("Enter HF repository name")
        else:
            st.session_state["chosen_embedding_model"] = "all-MiniLM-L6-v2"

        st.session_state["chunk_size"] = st.number_input("Chunk size", value=500, min_value=100, max_value=1000, step=100)
        st.session_state["chunk_overlap"] = st.number_input("Chunk overlap", value=0, min_value=0, max_value=100, step=10)

    if st.button("Build Vector DB"):
        st.session_state["client"] = RAGxplorer(embedding_model=st.session_state["chosen_embedding_model"])
        main_page.empty()
        main_button.empty()
        with st.spinner("Building Vector DB"):
            st.session_state["client"].load_pdf(uploaded_file, chunk_size=st.session_state["chunk_size"], chunk_overlap=st.session_state["chunk_overlap"])
            st.session_state['loaded'] = True
            st.rerun()
else:
    col1, col2 = st.columns(2)
    st.session_state['query'] = col1.text_area("Enter your query here")
    st.session_state['technique'] = col1.radio("Select retrival technique", ["naive", "HyDE", "multi_qns"], horizontal=True)
    st.session_state['top_k'] = col1.number_input("Top k", value=5, min_value=1, max_value=10, step=1)
    if col1.button("Execute Query"):
            st.session_state['chart'] = st.session_state["client"].visualize_query(st.session_state['query'], retrieval_method=st.session_state['technique'], top_k=st.session_state['top_k'])
    if st.session_state['chart'] is not None:
        col2.plotly_chart(st.session_state['chart'])

    if col1.button("Reset Application"):
        st.session_state['loaded'] = False
        st.rerun()
