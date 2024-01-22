"""
Module for streamlit related functions
"""
import streamlit as st

def st_initilize_session_state_as_none(key_list):
    """
    Initialise the session state for the st app
    """
    for key in key_list:
        if key not in st.session_state:
            st.session_state[key] = None

# UI Component
def st_header():
    """
    UI component: header
    """
    st.header("RAGxplorer 🗺️", divider='grey')
    st.markdown("#### Visualise which chunks are most relevant to your query.")

