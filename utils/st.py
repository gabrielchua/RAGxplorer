"""
st.py

This module contains utility functions for initializing and managing the session state in a Streamlit application, 
as well as creating UI components.

Functions:
    st_initialize_session_state_as_none(key_list)
    st_reset_application()
    st_header()
"""
from typing import List

import streamlit as st

def st_initialize_session_state_as_none(key_list: List[str]) -> None:
    """
    Initializes specified keys in the Streamlit session state to None if they are not already set.

    This function is useful for ensuring that certain keys exist in the session state before they are used in the app.

    Args:
        key_list (List[str]): A list of keys to initialize in the session state.
    """
    for key in key_list:
        if key not in st.session_state:
            st.session_state[key] = None

def st_reset_application() -> None:
    """
    Resets specific elements of the Streamlit application's session state and reruns the app.

    This function sets certain keys in the session state to their initial values and triggers a rerun of the app.
    It is useful for providing a reset mechanism within the application.
    """
    st.session_state['document'] = None
    st.session_state["document_projections_done"] = False
    st.experimental_rerun()  # Note: 'st.rerun()' is deprecated in favor of 'st.experimental_rerun()'

def st_header() -> None:
    """
    Creates a header UI component for the Streamlit application.
    
    This function uses Streamlit's built-in functions to display a header and a markdown text on the app's page.
    It's a simple way to add a consistent header to your Streamlit application.
    """
    
    def st_header() -> None:
        """
        Creates a header UI component for the Streamlit application.
    
        This function uses Streamlit's built-in functions to display a header and a markdown text on the app's page.
        It's a simple way to add a consistent header to your Streamlit application.
        """
        st.header("RAGxplorer 🦙🦺", divider='grey')
    st.header("RAGxplorer 🦙🦺", divider='grey')