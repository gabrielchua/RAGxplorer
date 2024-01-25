# tests/test_app.py

from unittest.mock import MagicMock

import pytest

from app import (build_vector_database,  # Add any other necessary imports
                 get_doc_embeddings, get_docs, get_embedding, get_projections,
                 prepare_projections_df, query_chroma, set_up_umap,
                 st_initialize_session_state_as_none, st_reset_application)


# Test st_initialize_session_state_as_none function
def test_st_initialize_session_state_as_none():
    session_state = {}
    st_initialize_session_state_as_none(session_state)
    assert session_state == {
        "document": None,
        "chroma": None,
        "filtered_df": None,
        "document_projections": None,
        "document_projections_done": False
    }

# Test st_reset_application function
def test_st_reset_application():
    session_state = {
        "document": "example.pdf",
        "chroma": MagicMock(),
        "filtered_df": MagicMock(),
        "document_projections": MagicMock(),
        "document_projections_done": True
    }
    st_reset_application(session_state)
    assert session_state == {
        "document": None,
        "chroma": None,
        "filtered_df": None,
        "document_projections": None,
        "document_projections_done": False
    }

# Test build_vector_database function
def test_build_vector_database():
    # Add test cases for building the VectorDB

# Test query_chroma function
def test_query_chroma():
    # Add test cases for querying the VectorDB

# Test get_embedding function
def test_get_embedding():
    # Add test cases for getting embeddings

# Test get_docs function
def test_get_docs():
    # Add test cases for getting document information

# Test get_doc_embeddings function
def test_get_doc_embeddings():
    # Add test cases for getting document embeddings

# Test set_up_umap function
def test_set_up_umap():
    # Add test cases for setting up UMAP

# Test get_projections function
def test_get_projections():
    # Add test cases for getting projections

# Test prepare_projections_df function
def test_prepare_projections_df():
    # Add test cases for preparing projections dataframe

# Add more test cases for other functions as needed
