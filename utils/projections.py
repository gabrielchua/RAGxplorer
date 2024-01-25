"""
Projections.py

This module provides functions to perform dimensionality reduction and to visualise the projections.

Functions:
    set_up_umap(embeddings: np.ndarray) -> umap.UMAP
    get_projections(embedding: np.ndarray, umap_transform: umap.UMAP) -> Tuple[np.ndarray, np.ndarray]
    _project_embeddings(embeddings: np.ndarray, umap_transform: umap.UMAP) -> np.ndarray
    prepare_projections_df() -> pd.DataFrame
    plot_embeddings(df: pd.DataFrame) -> DeltaGenerator
"""
from typing import (
    Tuple,
    Sequence
    )

import numpy as np
import umap
import streamlit as st 
import pandas as pd
import plotly.graph_objs as go
from streamlit.delta_generator import DeltaGenerator

from utils.constants import (
    VISUALISATION_SETTINGS,
    PLOT_SIZE
    )

def set_up_umap(embeddings: np.ndarray) -> umap.UMAP:
    """
    Sets up and fits a UMAP transformer to the embeddings.

    Args:
        embeddings (np.ndarray): An array of embeddings to fit the UMAP transformer.

    Returns:
        umap.UMAP: A fitted UMAP transformer.
    """
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    return umap_transform

def get_projections(embedding: list[Sequence[float] | Sequence[int]], umap_transform: umap.UMAP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects embeddings into a two-dimensional space using UMAP.

    Args:
        embedding (np.ndarray): An array of embeddings to project.
        umap_transform (umap.UMAP): A fitted UMAP transformer.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of x and y coordinates of the projected embeddings.
    """
    projections = _project_embeddings(embedding, umap_transform)
    x = projections[:, 0]
    y = projections[:, 1]
    return x, y

def _project_embeddings(embeddings: np.ndarray, umap_transform: umap.UMAP) -> np.ndarray:
    """
    Helper function to project embeddings using UMAP.

    Args:
        embeddings (np.ndarray): An array of embeddings to project.
        umap_transform (umap.UMAP): A fitted UMAP transformer.

    Returns:
        np.ndarray: An array of projected embeddings.
    """
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(embeddings):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def prepare_projections_df() -> pd.DataFrame:
    """
    Prepares the dataframe for the plot, using Streamlit's session state.

    Returns:
        pd.DataFrame: A DataFrame containing projection coordinates, documents, and additional formatting details for plotting.
    """
    df = pd.DataFrame({"x": st.session_state["document_projections"][0], 
                       "y": st.session_state["document_projections"][1]})
    
    df = df.assign(document=st.session_state["docs"])
    df['document_cleaned'] = df.document.str.wrap(80).apply(lambda x: x.replace('\n', '<br>'))
    df = df.assign(size=PLOT_SIZE)
    df = df.assign(category="Chunks")
    
    return df

def plot_embeddings(df: pd.DataFrame) -> DeltaGenerator:
    """
    Plots the embeddings in a two-dimensional space using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing x and y coordinates, and additional data for plotting.

    Returns:
        DeltaGenerator: The function creates a plot in Streamlit and does not return any value.
    """
    """
    Plots the embeddings in a two-dimensional space using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing x and y coordinates, and additional data for plotting.

    Returns:
        None: The function creates a plot in Streamlit and does not return any value.
    """
    fig = go.Figure()

    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        settings = VISUALISATION_SETTINGS.get(category, {'color': 'grey', 'opacity': 1, 'symbol': 'circle', 'size': 10})
        fig.add_trace(go.Scatter(
            x=category_df['x'],
            y=category_df['y'],
            mode='markers',
            name=category,
            marker=dict(
                color=settings['color'],
                opacity=settings['opacity'],
                symbol=settings['symbol'],
                size=settings['size'],
                line_width=0
            ),
            hoverinfo='text',
            text=category_df['document_cleaned']
        ))

    fig.update_layout(
        height=500,
        width=500,
        legend=dict(
            y=100,
            x=0.5,
            xanchor='center',
            yanchor='top',
            orientation='h'
        )
    )
                
    return st.plotly_chart(fig)
