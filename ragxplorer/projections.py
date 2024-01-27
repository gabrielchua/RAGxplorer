"""
This module provides functionalities for projecting high-dimensional data (embeddings)
into a lower-dimensional space for visualization, using UMAP (Uniform Manifold Approximation and Projection).
"""

import os
from typing import Tuple, List

import numpy as np
import umap
import pandas as pd
import plotly.graph_objs as go
from tqdm import tqdm

from .constants import VISUALISATION_SETTINGS, PLOT_SIZE

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def set_up_umap(embeddings: np.ndarray) -> umap.UMAP:
    """
    Sets up and fits a UMAP transformer to the given embeddings.

    Args:
        embeddings (np.ndarray): An array of embeddings to fit the UMAP transformer.

    Returns:
        umap.UMAP: A fitted UMAP transformer.
    """
    umap_transform = umap.UMAP().fit(embeddings)
    return umap_transform

def get_projections(embedding: np.ndarray, umap_transform: umap.UMAP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects embeddings into a two-dimensional space using the provided UMAP transformer.

    Args:
        embedding (np.ndarray): An array of embeddings to project.
        umap_transform (umap.UMAP): A fitted UMAP transformer.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y coordinates of the projected embeddings.
    """
    projections = _project_embeddings(embedding, umap_transform)
    x = projections[:, 0]
    y = projections[:, 1]
    return x, y

def _project_embeddings(embeddings: np.ndarray, umap_transform: umap.UMAP) -> np.ndarray:
    """
    Helper function to project embeddings using a UMAP transformer.

    Args:
        embeddings (np.ndarray): An array of embeddings to project.
        umap_transform (umap.UMAP): A fitted UMAP transformer.

    Returns:
        np.ndarray: An array of projected embeddings.
    """
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in tqdm(enumerate(embeddings), total=len(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def prepare_projections_df(document_ids: List[str], document_projections: Tuple[np.ndarray, np.ndarray], document_text: List[str]) -> pd.DataFrame:
    """
    Prepares a DataFrame for visualization from document IDs, projections, and texts.

    Args:
        document_ids (List[str]): List of document IDs.
        document_projections (Tuple[np.ndarray, np.ndarray]): Tuple of X and Y coordinates of document projections.
        document_text (List[str]): List of document texts.

    Returns:
        pd.DataFrame: DataFrame containing the information for visualization.
    """
    df = pd.DataFrame({"id": document_ids,
                       "x": document_projections[0],
                       "y": document_projections[1]})
    
    df['document'] = document_text
    df['document_cleaned'] = df.document.str.wrap(80).apply(lambda x: x.replace('\n', '<br>'))
    df['size'] = PLOT_SIZE
    df['category'] = "Chunks"
    
    return df

def plot_embeddings(df: pd.DataFrame) -> go.Figure:
    """
    Creates a Plotly figure to visualize the embeddings.

    Args:
        df (pd.DataFrame): DataFrame containing the data to visualize.

    Returns:
        go.Figure: A Plotly figure object for visualization.
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
        legend=dict(
            y=100,
            x=0.5,
            xanchor='center',
            yanchor='top',
            orientation='h'
        )
    )
                
    return fig
