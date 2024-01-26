"""
Module for projections
"""
from typing import Tuple

import numpy as np
import umap
import streamlit as st 
import pandas as pd
import plotly.graph_objs as go
from tqdm import tqdm

from .constants import (
    VISUALISATION_SETTINGS,
    PLOT_SIZE
    )

def set_up_umap(embeddings: np.ndarray) -> umap.UMAP:
    """
    Sets up and fits a UMAP transformer to the embeddings.
    
    Args:
        embeddings: An array of embeddings to fit the UMAP transformer.
    
    Returns:
        A fitted UMAP transformer.
    """
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    return umap_transform

def get_projections(embedding: np.ndarray, umap_transform: umap.UMAP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects embeddings into a two-dimensional space using UMAP.
    
    Args:
        embedding: An array of embeddings to project.
        umap_transform: A fitted UMAP transformer.
    
    Returns:
        A tuple of x and y coordinates of the projected embeddings.
    """
    projections = _project_embeddings(embedding, umap_transform)
    x = projections[:, 0]
    y = projections[:, 1]
    return x, y

def _project_embeddings(embeddings: np.ndarray, umap_transform: umap.UMAP) -> np.ndarray:
    """
    Helper function to project embeddings using UMAP.
    
    Args:
        embeddings: An array of embeddings to project.
        umap_transform: A fitted UMAP transformer.
    
    Returns:
        An array of projected embeddings.
    """
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in tqdm(enumerate(embeddings)):
        if len(embedding)  == 1:
            embedding = np.array(embedding)
            embedding = embedding.reshape(1, -1)
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def prepare_projections_df(document_projections, document_text):
    df = pd.DataFrame({"x": document_projections[0],
                    "y": document_projections[1]})
    
    df = df.assign(document=document_text)

    df['document_cleaned'] = df.document.str.wrap(80).apply(lambda x: x.replace('\n', '<br>'))

    df = df.assign(size=PLOT_SIZE)
    df = df.assign(category="Chunks")
    
    return df

def plot_embeddings(df):
    # Create a figure
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

    # Set the layout, including moving the legend to the top
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
