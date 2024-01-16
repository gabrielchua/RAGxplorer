import streamlit as st
import plotly.graph_objs as go
from constants import VISUALISATION_SETTINGS

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
                
    return st.plotly_chart(fig, use_container_width=True)