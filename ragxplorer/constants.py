"""
constants.py

This module defines constants used throughout the ragxplorer package.
These include settings for embeddings, query expansion, and visualization.
"""

# Embedding models available for use
OPENAI_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

# Prompts for Query Expansion
MULTIPLE_QNS_SYS_MSG = ("Given a question, your task is to generate 3 to 5 simple sub-questions related to the original question. "
                        "These sub-questions are to be short. Format your reply in json with numbered keys. "
                        "EXAMPLE: "
                        "INPUT: What are the top 3 reasons for the decline in Microsoft's revenue from 2022 to 2023? "
                        "OUTPUT: {'1': 'What was microsoft's revenue in 2022?', '2': 'What was microsoft's revenue in 2023?', '3': What are the drivers for revenue?}")

HYDE_SYS_MSG = ("Given a question, your task is to craft a template for the hypothetical answer. Do not include any facts, and instead label them as <PLACEHOLDERS>. "
                "FOR EXAMPLE: INPUT: 'What is the revenue of microsoft in 2021 and 2022?' "
                "OUTPUT: 'Microsoft's 2021 and 2022 revenue is <MONETARY SUM> and <MONETARY SUM> respectively.'")

# Constants for plots
PLOT_SIZE = 3

# Settings for data visualization
VISUALISATION_SETTINGS = {
    'Original Query': {'color': 'red', 'opacity': 1, 'symbol': 'cross', 'size': 15},
    'Retrieved': {'color': 'green', 'opacity': 1, 'symbol': 'circle', 'size': 10},
    'Chunks': {'color': 'blue', 'opacity': 0.4, 'symbol': 'circle', 'size': 10},
    'Sub-Questions': {'color': 'purple', 'opacity': 1, 'symbol': 'star', 'size': 15},
    'Hypothetical Ans': {'color': 'purple', 'opacity': 1, 'symbol': 'star', 'size': 15},
}
