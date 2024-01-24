"""
Embedding Functions
"""
import os

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction, 
    OpenAIEmbeddingFunction
    )

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

all_MiniLM_L6_v2 = SentenceTransformerEmbeddingFunction()

text_embedding_ada_002 = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002")
