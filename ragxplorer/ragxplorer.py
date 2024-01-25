"""
Ragxplorer.py
"""
import os
from typing import (
    Optional,
    Any
    )

from pydantic import BaseModel, Field
import pandas as pd

from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
    
from .embedding import (
    all_MiniLM_L6_v2,
    text_embedding_ada_002
)

from .rag import (
    build_vector_database,
    get_doc_embeddings,
    get_docs,
    query_chroma,
)

from .projections import (
    set_up_umap,
    get_projections,
    prepare_projections_df,
    plot_embeddings
)

class RAGxplorer(BaseModel):
    """
    Explorer class for managing the RAG exploration process.
    """
    embedding_model: Optional[str] = Field(default="all-MiniLM-L6-v2")
    _chosen_embedding_model: Optional[Any] = None
    _client: Optional[Any] = None
    _documents_embeddings: Optional[Any] = None
    _documents: Optional[Any] = None
    _projector: Optional[Any] = None
    _documents_projections: Optional[Any] = None
    _original_query: Optional[Any] = None
    _original_query_projection: Optional[Any] = None
    _actual_search_queries: Optional[Any] = None
    _retrieved_docs: Optional[Any] = None
    _base_df: Optional[Any] = None
    _query_df: Optional[Any] = None
    _visualisation_df: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_embedding_model()

    def _set_embedding_model(self):
        if self.embedding_model == 'all-MiniLM-L6-v2':
            self._chosen_embedding_model = all_MiniLM_L6_v2
        elif self.embedding_model == 'text-embedding-ada-002':
            self._chosen_embedding_model = text_embedding_ada_002
        else:
            try:
                self._chosen_embedding_model = HuggingFaceEmbeddingFunction(api_key = os.getenv("HF_API_KEY"), model_name = self.embedding_model)
            except:
                raise ValueError("Invalid embedding model. Please use all-MiniLM-L6-v2, text-embedding-ada-002, or a valid hugging face model")

    def load_document(self, document, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Load data from a PDF file and prepare it for exploration.
        
        Args:
            document: Path to the PDF document to load.
            chunk_size: Size of the chunks to split the document into.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        self._client = build_vector_database(document, chunk_size, chunk_overlap, self._chosen_embedding_model)
        self._documents_embeddings = get_doc_embeddings(self._client)
        self._documents = get_docs(self._client)
        self._projector = set_up_umap(self._documents_embeddings)
        self._documents_projections = get_projections(self._documents_embeddings, self._projector)
        self._base_df = prepare_projections_df(self._documents_projections, self._documents)

    def visualise_query(self, query, retrieval_method="naive", top_k=5, plot_size=5):
        """
        Visualize the query results in a 2D projection using Plotly.
        
        Args:
            query: The query string to visualize.
            retrieval_method: The method used for document retrieval.
            top_k: The number of top documents to retrieve.
            plot_size: The size of the plot to generate.
        
        Returns:
            A Plotly figure object representing the visualization.
        
        Raises:
            RuntimeError: If the document has not been loaded before visualization.
        """
        if self.client is None or self.base_df is None:
            raise RuntimeError("Please load the document first.")

        self._original_query = query
        self._original_query_projection = get_projections(self._chosen_embedding_model(self._original_query), self._projector)

        self._query_df = pd.DataFrame({"x": [self._original_query_projection[0][0]],
                                      "y": [self._original_query_projection[1][0]],
                                      "document_cleaned": query,
                                      "category": "Original Query",
                                      "size": plot_size})

        self._actual_search_queries = self._original_query

        self._retrieved_docs = query_chroma(self.client,
                                          self.actual_search_queries,
                                          top_k)

        self._retrieved_docs = [int(index) for index in self.retrieved_docs]

        self._base_df.loc[self._retrieved_docs, "category"] = "Retrieved"

        self._visualisation_df = pd.concat([self._base_df, self._query_df], axis = 0)

        return plot_embeddings(self._visualisation_df)
