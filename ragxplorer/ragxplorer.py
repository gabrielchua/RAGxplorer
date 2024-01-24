"""
ragxplorer.py
"""
import pandas as pd 
from typing import (
    Optional, 
    Any
    )

from pydantic import BaseModel, Field
import pandas as pd

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

class Explorer(BaseModel):
    """
    Ragxplorer
    """
    embedding_model: str = Field(pattern=r'^(all-MiniLM-L6-v2|text-embedding-ada-002)$', default="english")
    chosen_embedding_model: Optional[Any] = None
    client: Optional[Any] = None
    documents_embeddings: Optional[Any] = None
    documents: Optional[Any] = None
    projector: Optional[Any] = None
    documents_projections: Optional[Any] = None
    original_query: Optional[Any] = None
    original_query_projection: Optional[Any] = None
    actual_search_queries: Optional[Any] = None
    retrieved_docs: Optional[Any] = None
    base_df: Optional[Any] = None
    query_df: Optional[Any] = None
    visualisation_df: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_embedding_model()

    def _set_embedding_model(self):
        if self.embedding_model == 'all-MiniLM-L6-v2':
            self.chosen_embedding_model = all_MiniLM_L6_v2  # Assuming all_MiniLM_L6_v2 is a defined function
        elif self.embedding_model == 'text-embedding-ada-002':
            self.chosen_embedding_model = text_embedding_ada_002  # Assuming text_embedding_ada_002 is a defined function
        else:
            raise ValueError("Invalid chosen_embedding value")

    def load_document(self, document, chunk_size: int = 1000, chunk_overlap: int = 0):
        """
        Load data from a PDF file.
        """
        self.client = build_vector_database(document, chunk_size, chunk_overlap, self.chosen_embedding_model)
        self.documents_embeddings = get_doc_embeddings(self.client)
        self.documents = get_docs(self.client)
        self.projector = set_up_umap(self.documents_embeddings)
        self.documents_projections = get_projections(self.documents_embeddings, self.projector)
        self.base_df = prepare_projections_df(self.documents_projections, self.documents)

    def visualise_query(self, query, retrieval_method="naive", top_k=5, plot_size=5):
        """
        Visualize the data using Plotly based on the provided query.
        """
        if self.client is None or self.base_df is None:
            raise RuntimeError("Please load the document first.")

        self.original_query = query
        self.original_query_projection = get_projections(self.chosen_embedding_model(self.original_query), self.projector)

        self.query_df = pd.DataFrame({"x": [self.original_query_projection[0][0]],
                                      "y": [self.original_query_projection[1][0]],
                                      "document_cleaned": query,
                                      "category": "Original Query",
                                      "size": plot_size})

        self.actual_search_queries = self.original_query

        self.retrieved_docs = query_chroma(self.client,
                                          self.actual_search_queries,
                                          top_k)

        self.retrieved_docs = [int(index) for index in self.retrieved_docs]

        self.base_df.loc[self.retrieved_docs, "category"] = "Retrieved"

        self.visualisation_df = pd.concat([self.base_df, self.query_df], axis = 0)

        return plot_embeddings(self.visualisation_df)
