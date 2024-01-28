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

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenAIEmbeddingFunction,
    HuggingFaceEmbeddingFunction
    )

import plotly.graph_objs as go

from .rag import (
    build_vector_database,
    get_doc_embeddings,
    get_docs,
    query_chroma
    )

from .projections import (
    set_up_umap,
    get_projections,
    prepare_projections_df,
    plot_embeddings
    )

from .query_expansion import (
    generate_hypothetical_ans,
    generate_sub_qn
    )

from .constants import OPENAI_EMBEDDING_MODELS


class _Documents(BaseModel):
    text: Optional[Any] = None
    ids: Optional[Any] = None
    embeddings: Optional[Any] = None
    projections: Optional[Any] = None

class _Query(BaseModel):
    original_query: Optional[Any] = None
    original_query_projection: Optional[Any] = None
    actual_search_queries: Optional[Any] = None
    retrieved_docs: Optional[Any] = None

class _VizData(BaseModel):
    base_df: Optional[Any] = None
    query_df: Optional[Any] = None
    visualisation_df: Optional[Any] = None

class RAGxplorer(BaseModel):
    """
    RAGxplorer class for managing the RAG exploration process.
    """
    embedding_model: Optional[str] = Field(default="all-MiniLM-L6-v2")
    _chosen_embedding_model: Optional[Any] = None
    _vectordb: Optional[Any] = None
    _documents: _Documents = _Documents()
    _projector: Optional[Any] = None
    _query: _Query = _Query()
    _VizData: _VizData = _VizData()

    def __init__(self, **data):
        super().__init__(**data)
        self._set_embedding_model()

    def _set_embedding_model(self):
        """ Sets the embedding model """
        if self.embedding_model == 'all-MiniLM-L6-v2':
            self._chosen_embedding_model = SentenceTransformerEmbeddingFunction()

        elif self.embedding_model in OPENAI_EMBEDDING_MODELS:
            if "OPENAI_API_KEY" not in os.environ:
                raise OSError("OPENAI_API_KEY is not set")
            self._chosen_embedding_model = OpenAIEmbeddingFunction(api_key = os.getenv("OPENAI_API_KEY"), 
                                                                   model_name = self.embedding_model)
        else:
            try:
                if "HF_API_KEY" not in os.environ:
                    raise OSError("HF_API_KEY is not set")
                self._chosen_embedding_model = HuggingFaceEmbeddingFunction(api_key = os.getenv("HF_API_KEY"),
                                                                            model_name = self.embedding_model)
            except Exception as exc:
                raise ValueError("Invalid embedding model. Please use all-MiniLM-L6-v2, or a valid OpenAI or HuggingFace embedding model.") from exc

    def load_pdf(self, document_path: str, chunk_size: int = 1000, chunk_overlap: int = 0, verbose: bool = False):
        """
        Load data from a PDF file and prepare it for exploration.
        
        Args:
            document: Path to the PDF document to load.
            chunk_size: Size of the chunks to split the document into.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        if verbose:
            print(" ~ Building the vector database...")
        self._vectordb = build_vector_database(document_path, chunk_size, chunk_overlap, self._chosen_embedding_model)
        if verbose:
            print("Completed Building Vector Database ✓")
        self._documents.embeddings = get_doc_embeddings(self._vectordb)
        self._documents.text = get_docs(self._vectordb)
        self._documents.ids = self._vectordb.get()['ids']
        if verbose:
            print(" ~ Reducing the dimensionality of embeddings...")
        self._projector = set_up_umap(embeddings=self._documents.embeddings)
        self._documents.projections = get_projections(embedding=self._documents.embeddings,
                                                      umap_transform=self._projector)
        self._VizData.base_df = prepare_projections_df(document_ids=self._documents.ids,
                                                                document_projections=self._documents.projections,
                                                                document_text=self._documents.text)
        if verbose:
            print("Completed reducing dimensionality of embeddings ✓")

    def visualize_query(self, query: str, retrieval_method: str="naive", top_k:int=5, query_shape_size:int=5) -> go.Figure:
        """
        Visualize the query results in a 2D projection using Plotly.

        Args:
            query (str): The query string to visualize.
            retrieval_method (str): The method used for document retrieval. Defaults to 'naive'.
            top_k (int): The number of top documents to retrieve.
            query_shape_size (int): The size of the shape to represent the query in the plot.

        Returns:
            go.Figure: A Plotly figure object representing the visualization.

        Raises:
            RuntimeError: If the document has not been loaded before visualization.
        """
        if self._vectordb is None or self._VizData.base_df is None:
            raise RuntimeError("Please load the pdf first.")
        
        if retrieval_method not in ["naive", "HyDE", "multi_qns"]:
            raise ValueError("Invalid retrieval method. Please use naive, HyDE, or multi_qns.")

        self._query.original_query = query

        if (self.embedding_model == "all-MiniLM-L6-v2") or (self.embedding_model in OPENAI_EMBEDDING_MODELS):
            self._query.original_query_projection = get_projections(embedding=self._chosen_embedding_model(self._query.original_query),
                                                                umap_transform=self._projector)
        else:
            self._query.original_query_projection = get_projections(embedding=[self._chosen_embedding_model(self._query.original_query)],
                                                                    umap_transform=self._projector)

        self._VizData.query_df = pd.DataFrame({"x": [self._query.original_query_projection[0][0]],
                                      "y": [self._query.original_query_projection[1][0]],
                                      "document_cleaned": query,
                                      "category": "Original Query",
                                      "size": query_shape_size})

        if retrieval_method == "naive":
            self._query.actual_search_queries = self._query.original_query

        elif retrieval_method == "HyDE":
            if "OPENAI_API_KEY" not in os.environ:
                raise OSError("OPENAI_API_KEY is not set")
            self._query.actual_search_queries = generate_hypothetical_ans(query=self._query.original_query)

        elif retrieval_method == "multi_qns":
            if "OPENAI_API_KEY" not in os.environ:
                raise OSError("OPENAI_API_KEY is not set")
            self._query.actual_search_queries = generate_sub_qn(query=self._query.original_query)

        self._query.retrieved_docs = query_chroma(chroma_collection=self._vectordb,
                                            query=self._query.actual_search_queries,
                                            top_k=top_k)

        self._VizData.base_df.loc[self._VizData.base_df['id'].isin(self._query.retrieved_docs), "category"] = "Retrieved"
            
        self._VizData.visualisation_df = pd.concat([self._VizData.base_df, self._VizData.query_df], axis = 0)

        return plot_embeddings(self._VizData.visualisation_df)
