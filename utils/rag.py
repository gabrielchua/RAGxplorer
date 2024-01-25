"""
rag.py

This module contains functions for RAG (Retrieval Augmented Generation) - namely building and querying a 
vector database using ChromaDB, utilizing various embedding models. It includes utilities for loading 
PDF documents and chunking it.

Functions:
    build_vector_database(file, chunk_size, chunk_overlap, embedding_model)
    query_chroma(chroma_collection, query, top_k)
    get_doc_embeddings(chroma_collection)
    get_docs(chroma_collection)
    get_embedding(text)
    _load_pdf(file)
    _split_text_into_chunks(pdf_texts, chunk_size, chunk_overlap)
    _split_chunks_into_tokens(character_split_texts)
    _create_and_populate_chroma_collection(token_split_texts, embedding_model)
"""
import uuid
from typing import (
    List,
    Any,
    Sequence
    )

import chromadb
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenAIEmbeddingFunction
)

def build_vector_database(file: Any, chunk_size: int, chunk_overlap: int, embedding_model: str) -> chromadb.Collection:
    """
    Builds a vector database from a PDF file by splitting the text into chunks and embedding them.
    
    Args:
        file: The PDF file to process.
        chunk_size: The number of tokens in one chunk.
        chunk_overlap: The number of tokens shared between consecutive chunks.
    
    Returns:
        A Chroma collection object containing the embedded chunks.
    """
    pdf_texts = _load_pdf(file)
    character_split_texts = _split_text_into_chunks(pdf_texts, chunk_size, chunk_overlap)
    token_split_texts = _split_chunks_into_tokens(character_split_texts)
    chroma_collection = _create_and_populate_chroma_collection(token_split_texts, embedding_model)
    return chroma_collection

def _split_text_into_chunks(pdf_texts: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits the text from a PDF into chunks based on character count.
    
    Args:
        pdf_texts: List of text extracted from PDF pages.
        chunk_size: The number of tokens in one chunk.
        chunk_overlap: The number of tokens shared between consecutive chunks.
    
    Returns:
        A list of text chunks.
    """
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return character_splitter.split_text('\n\n'.join(pdf_texts))

def _split_chunks_into_tokens(character_split_texts: List[str]) -> List[str]:
    """
    Splits text chunks into smaller chunks based on token count.
    
    Args:
        character_split_texts: List of text chunks split by character count.
    
    Returns:
        A list of text chunks split by token count.
    """
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    return [text for chunk in character_split_texts for text in token_splitter.split_text(chunk)]

def _create_and_populate_chroma_collection(token_split_texts: List[str], embedding_model) -> chromadb.Collection:
    """
    Creates a Chroma collection and populates it with the given text chunks.
    
    Args:
        token_split_texts: List of text chunks split by token count.
    
    Returns:
        A Chroma collection object populated with the text chunks.
    """
    chroma_client = chromadb.Client()
    document_name = uuid.uuid4().hex
    if embedding_model == "all-MiniLM-L6-v2":
        chroma_collection = chroma_client.create_collection(document_name, embedding_function=SentenceTransformerEmbeddingFunction())
    elif embedding_model == "text-embedding-ada-002":
        openai_ef = OpenAIEmbeddingFunction(
                api_key=st.secrets['OPENAI_API_KEY'],
                model_name="text-embedding-ada-002")
        chroma_collection = chroma_client.create_collection(document_name, embedding_function=openai_ef)
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)
    return chroma_collection

def query_chroma(chroma_collection: chromadb.Collection, query: str, top_k: int) -> List[str]:
    """
    Queries the Chroma collection for the top_k most relevant chunks to the input query.
    
    Args:
        chroma_collection: The Chroma collection to query.
        query: The input query string.
        top_k: The number of top results to retrieve.
    
    Returns:
        A list of retrieved chunk IDs.
    """
    results = chroma_collection.query(query_texts=[query], n_results=top_k, include=['documents', 'embeddings'])
    retrieved_id = [int(index) for index in results['ids'][0]]
    return retrieved_id

def get_doc_embeddings(chroma_collection: chromadb.Collection) -> list[Sequence[float] | Sequence[int]] | None:
    """
    Retrieves the document embeddings from the Chroma collection.
    
    Args:
        chroma_collection: The Chroma collection to retrieve embeddings from.
    
    Returns:
        An array of embeddings.
    """
    embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
    return embeddings

def get_docs(chroma_collection: chromadb.Collection) -> list[str] | None:
    """
    Retrieves the documents from the Chroma collection.
    
    Args:
        chroma_collection: The Chroma collection to retrieve documents from.
    
    Returns:
        A list of documents.
    """
    documents = chroma_collection.get(include=['documents'])['documents']
    return documents

def get_embedding(model:str, text: str) -> list[Sequence[float] | Sequence[int]]:
    """
    Generates an embedding for the given text using a sentence transformer model.
    
    Args:
        text: The text to embed.
    
    Returns:
        An embedding of the text.
    """
    if model == "text-embedding-ada-002":
        return OpenAIEmbeddingFunction(
                api_key=st.secrets['OPENAI_API_KEY'],
                model_name="text-embedding-ada-002"
            )([text])
    else:
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")([text])

def _load_pdf(file: Any) -> List[str]:
    """
    Loads and extracts text from a PDF file.
    
    Args:
        file: The PDF file to load.
    
    Returns:
        A list of strings, each representing the text of a page.
    """
    pdf = PdfReader(file)
    pdf_texts = [p.extract_text().strip() for p in pdf.pages if p.extract_text()]
    return pdf_texts
