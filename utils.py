import random
import string
import chromadb
import umap
import plotly.express as px
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    SentenceTransformersTokenTextSplitter
    )
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def build_vector_database(file, chunk_size, chunk_overlap):
    pdf_texts = _load_pdf(file)

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    chroma_client = chromadb.Client()
    document_name = _generate_random_string(10)
    chroma_collection = chroma_client.create_collection(document_name, embedding_function=SentenceTransformerEmbeddingFunction())

    ids = [str(i) for i in range(len(token_split_texts))]

    chroma_collection.add(ids=ids, documents=token_split_texts)

    return chroma_collection

def query_chroma(chroma_collection, query, top_k):
    results = chroma_collection.query(query_texts=[query], n_results=top_k, include=['documents', 'embeddings'])
    retrieved_id = results['ids'][0]
    return retrieved_id

def get_doc_embeddings(chroma_collection):
    embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
    return embeddings

def get_docs(chroma_collection):
    documents = chroma_collection.get(include=['documents'])['documents']
    return documents

def set_up_umap(embeddings):
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    return umap_transform

def get_projections(embedding, umap_transform):
    projections = _project_embeddings(embedding, umap_transform)
    x = projections[:, 0]
    y = projections[:, 1]
    return x, y

def get_embedding(text):
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")([text])

def _project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(embeddings): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings   

def _load_pdf(file):
    pdf = PdfReader(file)
    pdf_texts = [p.extract_text().strip() for p in pdf.pages]
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts

def _generate_random_string(length):
    characters = string.ascii_letters
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string

