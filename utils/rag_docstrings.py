utils/rag_docstrings.py:

"""
rag_docstrings.py

This module contains docstrings for the functions in rag.py.

Functions:
    build_vector_database(file, chunk_size, chunk_overlap, embedding_model)
    _split_text_into_chunks(pdf_texts, chunk_size, chunk_overlap)
    _split_chunks_into_tokens(character_split_texts)
    _create_and_populate_chroma_collection(token_split_texts, embedding_model)
    query_chroma(chroma_collection, query, top_k)
    get_doc_embeddings(chroma_collection)
    get_docs(chroma_collection)
    get_embedding(text)
    _load_pdf(file)
"""
