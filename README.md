# RAGxplorer

<img src="images/logo.png" width="200">

RAGxplorer is an interactive tool for visualizing document chunks in the embedding space, designed to diagnose and explore applications of the Retriever-Answer Generator (RAG) model.

## Overview

RAGxplorer allows users to upload documents, convert them into chunked formats suitable for RAG applications, and visualize these chunks in an embedding space. This visualization aids in understanding how different chunks relate to each other and to specific queries, thereby providing insights into the workings of RAG-based systems.

<img src="images/example.png" width="650">

## Features

- **Document Upload**: Users can upload PDF documents.
- **Chunk Configuration**: Options to configure the chunk size and overlap
- **Choice of embedding model**: `all-MiniLM-L6-v2` or `text-embedding-ada-002`
- **Vector Database Creation**: Builds a vector database using Chroma
- **Query Expansion**: Generates sub-questions and hypothetical answers to enhance the retrieval process.
- **Interactive Visualization**: Utilizes Plotly to visualise the chunks.

## Installation

To run RAGxplorer, ensure you have Python installed, and then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Setup `OPENAI_API_KEY` (required) and `ANYSCALE_API_KEY` (if you need anyscale). Copy
    the `.streamlit/secrets.example.toml` file to `.streamlit/secrets.toml` and fill in the values.
2. To start the application, run:

```bash
streamlit run app.py
```

## Contributing

Contributions to RAGxplorer are welcome. Please read our [contributing guidelines (WIP)](.github/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT license - see the LICENSE file for details.

## Acknowledgments
- DeepLearning.AI and Chroma for the inspiration and foundational code.
- The Streamlit community for the support and resources.
