# RAGxplorer 🦙🦺 

<img src="images/logo.png" width="200">

RAGxplorer is an interactive streamlit tool to support the building of Retrieval Augmented Generation (RAG) applications by visualizing document chunks and the queries in the embedding space. 

## Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-xplorer.streamlit.app/)

_Due to [infra limitations](https://discuss.streamlit.io/t/is-there-streamlit-app-limitations-such-as-usage-time-users-etc/42800), this freely hosted demo may occassionaly go down. The best experience is to clone this repo, and run it locally._

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
