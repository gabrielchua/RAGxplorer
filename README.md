# RAGxplorer 🦙🦺 

<img src="images/logo.png" width="200">

RAGxplorer is an interactive streamlit tool to support the building of Retrieval Augmented Generation (RAG) applications by visualizing document chunks and the queries in the embedding space. 

> [!NOTE]
> This app was for a [Streamlit competition](https://www.linkedin.com/posts/streamlit_ragxplorer-explore-the-embeddings-of-activity-7154217315566321664-6-d6), and was very scrapily put together one evening.
>
> I am re-factoring the code to be a package. You can view the progress in the `experiment` branch. Installation and usage details can be found [here](#using-the-experimental-version).
>
> Until then, I appreciate your patience, and suggestions will be most welcomed [here](https://github.com/gabrielchua/RAGxplorer/issues/3).

## Demo 🔎
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-xplorer.streamlit.app/)

⚠️ _Due to [infra limitations](https://discuss.streamlit.io/t/is-there-streamlit-app-limitations-such-as-usage-time-users-etc/42800), this freely hosted demo may occasionally go down. The best experience is to clone this repo, and run it locally._

<img src="images/example.png" width="650">

## Features ✨

- **Document Upload**: Users can upload PDF documents.
- **Chunk Configuration**: Options to configure the chunk size and overlap
- **Choice of embedding model**: `all-MiniLM-L6-v2`, `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Vector Database Creation**: Builds a vector database using Chroma
- **Query Expansion**: Generates sub-questions and hypothetical answers to enhance the retrieval process.
- **Interactive Visualization**: Utilizes Plotly to visualise the chunks.

## Option 1 💻
### Installation ⚙️
To run RAGxplorer, ensure you have Python installed, and then install the necessary dependencies:

```bash
pip install -r requirements-local-deployment.txt
```

> [!TIP]
> ⚠️ Do not use `requirements.txt`. That is so the free streamlit deployment can run. That file includes an additional `pysqlite3-binary` dependency.
> 
> ⚠️ If it helps with troubleshooting, this application was built using Python 3.11

### Usage 🏎️

1. Setup `OPENAI_API_KEY` (required) and `ANYSCALE_API_KEY` (if you need anyscale). Copy
    the `.streamlit/secrets.example.toml` file to `.streamlit/secrets.toml` and fill in the values.
2. To start the application, run:
    ```bash
    streamlit run app.py
    ```

> [!NOTE]
> This repo is currently linked to the streamlit demo, and these lines were added due to the runtime in the free streamlit deployment env. See [here](https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950).

## Option 2: Docker 🐳

To run the project using Docker, run the following command:

```bash
docker-compose up -d
```

Once the image is built and the container is running, you can access the application at `http://localhost:8501`.

## Using the experimental version 🧪

### Installation
Enter the following in your terminal
```bash
git clone -b experiment https://github.com/gabrielchua/RAGxplorer.git
cd RAGxplorer
virtualenv venv # create a new virtual env
source venv/bin/activate # activate the virtual env
pip install -r requirements.txt
```

### Usage 
```python
from ragxplorer.ragxplorer import Explorer
client = Explorer(embedding_model="text-embedding-ada-002") # Please ensure "OPENAI_API_KEY" is set as an env variable
client.load_document("presentation.pdf")
client.visualise_query("What are the top revenue drivers for Microsoft?")
```

## Contributing 👋

Contributions to RAGxplorer are welcome. Please read our [contributing guidelines (WIP)](.github/CONTRIBUTING.md) for details.

## License 👀

This project is licensed under the MIT license - see the LICENSE file for details.

## Acknowledgments 💙
- DeepLearning.AI and Chroma for the inspiration and code labs in their [Advanced Retrival](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) course.
- The Streamlit community for the support and resources.
