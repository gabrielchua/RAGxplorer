# RAGxplorer 🦙🦺

[![PyPI version](https://img.shields.io/pypi/v/ragxplorer.svg)](https://pypi.org/project/ragxplorer/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragxplorer.streamlit.app/)

<img src="https://raw.githubusercontent.com/gabrielchua/RAGxplorer/main/images/logo.png" width="200">

RAGxplorer is a tool to build Retrieval Augmented Generation (RAG) visualisations.

# Quick Start ⚡

**Installation**

```bash
pip install ragxplorer
```

**Usage**

```python
from ragxplorer import RAGxplorer
client = RAGxplorer(embedding_model="thenlper/gte-large")
client.load_pdf("presentation.pdf", verbose=True)
client.visualize_query("What are the top revenue drivers for Microsoft?")
```

A quickstart Jupyter notebook tutorial on how to use `ragxplorer` can be found at <https://github.com/gabrielchua/RAGxplorer/blob/main/quickstart.ipynb>

Or as a Colab notebook:

<a target="_blank" href="https://colab.research.google.com/github/vince-lam/RAGxplorer/blob/issue29-create-tutorials/quickstart.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Streamlit Demo 🔎

The demo can be found here: <https://ragxplorer.streamlit.app/>

<img src="https://raw.githubusercontent.com/gabrielchua/RAGxplorer/main/images/example.png" width="650">

View the project [here](https://github.com/gabrielchua/RAGxplorer-demo)

# Contributing 👋

Contributions to RAGxplorer are welcome. Please read our [contributing guidelines (WIP)](.github/CONTRIBUTING.md) for details.

# License 👀

This project is licensed under the MIT license - see the [LICENSE](LICENSE) for details.

# Acknowledgments 💙

- DeepLearning.AI and Chroma for the inspiration and code labs in their [Advanced Retrival](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) course.
- The Streamlit community for the support and resources.
