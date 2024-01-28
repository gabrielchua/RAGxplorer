# RAGxplorer 🦙🦺 

[![PyPI version](https://img.shields.io/pypi/v/ragxplorer.svg)](https://pypi.org/project/ragxplorer/)

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

# Streamlit Demo (WIP) 🔎

<img src="https://raw.githubusercontent.com/gabrielchua/RAGxplorer/main/images/example.png" width="650">

# Contributing 👋

Contributions to RAGxplorer are welcome. Please read our [contributing guidelines (WIP)](.github/CONTRIBUTING.md) for details.

# License 👀

This project is licensed under the MIT license - see the LICENSE file for details.

# Acknowledgments 💙
- DeepLearning.AI and Chroma for the inspiration and code labs in their [Advanced Retrival](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) course.
- The Streamlit community for the support and resources.
