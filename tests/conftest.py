import pytest
from unittest.mock import MagicMock, patch
from ragxplorer.ragxplorer import RAGxplorer

@pytest.fixture
def ragxplorer_instance():
    return RAGxplorer()

@pytest.fixture
def mock_pdf_file(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%EOF")
    return pdf_path

@pytest.fixture
def mock_openai_api():
    with patch('ragxplorer.query_expansion.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        yield mock_client
