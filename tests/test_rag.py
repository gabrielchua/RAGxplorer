import unittest
from unittest.mock import patch, MagicMock
from ragxplorer.rag import _load_pdf, _split_text_into_chunks, _split_chunks_into_tokens, _create_and_populate_chroma_collection

class TestRag(unittest.TestCase):

    @patch('ragxplorer.rag.PdfReader')
    def test_load_pdf(self, mock_pdf_reader):
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = "Page 1 text"
        mock_pdf.pages[1].extract_text.return_value = "Page 2 text"
        mock_pdf_reader.return_value = mock_pdf

        result = _load_pdf("dummy_path.pdf")
        self.assertEqual(result, ["Page 1 text", "Page 2 text"])

    def test_split_text_into_chunks(self):
        pdf_texts = ["This is a test. " * 100]
        chunk_size = 100
        chunk_overlap = 0

        result = _split_text_into_chunks(pdf_texts, chunk_size, chunk_overlap)
        self.assertTrue(len(result) > 0)

    def test_split_chunks_into_tokens(self):
        character_split_texts = ["This is a test. " * 100]

        result = _split_chunks_into_tokens(character_split_texts)
        self.assertTrue(len(result) > 0)

    @patch('ragxplorer.rag.chromadb.Client')
    def test_create_and_populate_chroma_collection(self, mock_chroma_client):
        mock_collection = MagicMock()
        mock_chroma_client.return_value.create_collection.return_value = mock_collection

        token_split_texts = ["This is a test. " * 100]
        embedding_model = MagicMock()

        result = _create_and_populate_chroma_collection(token_split_texts, embedding_model)
        self.assertIsNotNone(result)
        mock_collection.add.assert_called_once()

if __name__ == '__main__':
    unittest.main()
