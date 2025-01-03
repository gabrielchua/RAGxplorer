import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import umap
from ragxplorer.ragxplorer import RAGxplorer

class TestRAGxplorer(unittest.TestCase):

    @patch('ragxplorer.ragxplorer.build_vector_database')
    @patch('ragxplorer.ragxplorer.get_doc_embeddings')
    @patch('ragxplorer.ragxplorer.get_docs')
    @patch('ragxplorer.ragxplorer.set_up_umap')
    @patch('ragxplorer.ragxplorer.get_projections')
    @patch('ragxplorer.ragxplorer.prepare_projections_df')
    def test_load_pdf(self, mock_prepare_projections_df, mock_get_projections, mock_set_up_umap, mock_get_docs, mock_get_doc_embeddings, mock_build_vector_database):
        mock_build_vector_database.return_value = MagicMock()
        mock_get_doc_embeddings.return_value = MagicMock()
        mock_get_docs.return_value = MagicMock()
        mock_set_up_umap.return_value = MagicMock()
        mock_get_projections.return_value = (MagicMock(), MagicMock())
        mock_prepare_projections_df.return_value = pd.DataFrame()

        ragxplorer = RAGxplorer()
        ragxplorer.load_pdf('dummy_path.pdf', verbose=True)

        mock_build_vector_database.assert_called_once()
        mock_get_doc_embeddings.assert_called_once()
        mock_get_docs.assert_called_once()
        mock_set_up_umap.assert_called_once()
        mock_get_projections.assert_called_once()
        mock_prepare_projections_df.assert_called_once()

    @patch('ragxplorer.ragxplorer.query_chroma')
    @patch('ragxplorer.ragxplorer.get_projections')
    @patch('ragxplorer.ragxplorer.plot_embeddings')
    def test_visualize_query(self, mock_plot_embeddings, mock_get_projections, mock_query_chroma):
        mock_get_projections.return_value = (MagicMock(), MagicMock())
        mock_query_chroma.return_value = ['doc1', 'doc2']
        mock_plot_embeddings.return_value = MagicMock()

        ragxplorer = RAGxplorer()
        ragxplorer._vectordb = MagicMock()
        ragxplorer._VizData.base_df = pd.DataFrame({'id': ['doc1', 'doc2'], 'category': ['Chunks', 'Chunks']})
        ragxplorer._chosen_embedding_model = MagicMock()
        ragxplorer._projector = MagicMock()

        fig = ragxplorer.visualize_query('dummy query')

        mock_get_projections.assert_called_once()
        mock_query_chroma.assert_called_once()
        mock_plot_embeddings.assert_called_once()
        self.assertIsNotNone(fig)

    def test_export_chroma(self):
        ragxplorer = RAGxplorer()
        ragxplorer._vectordb = MagicMock()
        exported_chroma = ragxplorer.export_chroma()
        self.assertIsNotNone(exported_chroma)

    @patch('ragxplorer.ragxplorer.get_doc_embeddings')
    @patch('ragxplorer.ragxplorer.get_docs')
    @patch('ragxplorer.ragxplorer.set_up_umap')
    @patch('ragxplorer.ragxplorer.get_projections')
    @patch('ragxplorer.ragxplorer.prepare_projections_df')
    def test_load_chroma(self, mock_prepare_projections_df, mock_get_projections, mock_set_up_umap, mock_get_docs, mock_get_doc_embeddings):
        mock_get_doc_embeddings.return_value = MagicMock()
        mock_get_docs.return_value = MagicMock()
        mock_set_up_umap.return_value = MagicMock()
        mock_get_projections.return_value = (MagicMock(), MagicMock())
        mock_prepare_projections_df.return_value = pd.DataFrame()

        ragxplorer = RAGxplorer()
        ragxplorer.load_chroma(MagicMock(), initialize_projector=True, recompute_projections=True, verbose=True)

        mock_get_doc_embeddings.assert_called_once()
        mock_get_docs.assert_called_once()
        mock_set_up_umap.assert_called_once()
        mock_get_projections.assert_called_once()
        mock_prepare_projections_df.assert_called_once()

    def test_export_projector(self):
        ragxplorer = RAGxplorer()
        ragxplorer._projector = MagicMock()
        exported_projector = ragxplorer.export_projector()
        self.assertIsNotNone(exported_projector)

    @patch('ragxplorer.ragxplorer.get_projections')
    @patch('ragxplorer.ragxplorer.prepare_projections_df')
    def test_load_projector(self, mock_prepare_projections_df, mock_get_projections):
        mock_get_projections.return_value = (MagicMock(), MagicMock())
        mock_prepare_projections_df.return_value = pd.DataFrame()

        ragxplorer = RAGxplorer()
        ragxplorer._documents.embeddings = MagicMock()
        ragxplorer._documents.ids = MagicMock()
        ragxplorer._documents.text = MagicMock()

        ragxplorer.load_projector(MagicMock(), recompute_projections=True)

        mock_get_projections.assert_called_once()
        mock_prepare_projections_df.assert_called_once()

if __name__ == '__main__':
    unittest.main()
