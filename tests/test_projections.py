import unittest
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from unittest.mock import patch, MagicMock
from ragxplorer.projections import set_up_umap, get_projections, prepare_projections_df, plot_embeddings

class TestProjections(unittest.TestCase):

    def test_set_up_umap(self):
        embeddings = np.random.rand(100, 50)
        umap_transform = set_up_umap(embeddings)
        self.assertIsNotNone(umap_transform)
        self.assertIsInstance(umap_transform, umap.UMAP)

    @patch('ragxplorer.projections._project_embeddings')
    def test_get_projections(self, mock_project_embeddings):
        mock_project_embeddings.return_value = np.random.rand(100, 2)
        embeddings = np.random.rand(100, 50)
        umap_transform = MagicMock()
        x, y = get_projections(embeddings, umap_transform)
        self.assertEqual(len(x), 100)
        self.assertEqual(len(y), 100)

    def test_prepare_projections_df(self):
        document_ids = [str(i) for i in range(100)]
        document_projections = (np.random.rand(100), np.random.rand(100))
        document_text = ["Document text " + str(i) for i in range(100)]
        df = prepare_projections_df(document_ids, document_projections, document_text)
        self.assertEqual(len(df), 100)
        self.assertIn('id', df.columns)
        self.assertIn('x', df.columns)
        self.assertIn('y', df.columns)
        self.assertIn('document', df.columns)
        self.assertIn('document_cleaned', df.columns)
        self.assertIn('size', df.columns)
        self.assertIn('category', df.columns)

    def test_plot_embeddings(self):
        df = pd.DataFrame({
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'document_cleaned': ["Document text " + str(i) for i in range(100)],
            'category': ['Chunks'] * 100,
            'size': [10] * 100
        })
        fig = plot_embeddings(df)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

if __name__ == '__main__':
    unittest.main()
