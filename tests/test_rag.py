import unittest
from unittest.mock import MagicMock

from ragxplorer.rag import (build_vector_database, get_doc_embeddings,
                            get_docs, query_chroma)


class TestRag(unittest.TestCase):
    def test_build_vector_database(self):
        # TODO: Write unit tests for build_vector_database function
        pass

    def test_query_chroma(self):
        # TODO: Write unit tests for query_chroma function
        pass

    def test_get_doc_embeddings(self):
        # TODO: Write unit tests for get_doc_embeddings function
        pass

    def test_get_docs(self):
        # TODO: Write unit tests for get_docs function
        pass

if __name__ == '__main__':
    unittest.main()
        
        # Call the function
        result = get_doc_embeddings(chroma_collection)
        
        # Assert that the chroma_collection.get() method was called with the correct arguments
        chroma_collection.get.assert_called_once_with(include=['embeddings'])
        
        # Assert that the result is the embeddings array
        np.testing.assert_array_equal(result, embeddings)

    def test_get_docs(self):
        # TODO: Write unit tests for get_docs function
        pass

if __name__ == '__main__':
    unittest.main()
