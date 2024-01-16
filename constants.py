PLOT_SIZE = 3

ABOUT_THIS_APP = "This application is inspired and adapts the code from [this excellent course](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) by DeepLearning.AI and Chroma."

CHUNK_EXPLAINER = "In RAG, your document is divided into parts (i.e. chunks) for searching, and relevant chunks are given to the LLM as additional context. \n \n"\
                "\"Chunk Size\" is the number of tokens in one of these chunks, and \"Chunk Overlap\" is the number of tokens shared between consecutive chunks to maintain context. \n\n"\
                "One word is about 3-4 tokens."

BUILD_VDB_LOADING_MSG = 'Building the vector database 🚧 ...'

VISUALISE_LOADING_MSG = 'Visualising your chunks 🎨 ...'

VISUALISATION_SETTINGS = {
            'Original Query': {'color': 'red', 'opacity': 1, 'symbol': 'cross', 'size': 15},
            'Retrived': {'color': 'green', 'opacity': 1, 'symbol': 'circle', 'size': 10},
            'Chunks': {'color': 'blue', 'opacity': 0.4, 'symbol': 'circle', 'size': 10},
            'Sub-Questions': {'color': 'purple', 'opacity': 1, 'symbol': 'star', 'size': 15},
            'Hypothetical Ans': {'color': 'purple', 'opacity': 1, 'symbol': 'star', 'size': 15},
        }
