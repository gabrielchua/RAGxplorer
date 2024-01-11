__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from utils import (
    build_vector_database,
    query_chroma,
    get_embedding,
    get_docs,
    get_doc_embeddings,
    set_up_umap,
    get_projections,
    )

st.set_page_config(
    page_title="RAGxplorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State
for key in ["document", "chroma", "filtered_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

PLOT_SIZE = 3

# HEADER
st.header("RAGxplorer 🗺️", divider='grey')

st.markdown("#### Visualise which chunks are most relevant to your query.")

# MAIN PANEL
if st.session_state['document'] is None:
    col1, col2 = st.columns(2)
    col1.markdown("### 1. Upload your PDF 📄")
    uploaded_file = col1.file_uploader("Upload your PDF", label_visibility="collapsed", type='pdf')
    
    col1.markdown("### 2. Configurations (Optional) 🔧")
    st.session_state["chunk_size"] = col1.number_input("Chunk Size", value = 1000, step = 50)
    st.session_state["chunk_overlap"] = col1.number_input("Chunk Overlap", step = 50)
    # st.session_state['openai_api_key'] = sidebar_1.text_input("OpenAI API Key", type="password")
    # sidebar_1.write("Enter your OpenAI API key if you want to use OpenAI's emebdding model. Else, `all-MiniLM-L6-v2` is used by default.")


    col1.markdown("### 3. Build VectorDB ⚡️")
    if col1.button("Build"):
        st.session_state['document'] = uploaded_file
        st.rerun()

    with col2.expander("**About this application**"):
        st.success("This application is inspired and adapts the code from [this excellent course](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) by DeepLearning.AI and Chroma.")


    with col2.expander("**EXPLAINER:** What is chunk size/overlap?"):
        st.info("In RAG, your document is divided into parts (i.e. chunks) for searching, and relevant chunks are given to the LLM as additional context. \n \n"\
                "\"Chunk Size\" is the number of tokens in one of these chunks, and \"Chunk Overlap\" is the number of tokens shared between consecutive chunks to maintain context. \n\n"\
                "One word is about 3-4 tokens.")


else:
    if st.session_state["chroma"] is None:
        with st.spinner('Building the vector database 🚧 ...'):
            st.session_state["chroma"] = build_vector_database(st.session_state['document'],
                                                               st.session_state["chunk_size"],
                                                               st.session_state["chunk_overlap"])
        with st.spinner('Visualising your chunks 🎨 ...'):
            st.session_state["document_embeddings"] = get_doc_embeddings(st.session_state["chroma"])
            st.session_state["docs"] = get_docs(st.session_state["chroma"])
            st.session_state["umap_transform"] = set_up_umap(st.session_state["document_embeddings"])
            st.session_state["document_projections"] = get_projections(st.session_state["document_embeddings"], st.session_state["umap_transform"])
            st.rerun()

    else:
        col3, col4 = st.columns([0.8, 0.2])
        query = col3.text_input("Enter your query")
        col4.write("")
        col4.write("")
        search = col4.button("Search")


        col5, _ ,col6 = st.columns([0.75, 0.05, 0.2])
        top_k = col6.number_input("Number of Chunks", value=5, min_value=1, max_value=10, step=1)
        strategy = col6.selectbox("Select your retrival strategy",
                                  ["Naive", "Query Expansion - Multiple Qns (not yet implemented)", "Query Expansion - Hypothetical Ans (not yet implemented)"])
        with col6.expander("Note"):
            st.warning("Query Expansion is not yet implemented")

        df = pd.DataFrame({"x": st.session_state["document_projections"][0], "y": st.session_state["document_projections"][1]})
        df = df.assign(document=st.session_state["docs"])

        df['document_cleaned'] = df.document.str.wrap(80).apply(lambda x: x.replace('\n', '<br>'))

        df = df.assign(size=PLOT_SIZE)
        df = df.assign(category="Chunks")

        if search:
            st.session_state['query_projections'] = get_projections(get_embedding(query), st.session_state["umap_transform"])
            st.session_state['retrieved_id'] = query_chroma(st.session_state["chroma"], query, top_k)
            st.session_state['retrieved_id'] = [int(index) for index in st.session_state['retrieved_id']]

            
            df.loc[st.session_state['retrieved_id'], "category"] = "Retrived"

            df_query = pd.DataFrame({"x": [st.session_state['query_projections'][0][0]],
                                    "y": [st.session_state['query_projections'][1][0]],
                                    "document": query,
                                    "category": "Original Query",
                                    "size": PLOT_SIZE
                                    })
            
            df = pd.concat([df, df_query], axis = 0)

            st.session_state["filtered_df"] = df[df['category'] == "Retrived"]

            
            st.markdown("### Retrived Chunk")
            st.dataframe(st.session_state["filtered_df"]['document'])

        category_settings = {
            'Original Query': {'color': 'red', 'opacity': 1, 'symbol': 'cross', 'size': 20},
            'Retrived': {'color': 'green', 'opacity': 1, 'symbol': 'circle', 'size': 10},
            'Chunks': {'color': 'blue', 'opacity': 0.4, 'symbol': 'circle', 'size': 10},
        }

        # Create a figure
        fig = go.Figure()

        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            settings = category_settings.get(category, {'color': 'grey', 'opacity': 1, 'symbol': 'circle', 'size': 10})
            fig.add_trace(go.Scatter(
                x=category_df['x'],
                y=category_df['y'],
                mode='markers',
                name=category,
                marker=dict(
                    color=settings['color'],
                    opacity=settings['opacity'],
                    symbol=settings['symbol'],
                    size=settings['size'],
                    line_width=0
                ),
                hoverinfo='text',
                text=category_df['document_cleaned']
            ))

        # Set the layout, including moving the legend to the top
        fig.update_layout(
            height=500,
            legend=dict(
                y=100,
                x=0.5,
                xanchor='center',
                yanchor='top',
                orientation='h'
            )
        )
                    
        # fig = px.scatter(data_frame=df,
        #                 x="x",
        #                 y="y",
        #                 size="size",
        #                 color="category",
        #                 symbol="category",
        #                 hover_name="document",
        #                 hover_data={"x": False, "y": False, "category": False, "size": False},
        #                 height=500)

        col5.plotly_chart(fig, use_container_width=True)