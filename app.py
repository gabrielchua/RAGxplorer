"""
Streamlit app
"""
# Line 6 to 8 is for streamlit commmunity deployment
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

import streamlit as st
import pandas as pd
from utils.st import(
    st_header,
    st_initilize_session_state_as_none,
    st_reset_application
)
from utils.rag import (
    build_vector_database,
    query_chroma,
    get_embedding,
    get_docs,
    get_doc_embeddings
    )
from utils.projections import (
    set_up_umap,
    get_projections,
    prepare_projections_df,
    plot_embeddings
    )  
from utils.query_expansion import (
    generate_sub_qn,
    generate_hypothetical_ans
)
from utils.constants import (
    PLOT_SIZE,
    ABOUT_THIS_APP,
    CHUNK_EXPLAINER,
    BUILD_VDB_LOADING_MSG,
    VISUALISE_LOADING_MSG
)

st.set_page_config(
    page_title="RAGxplorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State
st_initilize_session_state_as_none(["document", "chroma", "filtered_df", "document_projections"])

if "document_projections_done" not in st.session_state.keys():
    st.session_state["document_projections_done"] = False

# UI
st_header()

# View 1
if st.session_state['document'] is None:
    col1, col2 = st.columns(2)
    col1.markdown("### 1. Upload your PDF 📄")
    col1.markdown("For this demo, a 10-20 page PDF is recommended.")
    uploaded_file = col1.file_uploader("Upload your PDF", label_visibility="collapsed", type='pdf')
    col1.markdown("### 2. Configurations (Optional) 🔧")
    st.session_state["chunk_size"] = col1.number_input("Chunk Size", value = 1000, step = 50)
    st.session_state["chunk_overlap"] = col1.number_input("Chunk Overlap", step = 50)
    st.session_state["embedding_model"] = col1.selectbox("Select your embedding model",
                                  ["all-MiniLM-L6-v2",
                                   "text-embedding-ada-002", 
                                  # "gte-large"
                                   ])

    col1.markdown("### 3. Build VectorDB ⚡️")
    if col1.button("Build"):
        st.session_state['document'] = uploaded_file
        st.rerun()

    with col2.expander("**About this application**"):
        st.success(ABOUT_THIS_APP)

    with col2.expander("**EXPLAINER:** What is chunk size/overlap?"):
        st.info(CHUNK_EXPLAINER)

else:
    # View 2
    if st.session_state["chroma"] is None or st.session_state["document_projections_done"] == False:
        with st.spinner(BUILD_VDB_LOADING_MSG):
            st.session_state["chroma"] = build_vector_database(st.session_state['document'],
                                                               st.session_state["chunk_size"],
                                                               st.session_state["chunk_overlap"],
                                                               st.session_state["embedding_model"])
        with st.spinner(VISUALISE_LOADING_MSG):
            st.session_state["document_embeddings"] = get_doc_embeddings(st.session_state["chroma"])
            st.session_state["docs"] = get_docs(st.session_state["chroma"])
            st.session_state["umap_transform"] = set_up_umap(st.session_state["document_embeddings"])
            st.session_state["document_projections"] = get_projections(st.session_state["document_embeddings"],
                                                                       st.session_state["umap_transform"])
            
            st.session_state["document_projections_done"] = True
            st.rerun()

    # View 3
    elif st.session_state["document_projections_done"]:
        col3, col4a, col4b = st.columns([0.8, 0.1, 0.1])
        query = col3.text_input("Enter your query")
        col4a.write("")
        col4a.write("")
        search = col4a.button("Search")
        col4b.write("")
        col4b.write("")
        if col4b.button("Reset App ⚠️"):
            st_reset_application()

        col5, _ ,col6 = st.columns([0.75, 0.05, 0.2])
        top_k = col6.number_input("Number of Chunks", value=5, min_value=1, max_value=10, step=1)
        strategy = col6.selectbox("Select your retrival strategy",
                                  ["Naive", 
                                   "Query Expansion - Multiple Qns", 
                                   "Query Expansion - Hypothetical Ans"])

        df = prepare_projections_df()

        if search:

            st.session_state['query_projections'] = get_projections(get_embedding(query), st.session_state["umap_transform"])

            df_query = pd.DataFrame({"x": [st.session_state['query_projections'][0][0]],
                                    "y": [st.session_state['query_projections'][1][0]],
                                    "document_cleaned": query,
                                    "category": "Original Query",
                                    "size": PLOT_SIZE
                                    })

            if strategy == "Query Expansion - Multiple Qns":
                st.session_state['query_expansion_multi'] = generate_sub_qn(query)
                NUM_MULTI = len(st.session_state['query_expansion_multi'])
                st.session_state['query_projections_multi_qn'] = [get_projections(get_embedding(sub_qn), st.session_state["umap_transform"]) for sub_qn in st.session_state['query_expansion_multi']]


                df_query_multi = pd.DataFrame({"x": [projection[0][0] for projection in st.session_state['query_projections_multi_qn']],
                        "y": [projection[1][0] for projection in st.session_state['query_projections_multi_qn']],
                        "document_cleaned": st.session_state['query_expansion_multi'],
                        "category": ["Sub-Questions"]*NUM_MULTI,
                        "size": [PLOT_SIZE]*NUM_MULTI
                        })
                
                # df_query_multi = df_query_multi.document_cleaned.str.wrap(80).apply(lambda x: x.replace('\n', '<br>'))

                df_query = pd.concat([df_query, df_query_multi])

                chroma_search = st.session_state['query_expansion_multi']


            elif strategy == "Query Expansion - Hypothetical Ans":
                st.session_state['query_expansion_hypo'] = generate_hypothetical_ans(query)
                st.session_state['query_projections_hypo'] = get_projections(get_embedding(st.session_state['query_expansion_hypo']), st.session_state["umap_transform"])

                df_query_hypo = pd.DataFrame({"x": [st.session_state['query_projections_hypo'][0][0]],
                        "y": [st.session_state['query_projections_hypo'][1][0]],
                        "document_cleaned": st.session_state['query_expansion_hypo'],
                        "category": "Hypothetical Ans",
                        "size": PLOT_SIZE
                        })
                
                df_query_hypo = df_query_hypo.document_cleaned.str.wrap(80).apply(lambda x: x.replace('\n', '<br>'))
                
                df_query = pd.concat([df_query, df_query_hypo])

                chroma_search = st.session_state['query_expansion_hypo']

            else:
                chroma_search = query

            st.session_state['retrieved_id'] = query_chroma(st.session_state["chroma"],
                                                            chroma_search,
                                                            top_k)
            st.session_state['retrieved_id'] = [int(index) for index in st.session_state['retrieved_id']]
            df.loc[st.session_state['retrieved_id'], "category"] = "Retrieved"

            df = pd.concat([df, df_query], axis = 0)

            st.session_state["filtered_df"] = df[df['category'] == "Retrieved"]

            st.markdown("### Retrieved Chunks")
            st.dataframe(st.session_state["filtered_df"]['document'])

        with col5:
            plot_embeddings(df)
