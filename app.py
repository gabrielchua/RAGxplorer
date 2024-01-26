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
    st_initialize_session_state_as_none,
    st_reset_application
)
from utils.rag import (
    build_vector_database,
    query_chroma,
    get_embedding,
    get_docs,
    get_doc_ids,
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
    VISUALISE_LOADING_MSG,
    EMBEDDING_MODEL_LIST,
    LLM_LIST
)

st.set_page_config(
    page_title="RAGxplorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session State
st_initialize_session_state_as_none(["document", "chroma", "filtered_df", "document_projections"])

if "document_projections_done" not in st.session_state.keys():
    st.session_state["document_projections_done"] = False

# Header
st_header()

# Set-UP Page
if st.session_state['document'] is None:

    col1, _ ,col2 = st.columns([0.6, 0.1, 0.3])

    col1.markdown("### 1. Upload your PDF 📄 \n For this demo, a 10-20 page PDF is recommended")
    uploaded_file = col1.file_uploader("Upload your PDF", label_visibility="collapsed", type='pdf')

    col1.markdown("### 2. Configure your RAG (Optional) 🔧")
    col1a, col1b, col1c = col1.columns(3)
    st.session_state["chunk_size"] = col1a.number_input("Chunk Size", value = 1000, step = 50)
    st.session_state["chunk_overlap"] = col1b.number_input("Chunk Overlap", step = 50)
    st.session_state["embedding_model"] = col1a.selectbox("Select your embedding model", EMBEDDING_MODEL_LIST)
    st.session_state["llm_model"] = col1b.selectbox("Select your LLM (Coming Soon)", LLM_LIST)

    col1.markdown("### 3. Build the VectorDB ⚡️")
    if col1.button("Build"):
        st.session_state['document'] = uploaded_file
        st.rerun()

    with col2.expander("**About this application**"):
        st.success(ABOUT_THIS_APP)
    with col2.expander("**EXPLAINER:** What does chunk size/overlap mean?"):
        st.info(CHUNK_EXPLAINER)

else:
    # Loading Page
    if st.session_state["chroma"] is None or st.session_state["document_projections_done"] == False:
        with st.spinner(BUILD_VDB_LOADING_MSG):
            st.session_state["chroma"] = build_vector_database(st.session_state['document'],
                                                               st.session_state["chunk_size"],
                                                               st.session_state["chunk_overlap"],
                                                               st.session_state["embedding_model"])
        with st.spinner(VISUALISE_LOADING_MSG):
            st.session_state["ids"] = get_doc_ids(st.session_state["chroma"])
            st.session_state["document_embeddings"] = get_doc_embeddings(st.session_state["chroma"])
            st.session_state["docs"] = get_docs(st.session_state["chroma"])
            st.session_state["umap_transform"] = set_up_umap(st.session_state["document_embeddings"])
            st.session_state["document_projections"] = get_projections(st.session_state["document_embeddings"], st.session_state["umap_transform"])
            st.session_state["document_projections_done"] = True
            st.rerun()

    # Explorer Page
    elif st.session_state["document_projections_done"]:
        col3, _, col4= st.columns([0.4, 0.1, 0.5])

        query = col3.text_input("Enter your query")

        strategy = col3.selectbox("Retrival strategy",
                                  ["Naive", 
                                   "Query Expansion - Multiple Qns", 
                                   "Query Expansion - Hypothetical Ans"])

        top_k = col3.number_input("Top K", value=5, min_value=1, max_value=10, step=1)

        df = prepare_projections_df()

        search = col3.button("Search ⚡")
        if col3.button("Reset ⚠️"):
            st_reset_application()



        if search:

            st.session_state['query_projections'] = get_projections(get_embedding(model=st.session_state["embedding_model"], text=query), st.session_state["umap_transform"])

            df_query = pd.DataFrame({"x": [st.session_state['query_projections'][0][0]],
                                    "y": [st.session_state['query_projections'][1][0]],
                                    "document_cleaned": query,
                                    "category": "Original Query",
                                    "size": PLOT_SIZE
                                    })

            if strategy == "Query Expansion - Multiple Qns":
                st.session_state['query_expansion_multi'] = generate_sub_qn(query)
                NUM_MULTI = len(st.session_state['query_expansion_multi'])
                st.session_state['query_projections_multi_qn'] = [get_projections(get_embedding(model=st.session_state["embedding_model"], text=sub_qn), st.session_state["umap_transform"]) for sub_qn in st.session_state['query_expansion_multi']]


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
                st.session_state['query_projections_hypo'] = get_projections(get_embedding(model=st.session_state["embedding_model"], text=st.session_state['query_expansion_hypo']), st.session_state["umap_transform"])

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
            df.loc[df['id'].isin(st.session_state['retrieved_id']), "category"] = "Retrieved"

            df = pd.concat([df, df_query], axis = 0)

            st.session_state["filtered_df"] = df[df['category'] == "Retrieved"]

            st.markdown("### Retrieved Chunks")
            st.dataframe(st.session_state["filtered_df"]['document'])

        with col4:
            plot_embeddings(df)
