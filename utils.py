import streamlit as st 
import pandas as pd
from constants import PLOT_SIZE

def st_initilize_session_state_as_none(key_list):
    for key in key_list:
        if key not in st.session_state:
            st.session_state[key] = None


def prepare_projections_df():
    df = pd.DataFrame({"x": st.session_state["document_projections"][0], 
                    "y": st.session_state["document_projections"][1]})
    
    df = df.assign(document=st.session_state["docs"])

    df['document_cleaned'] = df.document.str.wrap(80).apply(lambda x: x.replace('\n', '<br>'))

    df = df.assign(size=PLOT_SIZE)
    df = df.assign(category="Chunks")
    
    return df
