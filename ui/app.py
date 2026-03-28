import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
from rag.pipeline import run_pipeline


st.set_page_config(page_title="EDA Decision Agent")

st.title("EDA Decision Agent")

import streamlit as st

st.title("EDA Decision Agent")

if st.button("Run Analysis"):
    results = run_pipeline("reports.txt")

    for r in results:
        st.subheader(r["column"])
        st.text(r["decision"])