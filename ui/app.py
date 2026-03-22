import sys
import os

# 🔥 MUST be BEFORE any project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from scripts.run_pipeline import run_pipeline


st.set_page_config(page_title="EDA Decision Agent")

st.title("EDA Decision Agent")

if st.button("Run Analysis"):
    results = run_pipeline("reports.txt")

    for r in results:
        st.subheader(f"Column: {r['column']}")

        for line in r["decision"].split("\n"):
            st.write(line)