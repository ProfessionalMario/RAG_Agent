import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
from rag.pipeline import run_pipeline


st.set_page_config(page_title="EDA Decision Agent")

st.title("EDA Decision Agent")

import streamlit as st


report = st.text_area("Paste Report")

# if st.button("Run"):
#     response = call_llm(PROMPT)
#     st.subheader("Raw Output")
#     st.code(response)

#     parsed = extract_json(response)
#     st.subheader("Parsed JSON")
#     st.json(parsed)