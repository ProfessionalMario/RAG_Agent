import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

st.set_page_config(page_title="EDA Decision Agent", layout="wide")

st.title("EDA Decision Agent")
st.markdown("Automate data preprocessing decisions based on your EDA report.")

if not os.getenv("GEMINI_API_KEY"):
    st.warning(
        "**GEMINI_API_KEY is not configured.** "
        "Please add your Gemini API key as a secret named `GEMINI_API_KEY` to enable the pipeline."
    )
    
report = st.text_area("Paste EDA Report", height=300, placeholder="Paste your EDA report text here...")

if st.button("Run Pipeline", type="primary"):
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Cannot run pipeline: GEMINI_API_KEY is not set.")
    elif not report.strip():
        st.error("Please paste an EDA report before running.")
    else:
        with st.spinner("Running pipeline..."):
            try:
                import tempfile
                from rag.pipeline import run_pipeline

                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                    tmp.write(report)
                    tmp_path = tmp.name

                results = run_pipeline(tmp_path)
                os.unlink(tmp_path)

                st.success("Pipeline complete!")
                for r in results:
                    with st.expander(f"Column: {r.get('column', 'N/A')}"):
                        st.write("**Decision:**", r.get("decision", "N/A"))
                        if "query" in r:
                            st.write("**Query:**", r["query"])
                        if "error" in r:
                            st.error(r["error"])

            except Exception as e:
                st.error(f"Pipeline failed: {e}")
