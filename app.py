"""
app.py
------
Streamlit front-end for the Intelligent Question Answering System
for College Documents.

Run it with:
    streamlit run app.py

Expected Features (from project brief):
    - Upload PDF
    - Ask Questions
    - Retrieve Answers
    - Highlight Source Text
"""

import streamlit as st
import tempfile
import os
from rag_utils import CollegeDocQA

st.set_page_config(page_title="College Docs Q&A", page_icon="🎓", layout="centered")

st.title("🎓 Intelligent Question Answering System for College Documents")
st.caption(
    "Upload a college document (prospectus, regulations, academic calendar) "
    "and ask questions in plain English. Answers are retrieved directly from "
    "the document — no large language model is trained."
)

# ---------------------------------------------------------------------
# Session state: keep the loaded QA pipeline alive between interactions
# ---------------------------------------------------------------------
if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------
# 1. UPLOAD PDF
# ---------------------------------------------------------------------
st.subheader("1. Upload PDF")
uploaded_file = st.file_uploader("Choose a college document (PDF)", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.slider("Chunk size (words)", 100, 800, 400, step=50)
with col2:
    top_k = st.slider("Chunks to retrieve per question", 1, 5, 3)

if uploaded_file is not None:
    if st.session_state.doc_name != uploaded_file.name:
        with st.spinner("Reading PDF, splitting into chunks, and building the search index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            pipeline = CollegeDocQA()
            try:
                info = pipeline.load_pdf(tmp_path, chunk_size=chunk_size)
                st.session_state.qa_pipeline = pipeline
                st.session_state.doc_name = uploaded_file.name
                st.session_state.history = []
                st.success(
                    f"Indexed **{uploaded_file.name}** — "
                    f"{info['pages']} pages → {info['chunks']} searchable chunks."
                )
            except ValueError as e:
                st.error(str(e))
            finally:
                os.unlink(tmp_path)

st.divider()

# ---------------------------------------------------------------------
# 2. ASK QUESTIONS
# ---------------------------------------------------------------------
st.subheader("2. Ask Questions")

if st.session_state.qa_pipeline is None:
    st.info("Upload a PDF above to get started.")
else:
    question = st.text_input(
        "Ask something about the document",
        placeholder="e.g. What is the attendance policy?"
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        with st.spinner("Searching the document..."):
            result = st.session_state.qa_pipeline.ask(question, top_k=top_k)
        st.session_state.history.insert(0, {"question": question, "result": result})

    # -------------------------------------------------------------
    # 3 & 4. RETRIEVE ANSWERS + HIGHLIGHT SOURCE TEXT
    # -------------------------------------------------------------
    for item in st.session_state.history:
        st.markdown(f"**Q: {item['question']}**")
        st.write(item["result"]["answer"])

        with st.expander("📄 Source text (highlighted)"):
            for src in item["result"]["sources"]:
                st.markdown(
                    f"**Page {src['page']}** · relevance score: `{src['score']}`"
                )
                st.markdown(
                    f"<div style='background-color:#fff3b0; padding:8px; "
                    f"border-radius:6px; margin-bottom:10px;'>{src['excerpt']}</div>",
                    unsafe_allow_html=True,
                )
        st.divider()

# ---------------------------------------------------------------------
# Sidebar: concepts covered (matches the project brief)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Concepts Covered")
    st.markdown(
        "- PDF Extraction\n"
        "- Document Chunking\n"
        "- Embeddings\n"
        "- Semantic Search\n"
        "- Retrieval-Augmented Generation (RAG)"
    )
    st.header("Learning Outcome")
    st.markdown(
        "Learn modern NLP and LLM workflows **without training a large "
        "language model** — instead, we use a small pre-trained embedding "
        "model plus retrieval."
    )
