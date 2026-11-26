# Intelligent Question Answering System for College Documents

A working RAG (Retrieval-Augmented Generation) system that answers questions
about college documents — prospectus, university regulations, academic
calendar — **without training a large language model**.

## How to run it

### Option A: Google Colab (recommended, matches your usual workflow)
```python
!pip install streamlit pypdf sentence-transformers faiss-cpu -q
!wget -q -O - ipv4.icanhazip.com   # (optional) get a public IP for the tunnel
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```
Or simply run `rag_utils.py` directly in a notebook cell (see "Quick test"
below) without the web UI at all.

### Option B: Your own machine
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### Quick test (no UI, just the terminal)
```bash
python rag_utils.py your_document.pdf "What is the attendance policy?"
```

> Note: the first time you run this, it downloads a small (~90MB) pre-trained
> embedding model from Hugging Face. This needs an internet connection once —
> after that it's cached locally and works offline.

## Project files

| File | Purpose |
|---|---|
| `rag_utils.py` | Core pipeline: PDF extraction → chunking → embeddings → semantic search → RAG answer generation |
| `app.py` | Streamlit UI: Upload PDF, Ask Questions, Retrieve Answers, Highlight Source Text |
| `make_sample_pdf.py` | Generates a sample college regulations PDF for testing |
| `requirements.txt` | Python dependencies |

## How each concept from the brief maps to the code

**1. PDF Extraction** — `extract_text_from_pdf()` in `rag_utils.py`
uses `pypdf` to pull raw text out of each page of the uploaded PDF.

**2. Document Chunking** — `chunk_text()` splits each page into
overlapping ~400-word chunks. Chunking matters because (a) embedding
models have an input size limit, and (b) smaller chunks give more
precise search matches than searching whole pages.

**3. Embeddings** — `EmbeddingModel` uses `sentence-transformers`
(`all-MiniLM-L6-v2`), a small pre-trained model, to turn each chunk of
text into a 384-number vector that captures its *meaning*. This model
is only used for inference — it is not trained or fine-tuned here.

**4. Semantic Search** — `VectorStore` wraps FAISS (Facebook AI
Similarity Search) to store all chunk vectors and quickly find the
ones closest in meaning to a question's vector. This is "semantic"
search because it matches by meaning, not exact keywords — e.g. a
question about "leave policy" can still match a chunk that says
"attendance requirements".

**5. Retrieval-Augmented Generation (RAG)** — `generate_answer()`
takes the retrieved chunks and assembles a grounded answer directly
from them (via `highlight_answer_span()`, which picks the sentence(s)
most relevant to the question). This is the "Generation" step — it's
template-based here rather than using a large language model, which
keeps the whole system lightweight, explainable, and free to run,
while still following the RAG pattern: **R**etrieve relevant chunks,
then **G**enerate an answer grounded in them.

## Extending it (optional, for a stronger submission)

- Swap the template-based `generate_answer()` for a call to a real LLM
  (e.g. the Anthropic API) that takes the retrieved chunks as context
  and writes a more natural-sounding answer — this turns "extractive"
  RAG into full "generative" RAG.
- Support multiple PDFs at once (loop `load_pdf()` over several files
  and tag chunks with a `source_doc` field).
- Add a confidence threshold: if the top similarity score is too low,
  reply "This isn't covered in the uploaded document" instead of
  guessing.
