"""
rag_utils.py
------------
Core building blocks for the "Intelligent Question Answering System
for College Documents" project.

This file covers every concept listed in your project slide:
    1. PDF Extraction
    2. Document Chunking
    3. Embeddings
    4. Semantic Search
    5. Retrieval-Augmented Generation (RAG)

No large language model is trained here. We use:
    - a small, pre-trained sentence-embedding model (all-MiniLM-L6-v2)
      just to convert text into vectors.
    - FAISS to search those vectors quickly.
    - simple, template-based generation to turn the retrieved chunks
      into a readable answer (this is the "G" in RAG).

Each function has comments explaining WHAT it does and WHY, so this
file can also be used as a learning reference.
"""

import os
import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ---------------------------------------------------------------------
# STEP 1: PDF EXTRACTION
# ---------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Reads a PDF file and pulls out the raw text, page by page.

    Returns a list of dicts like:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    Keeping the page number attached to each chunk of text is what
    later lets us say "this answer came from page 4" (source
    highlighting).
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


# ---------------------------------------------------------------------
# STEP 2: DOCUMENT CHUNKING
# ---------------------------------------------------------------------
def chunk_text(pages: list[dict], chunk_size: int = 400, overlap: int = 60) -> list[dict]:
    """
    Large documents can't be searched effectively as one giant blob of
    text, and embedding models have a limited input size. So we split
    each page's text into smaller overlapping "chunks".

    chunk_size : target number of words per chunk
    overlap    : number of words repeated between consecutive chunks,
                 so we don't accidentally cut a sentence/answer in half
                 right at a chunk boundary.

    Returns a list of dicts:
        [{"id": 0, "page": 1, "text": "..."}, ...]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "id": chunk_id,
                "page": page["page"],
                "text": chunk_str
            })
            chunk_id += 1
            if end >= len(words):
                break
            start = end - overlap  # step back by `overlap` words

    return chunks


# ---------------------------------------------------------------------
# STEP 3: EMBEDDINGS
# ---------------------------------------------------------------------
class EmbeddingModel:
    """
    Wraps a small pre-trained sentence embedding model.

    An "embedding" is just a list of numbers (a vector) that represents
    the MEANING of a piece of text. Two chunks of text that mean similar
    things end up with vectors that point in a similar direction.
    We never train this model ourselves -- it already understands
    general English from its pre-training, we just use it.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Turns a list of strings into a 2D numpy array of vectors."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # important: makes cosine search = dot product
        )
        return embeddings.astype("float32")


# ---------------------------------------------------------------------
# STEP 4: SEMANTIC SEARCH (vector store + retrieval)
# ---------------------------------------------------------------------
class VectorStore:
    """
    A tiny wrapper around FAISS that stores chunk embeddings and lets
    us search for the chunks most similar (in meaning) to a question.
    """

    def __init__(self, dim: int):
        # IndexFlatIP = "flat" index using Inner Product.
        # Since our vectors are normalized, inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[dict] = []

    def add(self, chunks: list[dict], embeddings: np.ndarray):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Finds the top_k chunks whose embeddings are closest (most
        similar in meaning) to the query embedding.
        """
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            match = dict(self.chunks[idx])
            match["score"] = float(score)
            results.append(match)
        return results


# ---------------------------------------------------------------------
# STEP 5: RETRIEVAL-AUGMENTED GENERATION (RAG)
# ---------------------------------------------------------------------
def highlight_answer_span(question: str, chunk_text: str, max_sentences: int = 2) -> str:
    """
    Very simple 'source highlighting': out of all sentences in the
    retrieved chunk, pick the ones that share the most words with the
    question. This is the piece of text we'll show the user as
    "where this answer came from".
    """
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
    q_words = set(w.lower() for w in re.findall(r"\w+", question) if len(w) > 2)

    scored = []
    for s in sentences:
        s_words = set(w.lower() for w in re.findall(r"\w+", s))
        overlap = len(q_words & s_words)
        scored.append((overlap, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for score, s in scored[:max_sentences] if score > 0]

    if not best:
        # fall back to the first sentence(s) of the chunk
        best = sentences[:max_sentences]

    return " ".join(best).strip()


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """
    This is the 'Generation' step of RAG. Instead of training or
    calling a large language model, we build a clear, grounded answer
    directly out of the retrieved text -- this keeps the whole system
    lightweight, explainable, and free to run.

    Returns:
        {
            "answer": "...",
            "sources": [{"page": 3, "excerpt": "...", "score": 0.81}, ...]
        }
    """
    if not retrieved_chunks:
        return {"answer": "I couldn't find anything relevant in the uploaded document.",
                "sources": []}

    sources = []
    highlighted_pieces = []
    for chunk in retrieved_chunks:
        excerpt = highlight_answer_span(question, chunk["text"])
        highlighted_pieces.append(excerpt)
        sources.append({
            "page": chunk["page"],
            "excerpt": excerpt,
            "score": round(chunk["score"], 3)
        })

    # Combine the best excerpt(s) into one readable answer.
    answer = " ".join(dict.fromkeys(highlighted_pieces))  # dedupe, keep order
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------
# END-TO-END PIPELINE
# ---------------------------------------------------------------------
class CollegeDocQA:
    """
    Ties all five steps together into one easy-to-use pipeline:

        qa = CollegeDocQA()
        qa.load_pdf("prospectus.pdf")
        result = qa.ask("What is the attendance policy?")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = EmbeddingModel(model_name)
        self.store = None

    def load_pdf(self, pdf_path: str, chunk_size: int = 400, overlap: int = 60):
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            raise ValueError("No extractable text found in this PDF (it may be scanned images).")

        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
        embeddings = self.embedder.encode([c["text"] for c in chunks])

        self.store = VectorStore(dim=embeddings.shape[1])
        self.store.add(chunks, embeddings)
        return {"pages": len(pages), "chunks": len(chunks)}

    def ask(self, question: str, top_k: int = 3) -> dict:
        if self.store is None:
            raise RuntimeError("No document loaded yet. Call load_pdf() first.")
        q_emb = self.embedder.encode([question])
        retrieved = self.store.search(q_emb, top_k=top_k)
        return generate_answer(question, retrieved)


if __name__ == "__main__":
    # Quick manual test from the command line:
    #   python rag_utils.py sample.pdf "What is the attendance policy?"
    import sys
    if len(sys.argv) >= 3:
        qa = CollegeDocQA()
        info = qa.load_pdf(sys.argv[1])
        print(f"Loaded {info['pages']} pages -> {info['chunks']} chunks")
        result = qa.ask(sys.argv[2])
        print("\nANSWER:", result["answer"])
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  Page {s['page']} (score {s['score']}): {s['excerpt'][:150]}")
    else:
        print("Usage: python rag_utils.py <pdf_path> <question>")
