
"""
Retriever.py
Unified Retriever System for FEVER dataset.
Builds, persists, and reloads BM25 or Dense (Chroma-based) retrievers.

Usage:
    python retriever_setup.py  # builds and tests the retriever
"""

import os
import re
import ast
import nltk
import logging
import unicodedata
import pandas as pd
from typing import List
from datasets import load_dataset, Dataset
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from nltk.tokenize import sent_tokenize

# ------------------------------------
# CONFIG
# ------------------------------------
RETRIEVER_TYPE = "dense"  # ← "dense" or "bm25"
CHROMA_DIR = "./data_store/chroma_fever_store/"
FEVER_PATH = "./data/fever_shared_task_dev.jsonl"
WIKI_DIR = "./data/wiki-pages/wiki-pages"

os.makedirs(CHROMA_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------
# HELPER: Normalize label (optional)
# ------------------------------------
def normalize_label(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ------------------------------------
# RETRIEVER CONTROLLER CLASS
# ------------------------------------
class RetrieverController:
    """
    Unified wrapper for BM25 and Dense (Chroma) retrievers.
    Compatible with LangChain retriever API.
    """

    def __init__(self,embeddings, default_type="bm25", default_k=5, chroma_dir="./data_store/chroma_fever_store"):
        self.k = default_k
        self.active_type = default_type
        self.bm25 = None
        self.chroma = None
        self.embeddings = embeddings
        self.chroma_dir = chroma_dir
        logging.info(f"📦 RetrieverController initialized: type={default_type}, k={default_k}")

    def switch(self, retriever_type: str):
        if retriever_type not in ["bm25", "dense"]:
            raise ValueError("Retriever type must be 'bm25' or 'dense'.")
        self.active_type = retriever_type
        logging.info(f"🔄 Switched to retriever: {retriever_type}")

    def set_k(self, k: int):
        self.k = k
        logging.info(f"🔢 top-k set to {k}")

    def retrieve(self, query: str) -> List:
        if self.active_type == "bm25":
            return self._retrieve_bm25(query)
        elif self.active_type == "dense":
            return self._retrieve_dense(query)
        else:
            raise RuntimeError(f"Unknown retriever type: {self.active_type}")

    # ---------- BM25 ----------
    def _retrieve_bm25(self, query: str) -> List:
        if self.bm25 is None:
            raise RuntimeError("BM25 retriever not initialized. Use build_bm25(docs) first.")
        results = self.bm25.invoke(query, k=self.k)
        logging.info(f"BM25 retrieved {len(results)} docs.")
        return results

    def build_bm25(self, docs: List):
        self.bm25 = BM25Retriever.from_documents(docs)
        logging.info(f"✅ BM25 retriever built from {len(docs)} documents.")

    # ---------- DENSE ----------
    def _retrieve_dense(self, query: str) -> List:
        if self.chroma is None:
            self.chroma = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings
            )
        results = self.chroma.similarity_search(query, k=self.k)
        logging.info(f"Dense retriever got {len(results)} docs.")
        return results


# ------------------------------------
# RETRIEVER BUILD PIPELINE
# ------------------------------------
def build_retriever(retriver_type):
    # NLTK setup
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # 1. Load FEVER dataset
    logging.info("📘 Loading FEVER dataset...")
    df = pd.read_json(FEVER_PATH, lines=True)
    df["evidence"] = df["evidence"].astype(str)
    fever = Dataset.from_pandas(df)

    # 2. Load Wikipedia pages
    logging.info("📘 Loading Wikipedia pages...")
    jsonl_files = [os.path.join(WIKI_DIR, f) for f in os.listdir(WIKI_DIR) if f.endswith(".jsonl")]
    wiki_corpus = load_dataset("json", data_files=jsonl_files, split="train")

    # 3. Build sentence lookup
    logging.info("🔍 Building sentence lookup...")
    wiki_lookup = {}
    for page in wiki_corpus:
        page_id = page["id"]
        text = page["text"]
        sentences = sent_tokenize(text)
        wiki_lookup[page_id] = {i + 1: s for i, s in enumerate(sentences)}

    # 4. Convert FEVER to LangChain Documents
    logging.info("🧩 Building documents...")
    documents = []
    for item in fever:
        claim = item["claim"]
        evidences = item.get("evidence", [])
        evidence_texts = []

        if evidences:
            try:
                evidences = ast.literal_eval(evidences)
                for ev_set in evidences:
                    for ev in ev_set:
                        wiki_name = ev[2]
                        sent_idx = ev[3]
                        if wiki_name in wiki_lookup and sent_idx is not None:
                            sentence = wiki_lookup[wiki_name].get(sent_idx)
                            if sentence:
                                evidence_texts.append(f"{wiki_name}:{sent_idx}:{sentence}")
            except Exception as e:
                logging.warning(f"Skipping malformed evidence: {e}")

        combined_text = claim
        if evidence_texts:
            combined_text += " -- wikiReference --> " + " ".join(evidence_texts)

        documents.append(
            Document(
                page_content=combined_text,
                metadata={
                    "id": item.get("id", ""),
                    "label": item.get("label", ""),
                    "verifiable": item.get("verifiable", ""),
                },
            )
        )

    logging.info(f"✅ Built {len(documents)} documents")

    # 5. Build and persist retriever
    retriever = RetrieverController(default_type=retriver_type, chroma_dir=CHROMA_DIR)

    if retriver_type == "dense":
        logging.info("🚀 Building dense retriever (Chroma)...")
        vectorstore = Chroma.from_documents(
            documents,
            embedding=retriever.embeddings,
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
        retriever.chroma = vectorstore
        logging.info(f"✅ Dense retriever built and stored in {CHROMA_DIR}")

    elif retriver_type == "bm25":
        logging.info("🚀 Building BM25 retriever...")
        retriever.build_bm25(documents)
        logging.info("✅ BM25 retriever built and ready to use")

    else:
        raise ValueError("retriever_type must be 'dense' or 'bm25'")

    logging.info("🎉 Retriever build completed successfully.")
    return retriever


# ------------------------------------
# MAIN EXECUTION (Build + Test)
# ------------------------------------
if __name__ == "__main__":
    retriever = build_retriever("dense")
    test_query = "Was Albert Einstein a physicist?"
    logging.info(f"🧠 Testing retriever on query: {test_query}")
    results = retriever.retrieve(test_query)
    for i, doc in enumerate(results[:3], 1):
        logging.info(f"Result {i}: {doc.page_content[:180]}...")
