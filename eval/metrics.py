# eval/metrics.py
"""
RAGAS-based evaluation harness for adaptive RAG trials.

Computes:
 - Faithfulness (standard + optional HHEM classifier)
 - Answer Accuracy (LLM-as-a-judge)
 - Context Relevance
 - Context Precision
 - Context Recall

Each metric is computed using the official RAGAS implementations.
Supports both synchronous and asynchronous batch evaluation.

Requirements:
    pip install ragas==0.1.16
    pip install openai  # or mistralai, depending on your evaluator_llm choice
"""
from ragas.metrics import (
    Faithfulness,
    FaithfulnesswithHHEM,
    AnswerAccuracy,
    ContextRelevance,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from ragas import evaluate
import pandas as pd
from datasets import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer, util

def evaluate_ragas(
    llm,
    embeddings, 
    dataset,
    device: str = "cpu",
    batch_size: int = 10,
):
    """
    Evaluate a single query–answer pair using RAGAS metrics with Mistral support.
    """
    rag_df = pd.DataFrame([
        {
            "question": str(d["question"]),
            "answer": str(d["answer"]),
            "contexts": [str(doc.page_content) for doc in d["contexts"]],
            "ground_truth": str(d["ground_truth"])
        }
        for d in dataset
    ])
    rag_df.fillna("", inplace=True)

    eval_dataset = Dataset.from_pandas(rag_df)

    faithfulness_metric = Faithfulness(llm=llm)
    accuracy_metric = AnswerAccuracy(llm=llm)
    relevance_metric = ContextRelevance(llm=llm)
    precision_metric = LLMContextPrecisionWithReference(llm=llm)
    recall_metric = LLMContextRecall(llm=llm)
    metrics = evaluate(eval_dataset,metrics=[faithfulness_metric,accuracy_metric,relevance_metric,precision_metric,recall_metric],
                      llm=llm,embeddings=embeddings)
    print("metrics is",metrics)
    # Safely extract scores as dict
    if hasattr(metrics, "_scores_dict"):
        metrics_dict = metrics._scores_dict
    else:
        # fallback: try iterating
        metrics_dict = {k: v for k, v in metrics.items()} if hasattr(metrics, "items") else {}
    
    return metrics_dict




# ---------------------------------------------------------------------
# Setup embedding model (shared for all metric calls)
# ---------------------------------------------------------------------
_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def cosine_sim(a, b):
    """Compute cosine similarity between two sentences."""
    emb_a = _embed_model.encode(a, convert_to_tensor=True)
    emb_b = _embed_model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).cpu().item())

def jaccard_overlap(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def simple_evaluate_ragas(query, answer, retrieved_docs, gold_answer):
    """
    Evaluate a single query–answer pair using simplified RAGAS-style metrics.
    
    Lightweight RAGAS-style metrics implementation for adaptive RAG evaluation.

    Computes:
    - Faithfulness: semantic similarity between answer and retrieved context.
    - Answer Relevance: similarity between query and answer.
    - Context Precision / Recall: keyword overlap between answer and retrieved context.

    """
    # Aggregate retrieved context
    context_text = " ".join(d.page_content for d in retrieved_docs) if retrieved_docs else ""

    # 1️⃣ Faithfulness — similarity between answer and context
    faithfulness = cosine_sim(answer, context_text) if context_text else 0.0

    # 2️⃣ Answer Relevance — similarity between query and answer
    answer_relevance = cosine_sim(query, answer)

    # 3️⃣ Context Precision — fraction of context words that appear in the answer
    context_precision = jaccard_overlap(answer, context_text)

    # 4️⃣ Context Recall — fraction of gold answer words covered by context
    gold_answer = gold_answer or ""
    context_recall = jaccard_overlap(gold_answer, context_text)

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(answer_relevance, 3),
        "context_precision": round(context_precision, 3),
        "context_recall": round(context_recall, 3),
    }