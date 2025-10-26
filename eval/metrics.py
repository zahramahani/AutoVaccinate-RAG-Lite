# eval/metrics.py
"""
Lightweight RAGAS-style metrics implementation for adaptive RAG evaluation.

Computes:
 - Faithfulness: semantic similarity between answer and retrieved context.
 - Answer Relevance: similarity between query and answer.
 - Context Precision / Recall: keyword overlap between answer and retrieved context.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

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
def evaluate_ragas(query, answer, retrieved_docs, gold_answer):
    """
    Evaluate a single query–answer pair using simplified RAGAS-style metrics.
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
# eval/metrics.py
# """
# RAGAS-based evaluation harness for adaptive RAG trials.

# Computes:
#  - Faithfulness (standard + optional HHEM classifier)
#  - Answer Accuracy (LLM-as-a-judge)
#  - Context Relevance
#  - Context Precision
#  - Context Recall

# Each metric is computed using the official RAGAS implementations.
# Supports both synchronous and asynchronous batch evaluation.

# Requirements:
#     pip install ragas==0.1.16
#     pip install openai  # or mistralai, depending on your evaluator_llm choice
# """
# import os
# os.environ["OPENAI_API_VERSION"] = "2022-12-01"
# # import os export OPENAI_API_KEY="
# # os.environ["OPENAI_API_KEY"] = ""
# import asyncio
# from ragas.dataset_schema import SingleTurnSample
# from ragas.metrics import (
#     Faithfulness,
#     FaithfulnesswithHHEM,
#     AnswerAccuracy,
#     ContextRelevance,
#     LLMContextPrecisionWithReference,
#     LLMContextRecall,
# )
# import re
# from pydantic import ValidationError
# import ragas.prompt.pydantic_prompt as pyd_prompt

# # --- Patch RAGAS JSON validation ---
# def clean_json_output(output_string: str):
#     match = re.search(r"\{[\s\S]*\}", output_string)
#     if not match:
#         raise ValueError("No JSON object found in model output.")
#     return match.group(0)

# # Save original validator
# orig_validate_json = pyd_prompt.PydanticPrompt.generate_multiple

# async def safe_generate_multiple(self, *args, **kwargs):
#     try:
#         return await orig_validate_json(self, *args, **kwargs)
#     except ValidationError as e:
#         if hasattr(e, "args") and e.args:
#             text = e.args[0]
#             try:
#                 fixed_json = clean_json_output(text)
#                 return self.output_model.model_validate_json(fixed_json)
#             except Exception:
#                 raise
#         raise

# # Patch it globally
# pyd_prompt.PydanticPrompt.generate_multiple = safe_generate_multiple



# # ---------------------------------------------------------------------
# # Evaluation Harness
# # ---------------------------------------------------------------------
# async def evaluate_ragas(
#     query: str,
#     answer: str,
#     retrieved_docs,
#     gold_answer: str,
#     evaluator_llm,
#     use_hhem: bool = False,
#     device: str = "cpu",
#     batch_size: int = 10,
# ):
#     """
#     Evaluate a single query–answer pair using official RAGAS metrics.

#     Args:
#         query (str): The user question.
#         answer (str): The model-generated answer.
#         retrieved_docs (List[Document]): Retrieved LangChain docs.
#         gold_answer (str): Reference answer (ground truth).
#         evaluator_llm: Wrapped evaluator LLM (RAGAS-compatible).
#         use_hhem (bool): If True, use FaithfulnesswithHHEM instead of vanilla Faithfulness.
#         device (str): Device for HHEM model (e.g., 'cuda:0' or 'cpu').
#         batch_size (int): Inference batch size for HHEM.

#     Returns:
#         dict: RAGAS metric scores in [0,1].
#     """
    
#     retrieved_contexts = [d.page_content for d in retrieved_docs] if retrieved_docs else []

#     # Build a single-turn sample
#     sample = SingleTurnSample(
#         user_input=query,
#         response=answer,
#         reference=gold_answer,
#         retrieved_contexts=retrieved_contexts,
#     )

#     # Initialize metrics
#     faithfulness_metric = (
#         FaithfulnesswithHHEM(device=device, batch_size=batch_size)
#         if use_hhem
#         else Faithfulness(llm=evaluator_llm)
#     )
#     accuracy_metric = AnswerAccuracy(llm=evaluator_llm)
#     relevance_metric = ContextRelevance(llm=evaluator_llm)
#     precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)
#     recall_metric = LLMContextRecall(llm=evaluator_llm)

#     # Compute all scores concurrently
#     scores = await asyncio.gather(
#         faithfulness_metric.single_turn_ascore(sample),
#         accuracy_metric.single_turn_ascore(sample),
#         relevance_metric.single_turn_ascore(sample),
#         precision_metric.single_turn_ascore(sample),
#         recall_metric.single_turn_ascore(sample),
#     )

#     return {
#         "faithfulness": round(float(scores[0]), 3),
#         "answer_accuracy": round(float(scores[1]), 3),
#         "context_relevance": round(float(scores[2]), 3),
#         "context_precision": round(float(scores[3]), 3),
#         "context_recall": round(float(scores[4]), 3),
#     }
