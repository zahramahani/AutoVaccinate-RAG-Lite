"""
main_lmdb_patch.py
Adaptive LMDB-based RAG pipeline with patch primitives:
 - RetrieverController (k-tuning, retriever switch)
 - PromptPatcher (template switching)
 - RerankerWrapper (toggle cross-encoder)
 - Reindexer (refresh Chroma index)
 - Safe retry wrapper for Mistral API
"""

import os
import json
import time
import getpass
import logging
from pathlib import Path
from datasets import load_from_disk
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from retriever import RetrieverController,build_retriever
from prompt_patch import PromptPatcher
from reranker import RerankerWrapper
from reindexer import Reindexer
from linker import extract_entities, link_entity
from kg_query import get_kg_triples, facts_to_text
from detector import detect_failures
from eval.metrics import evaluate_ragas
import asyncio
# ---------------------------------------------------------------------
# CONFIG MISTERAL
# ---------------------------------------------------------------------
# from langchain_core.language_models import BaseLLM
# from langchain_core.outputs import LLMResult, Generation
# from langchain_core.messages import HumanMessage
# from langchain_mistralai import ChatMistralAI
# from pydantic import PrivateAttr
# import json
# import asyncio
# from typing import List, Any


# class MistralRagasLLM(BaseLLM):

#     """
#     Wrapper for ChatMistralAI to make it fully compatible with RAGAS evaluation pipelines.
#     """

#     _mistral_llm: ChatMistralAI = PrivateAttr()
#     temperature: float = 0.1

#     def _wrap_statements_json(self, text: str) -> str:
#         """Always return valid RAGAS StatementGeneratorOutput JSON"""
#         return json.dumps({"statements": [{"text": text.strip()}]})


#     def __init__(self, mistral_llm: ChatMistralAI, temperature: float = 0.1, **kwargs):
#         super().__init__(**kwargs)
#         self._mistral_llm = mistral_llm
#         self.temperature = temperature

#     @property
#     def _llm_type(self) -> str:
#         return "mistral-ragas-wrapper"

#     def _format_output(self, text: str) -> str:
#         try:
#             parsed = json.loads(text)
#             # If valid JSON already, return as-is
#             return json.dumps(parsed)
#         except json.JSONDecodeError:
#             # Wrap plain string in the expected RAGAS format
#             return json.dumps({"statements": [{"text": text.strip()}]})

#     def _generate(self, prompts: List[str], **kwargs: Any) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             prompt = str(prompt)
#             result = self._mistral_llm.invoke(prompt)
#             formatted = self._format_output(result.content)
#             generations.append([Generation(text=formatted)])

#         return LLMResult(generations=generations, llm_output={"token_usage": {}})

#     async def _agenerate(self, prompts: List[str], **kwargs: Any) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             prompt = str(prompt)
#             result = await self._mistral_llm.agenerate(messages=[[HumanMessage(content=prompt)]])
#             text = self._wrap_statements_json(result.generations[0][0].text.strip())
#             formatted = self._format_output(text)  # returns a plain string
#             generations.append([Generation(text=formatted)])
#         return LLMResult(generations=generations, llm_output={"token_usage": {}})

    
#     # inside MistralRagasLLM
#     async def agenerate_text(self, prompt: str, **kwargs) -> str:
#         # Call Mistral with proper batch-of-conversation format
#         resp = await self._mistral_llm.agenerate(
#             messages=[[HumanMessage(content=prompt)]]  # ✅ fix
#         )

#         # Extract text
#         text_out = resp.generations[0][0].text.strip()

#         # Ensure JSON for RAGAS
#         try:
#             parsed = json.loads(text_out)
#             if isinstance(parsed, dict):
#                 return json.dumps(parsed)
#         except json.JSONDecodeError:
#             return json.dumps({"statements": [{"text": text_out}]})

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

llm = init_chat_model("mistral-small", model_provider="mistralai")
# llm = MistralRagasLLM(unwrapped_llm)
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------
# CONFIG OPENAI
# ---------------------------------------------------------------------
# from openai import AsyncOpenAI
# from langchain_openai import ChatOpenAI

# # Make sure you have your key set or prompt securely for it
# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# # Initialize the OpenAI client
# client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# # Use GPT-4o as your LLM inside LangChain
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.2,
#     max_tokens=1024,
# )

# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------
# LOGGING CONFIG — only INFO logs go to file
# ---------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "patch_info.log"

# Create separate handlers
file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# Root logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

OUTPUT_LOG = LOG_DIR / "patch_trials.jsonl"
# ---------------------------------------------------------------------
# CONTROLLERS
# ---------------------------------------------------------------------
prompt_patcher = PromptPatcher(config_path="configs/prompts.json")
reranker = RerankerWrapper(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reindexer = Reindexer(
    source_dir="./data_store/chroma_fever_store",
    embed_model=embeddings
)

# ---------------------------------------------------------------------
# VECTORSTORE
# ---------------------------------------------------------------------
vectorstore = Chroma(
    persist_directory="./data_store/chroma_fever_store",
    embedding_function=embeddings
)

# # ---------------------------------------------------------------------
# # DATASET (FEVER-style samples with gold answers)
# # ---------------------------------------------------------------------
# dataset = [
#     {"query": "Was Colin Kaepernick a quarterback for the 49ers?", "answer": "Yes"},
#     {"query": "Who directed Moonraker?", "answer": "Lewis Gilbert"},
#     {"query": "Was The Godfather written by Mario Puzo?", "answer": "Yes"},
# ]


# ---------------------------------------------------------------------
# DATASET: Load 1k HotpotQA subset (multi-hop reasoning)
# ---------------------------------------------------------------------


# Load the saved 1k HotpotQA subset
dataset = load_from_disk("./data/hotpotqa_1k")

# Convert to a list of FEVER-style {query, answer} dicts
formatted_dataset = []
for item in dataset:
    query = item["question"]
    answer = item["answer"]
    # You can optionally join context docs for RAG retrieval evaluation
    context = " ".join([" ".join(c[1]) for c in item["context"]])
    formatted_dataset.append({
        "query": query,
        "answer": answer,
        "context": context
    })

# Final dataset for evaluation
dataset = formatted_dataset

print(f"Loaded {len(dataset)} HotpotQA samples.")
print("Example:", dataset[0])
# ---------------------------------------------------------------------
# MAIN PATCHED LOOP
# ---------------------------------------------------------------------
def run_patch_trials(dataset):
    results = []

    patch_combos = [
        ("dense", 5, "default", False),
        ("dense", 7, "verbose", True),
        ("bm25", 5, "verifier", True),
    ]

    for patch_id, (retriever_type, k, prompt_id, rerank_on) in enumerate(patch_combos):
        logging.info(f"=== Trial {patch_id}: retriever={retriever_type}, k={k}, prompt={prompt_id}, rerank={rerank_on}")

        if retriever_type=="bm25":
            retriever = build_retriever("bm25")# or dense
        else:
            retriever = RetrieverController(default_type=retriever_type, default_k=k)

        retriever.switch(retriever_type)
        retriever.set_k(k)
        prompt_patcher.use(prompt_id)
        reranker.toggle(rerank_on)

        start_time = time.time()
        trial_metrics = []

        for sample in dataset:
            query = sample["query"]
            gold_answer = sample["answer"]
            logging.info(f"🧠 Query: {query}")

            # Retrieve docs
            retrieved_docs = retriever.retrieve(query)
            if rerank_on:
                retrieved_docs = reranker.rerank(query, retrieved_docs)
            retrieved_texts = [d.page_content for d in retrieved_docs]

            # Extract & link entities
            ents = extract_entities(query + " " + " ".join(retrieved_texts))
            candidates = {e: link_entity(e) for e in ents}
            qids = [q for c in candidates.values() for q in c]

            # KG triples
            triples = get_kg_triples(qids, limit_per_q=5)
            kg_text = "\n".join(facts_to_text(triples))

            # Prompt assembly
            response = prompt_patcher.run_with_model(llm, query, retrieved_docs, kg_text)
            answer = response.content.strip() if response else "DEBUG: Skipped due to capacity issue"
            # answer = response.generations[0][0].text.strip() if response and response.generations else "DEBUG: Skipped due to capacity issue"


            # Detect failures
            result = detect_failures(query, retrieved_texts, answer)

            #ReIndex
            # if result.get("nli")!="ENTAILS" and retriever_type!="bm25":
            #     logging.warning("🔄 Reindex triggered due to KG inconsistency spike.")
            #     reindexer.rebuild_index()

            # Evaluate
            # metrics =await evaluate_ragas(query, answer, retrieved_docs, gold_answer, llm, use_hhem=False, device="cuda:0")
        
            metrics = evaluate_ragas(query, answer, retrieved_docs, gold_answer=gold_answer)
            # trial_metrics.append(metrics)

            # Logging per query
            record = {
                "query": query,
                "gold_answer": gold_answer,
                "retriever": retriever_type,
                "prompt_id": prompt_id,
                "rerank": rerank_on,
                "answer": answer,
                "kg_result": result.get("kg_aggregate"),
                "metrics": metrics,
            }
            results.append(record)

        # Aggregate metrics
        avg_metrics = {
            "faithfulness": sum(m["faithfulness"] for m in trial_metrics) / len(trial_metrics),
            "answer_relevance": sum(m["answer_relevance"] for m in trial_metrics) / len(trial_metrics),
            "context_precision": sum(m["context_precision"] for m in trial_metrics) / len(trial_metrics),
            "context_recall": sum(m["context_recall"] for m in trial_metrics) / len(trial_metrics),
        }

        log_entry = {
            "trial_id": patch_id,
            "retriever": retriever_type,
            "k": k,
            "prompt_id": prompt_id,
            "rerank": rerank_on,
            "avg_metrics": avg_metrics,
            "runtime_sec": round(time.time() - start_time, 2),
        }
        with open(OUTPUT_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return results



def main():
    results = run_patch_trials(dataset)
    logging.info(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()