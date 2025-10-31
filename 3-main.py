"""
main.py
Adaptive LMDB-based RAG pipeline with contextual bandit patch selection,
real-time cost profiling, and RAGAS evaluation.
"""

import os
import json
import time
import getpass
import logging
import asyncio
import math
from pathlib import Path
from datasets import load_from_disk, Dataset
from langchain.chat_models import init_chat_model
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from retriever import RetrieverController, build_retriever
from prompt_patch import PromptPatcher
from reranker import RerankerWrapper
from reindexer import Reindexer
from linker import extract_entities, link_entity
from kg_query import get_kg_triples, facts_to_text
from detector import detect_failures
from eval.metrics import evaluate_ragas

# NEW: Bandit and cost model
from bandit_selector import BanditPatchSelector
from cost_model import calculate_patch_cost, CostProfiler,calculate_combined_reward
from utils import analyze_bandit_log, save_trial_results, plot_cost_breakdown

# API key setup
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

# LLM and embeddings
llm_base = init_chat_model("mistral-small", model_provider="mistralai")
embeddings_base = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Wrap for RAGAS compatibility
llm_ragas = LangchainLLMWrapper(llm_base)
embeddings_ragas = LangchainEmbeddingsWrapper(embeddings_base)

# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "patch_info.log"

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

OUTPUT_LOG = LOG_DIR / "patch_trials.jsonl"
BANDIT_LOG = LOG_DIR / "bandit_history.jsonl"
COST_LOG = LOG_DIR / "cost_breakdown.jsonl"
RAGAS_LOG = LOG_DIR / "ragas_scores.jsonl"

# Controllers
prompt_patcher = PromptPatcher(config_path="configs/prompts.json")
reranker = RerankerWrapper(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reindexer = Reindexer(
    source_dir="./data_store/chroma_fever_store",
    embed_model=embeddings_base
)

# Vectorstore
vectorstore = Chroma(
    persist_directory="./data_store/chroma_fever_store",
    embedding_function=embeddings_base
)

# --- PATCH SPACE DEFINITION ---
PATCH_SPACE = [
    {"retriever_type": "dense", "k": 5, "prompt_id": "default", "rerank_on": False},
    {"retriever_type": "dense", "k": 7, "prompt_id": "verbose", "rerank_on": True},
    # {"retriever_type": "bm25", "k": 5, "prompt_id": "verifier", "rerank_on": True},
    {"retriever_type": "dense", "k": 3, "prompt_id": "verifier", "rerank_on": False},
    # {"retriever_type": "bm25", "k": 10, "prompt_id": "default", "rerank_on": False},
]

# --- DATASET LOADING ---
dataset = load_from_disk("./data/hotpotqa_1k")
formatted_dataset = []
for item in dataset:
    query = item["question"]
    answer = item["answer"]
    context = " ".join([" ".join(c[1]) for c in item["context"]])
    formatted_dataset.append({
        "query": query,
        "answer": answer,
        "context": context
    })

dataset = formatted_dataset
logging.info(f"✅ Loaded {len(dataset)} HotpotQA samples.")


# --- MAIN PATCH TRIAL LOOP WITH REAL-TIME PROFILING + RAGAS ---
async def run_patch_trials(dataset, max_samples=None):
    """Run adaptive RAG with bandit-driven patch selection and cost profiling."""
    results = []
    cost_breakdown_log = []
    ragas_eval_data = []  # Collect data for RAGAS batch evaluation
    
    # Initialize bandit
    bandit = BanditPatchSelector(patch_space=PATCH_SPACE, alpha=0.5, dim=5)
    logging.info(f"🤖 Initialized LinUCB bandit with {len(PATCH_SPACE)} arms")
    
    # Limit dataset size
    if max_samples:
        dataset = dataset[:max_samples]
        logging.info(f"⚠️ Running on first {max_samples} samples only")
    
    for sample_idx, sample in enumerate(dataset):
        query = sample["query"]
        gold_answer = sample["answer"]
        logging.info(f"{'='*80}")
        logging.info(f"🧠 Query {sample_idx+1}/{len(dataset)}: {query[:100]}...")
        
        # --- START PROFILING ---
        profiler = CostProfiler()
        profiler.start()
        
        # --- INITIAL PASS ---
        initial_retriever = RetrieverController(embeddings_base,default_type="dense", default_k=5)
        retrieved_docs = initial_retriever.retrieve(query)
        profiler.log_measurement("retrieval_initial")
        retrieved_texts = [d.page_content for d in retrieved_docs]
        
        # Extract entities & KG
        ents = extract_entities(query + " " + " ".join(retrieved_texts))
        candidates = {e: link_entity(e) for e in ents}
        qids = [q for c in candidates.values() for q in c]
        triples = get_kg_triples(qids, limit_per_q=5)
        profiler.log_measurement("kg_retrieval")
        kg_text = "".join(facts_to_text(triples))
        
        # Generate initial answer
        response, tokens_used = prompt_patcher.run_with_model(
            llm_base, query, retrieved_docs, kg_text
        )
        profiler.add_api_tokens(tokens_used)
        profiler.log_measurement("generation_initial")
        
        initial_answer = response.content.strip() if response else "ERROR: Skipped"
        
        # Detect failures
        failure_report = detect_failures(query, retrieved_texts, initial_answer)
        profiler.log_measurement("failure_detection")
        logging.info(f"🔍 Initial failure: {failure_report['failure_label']}")
        
        # --- BANDIT SELECTION ---
        selected_patch, context = bandit.get_best_patch(failure_report)
        logging.info(f"🎯 Bandit selected: {selected_patch}")
        
        # --- APPLY PATCH ---
        retriever_type = selected_patch["retriever_type"]
        k = selected_patch["k"]
        prompt_id = selected_patch["prompt_id"]
        rerank_on = selected_patch["rerank_on"]
        
        # Rebuild retriever
        if retriever_type == "bm25":
            retriever = build_retriever("bm25")
        else:
            retriever = RetrieverController(embeddings_base,default_type=retriever_type, default_k=k)
            retriever.set_k(k)
        
        prompt_patcher.use(prompt_id)
        reranker.toggle(rerank_on)
        
        # Re-retrieve with patched config
        retrieved_docs = retriever.retrieve(query)
        profiler.log_measurement("retrieval_patched")
        
        if rerank_on:
            retrieved_docs = reranker.rerank(query, retrieved_docs)
            profiler.log_measurement("reranking")
        
        retrieved_texts = [d.page_content for d in retrieved_docs]
        
        # Re-generate answer
        response, tokens_used = prompt_patcher.run_with_model(
            llm_base, query, retrieved_docs, kg_text
        )
        profiler.add_api_tokens(tokens_used)
        profiler.log_measurement("generation_patched")
        
        final_answer = response.content.strip() if response else "ERROR: Skipped"
        
        # Re-evaluate
        final_failure = detect_failures(query, retrieved_texts, final_answer)
        profiler.log_measurement("failure_detection_final")
        logging.info(f"✅ Final failure: {final_failure['failure_label']}")
        
        # --- COLLECT DATA FOR RAGAS EVALUATION ---
        ragas_eval_data.append({
            "question": query,
            "answer": final_answer,
            "contexts": retrieved_texts,
            "ground_truth": gold_answer
        })
        
        # --- STOP PROFILING ---
        #it doesnt need ragas Patch selection: contextual bandit (LinUCB/Thompson) optimizing (factuality↑, KG-consistency↑, latency/VRAM↓).
        real_metrics = profiler.stop()
        component_breakdown = profiler.get_component_breakdown()
        
        # --- REWARD CALCULATION (Binary for now, will enhance with RAGAS) ---
        reward = 1.0 if (
            final_failure.get("nli") == "ENTAILS" and
            final_failure.get("kg_aggregate") == "KG_CONSISTENT"
        ) else 0.0

        # no need
        # reward = calculate_combined_reward(
        #     failure_report=final_failure,
        #     ragas_scores=results[-1].get("ragas_scores") if results else None,  # use scores if we already have them
        #     use_ragas=True,        # toggle to False to fall back to pure binary
        #     ragas_weight=0.7      # 70 % RAGAS, 30 % binary
        # )
        
        # Calculate cost (using real metrics)
        patch_cost = calculate_patch_cost(selected_patch, real_metrics)
        
        logging.info(
            f"📊 Real metrics: latency={real_metrics['latency']:.3f}s, "
            f"memory={real_metrics['memory_gb']:.3f}GB, "
            f"tokens={real_metrics['api_tokens']}, "
            f"cost={patch_cost:.3f}"
        )
        
        # --- UPDATE BANDIT ---
        # --- UTILITY CALCULATION (Reward - Cost) ---
        lambda_param = 0.3  # How much to penalize cost
        utility = reward - lambda_param * patch_cost

        # --- UPDATE BANDIT ---
        arm_idx = PATCH_SPACE.index(selected_patch)
        bandit.update(arm=arm_idx, context=context, reward=utility, cost=patch_cost)

        logging.info(f"🎯 Utility: {utility:.3f} (reward: {reward:.3f}, cost: {patch_cost:.3f})")

        # arm_idx = PATCH_SPACE.index(selected_patch)
        # bandit.update(arm=arm_idx, context=context, reward=reward, cost=patch_cost)
        
        # --- LOGGING ---
        record = {
            "sample_idx": sample_idx,
            "query": query,
            "gold_answer": gold_answer,
            "selected_patch": selected_patch,
            "failure_before": failure_report,
            "failure_after": final_failure,
            "final_answer": final_answer,
            "reward": reward,
            "cost_normalized": patch_cost,
            "real_latency": real_metrics["latency"],
            "real_memory_gb": real_metrics["memory_gb"],
            "real_api_tokens": real_metrics["api_tokens"],
            "component_breakdown": component_breakdown,
            "timestamp": time.time()
        }
        results.append(record)
        
        # Log cost breakdown
        cost_breakdown_log.append({
            "sample_idx": sample_idx,
            "patch": selected_patch,
            "breakdown": component_breakdown,
            "total_latency": real_metrics["latency"],
            "memory_gb": real_metrics["memory_gb"],
            "api_tokens": real_metrics["api_tokens"]
        })
        
        # Save incrementally
        with open(OUTPUT_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        # Optional: trigger reindex
        if final_failure.get("kg_aggregate") == "KG_INCONSISTENT":
            logging.warning("🔄 Persistent KG inconsistency detected")
            #Uncomment to enable: 
            # reindexer.rebuild_index()
    
    # --- RAGAS BATCH EVALUATION ---
    logging.info("📊 Running RAGAS evaluation on all samples...")
    ragas_dataset = Dataset.from_list(ragas_eval_data)
    
    try:
        ragas_scores = evaluate_ragas(llm_ragas, embeddings_ragas, ragas_dataset)
        
        # Log RAGAS scores
        with open(RAGAS_LOG, "w") as f:
            json.dump(ragas_scores, f, indent=2)
        
        # Compute aggregates (handle NaN values)
        avg_ragas = {}
        for metric, values in ragas_scores.items():
            clean_values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
            avg_ragas[metric] = sum(clean_values) / len(clean_values) if clean_values else 0.0
        
        logging.info(f"✅ RAGAS scores saved to {RAGAS_LOG}")
        logging.info(f"📈 Average RAGAS scores: {json.dumps(avg_ragas, indent=2)}")
        
        # Add RAGAS scores to results
        for i, result in enumerate(results):
            result["ragas_scores"] = {k: v[i] for k, v in ragas_scores.items()}
        
    except Exception as e:
        logging.error(f"❌ RAGAS evaluation failed: {e}")
        avg_ragas = {}
    
    # --- SAVE BANDIT HISTORY ---
    bandit_history = bandit.get_history()
    with open(BANDIT_LOG, "w") as f:
        for entry in bandit_history:
            f.write(json.dumps(entry) + "\n")
    logging.info(f"✅ Bandit history saved to {BANDIT_LOG}")
    
    # --- SAVE COST BREAKDOWN ---
    with open(COST_LOG, "w") as f:
        for entry in cost_breakdown_log:
            f.write(json.dumps(entry) + "\n")
    logging.info(f"✅ Cost breakdown saved to {COST_LOG}")
    
    return results, avg_ragas


def main():
    """Main execution."""
    logging.info("🚀 Starting AutoVaccinate-RAG-Lite (Week 4: Real-Time Profiling + RAGAS)")
    
    # Run trials
    results, avg_ragas = asyncio.run(run_patch_trials(dataset, max_samples=50))
    
    logging.info(f"{'='*80}")
    logging.info(f"✅ Completed {len(results)} trials")
    
    # Save results
    save_trial_results(results, OUTPUT_LOG)
    
    # Analyze bandit performance
    logging.info("📊 Analyzing bandit performance...")
    df = analyze_bandit_log(BANDIT_LOG, PATCH_SPACE)
    
    if df is not None:
        print("" + "="*80)
        print("BANDIT PERFORMANCE SUMMARY")
        print("="*80)
        summary = df.groupby("arm").agg({
            "reward": ["mean", "std", "count"],
            "cost": ["mean", "std"]
        }).round(3)
        print(summary)
        
        success_rate = df["reward"].mean()
        avg_cost = df["cost"].mean()
        print(f"🎯 Overall Success Rate: {success_rate:.2%}")
        print(f"💰 Average Cost per Trial: {avg_cost:.3f}")
    
    # Display RAGAS results
    if avg_ragas:
        print("" + "="*80)
        print("RAGAS EVALUATION SUMMARY")
        print("="*80)
        for metric, score in avg_ragas.items():
            print(f"{metric:20s}: {score:.4f}")
    
    # Plot cost breakdown
    logging.info("📊 Generating cost breakdown visualization...")
    plot_cost_breakdown(COST_LOG)
    
    logging.info("🎉 AutoVaccinate-RAG-Lite Week 4 Complete!")
    logging.info(f"📁 Results saved to: {OUTPUT_LOG}")
    logging.info(f"📁 Bandit log: {BANDIT_LOG}")
    logging.info(f"📁 Cost breakdown: {COST_LOG}")
    logging.info(f"📁 RAGAS scores: {RAGAS_LOG}")


if __name__ == "__main__":
    main()
