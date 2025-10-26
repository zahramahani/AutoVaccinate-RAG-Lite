# AutoVaccinate-RAG-Lite
A lightweight Retrieval-Augmented Generation (RAG) prototype that detects its own factual or regression failures and applies the most cost-effective corrective patch under knowledge-graph constraints.
---

## 🚀 Overview

**AutoVaccinate-RAG-Lite** explores the idea of **auto-immunizing** a RAG system — i.e., enabling it to:
1. Detect its own factual or consistency failures.
2. Diagnose the likely failure mode (retrieval vs generation).
3. Apply a minimal and cost-bounded patch (retriever tuning, prompt edits, reranker toggle, or LoRA micro-adapter).

The goal is to demonstrate measurable factuality improvements (e.g., FEVER or HotpotQA) while maintaining low latency and VRAM cost.

---

## 🧩 Core Components

| Component | Description |
|------------|-------------|
| **Datasets** | [FEVER](https://fever.ai/dataset/fever.html), [MetaQA](https://github.com/yuyuz/MetaQA),1k HotpotQA subset |
| **Knowledge Graph (KG)** | Tiny domain KG from Wikidata slice |
| **RAG Stack** |  LangChain + Chroma + SentenceTransformer (`all-MiniLM-L6-v2`) |
| **Failure Detectors** | 1. KG-consistency check  2. FEVER-style entailment (NLI-lite)|
| **Patch Set** | - Retriever tuning (k, BM25 vs dense, re-index) <br> - Prompt edits <br> - Reranker on/off <br> - LoRA micro-adapter (PEFT/LoRA) |
| **Patch Selector** | Contextual bandit (LinUCB / Thompson) optimizing factuality↑, KG-consistency↑, latency↓ |
| **Evaluation** | [RAGAS](https://docs.ragas.io/en/stable/) faithfulness & relevancy + task accuracy metrics |

---

## 🧱 Phases

### **Phase 1 — Core RAG Setup ✅**
- Built a minimal RAG pipeline using FEVER development split.  
- Implemented a lightweight Knowledge Graph loader (MetaQA or Wikidata slice).  
- Integrated CPU-friendly embeddings (`all-MiniLM-L6-v2`) with Chroma backend.  

**Status:** ✅ Completed  

---

### **Phase 2 — Failure Detectors ✅**
- Added **KG-consistency checker** to verify entity–relation coherence.  
- Added **FEVER-style NLI entailment detector** for factual verification.  
- Defined structured failure labels for retrieval vs generation breakdowns.  

**Status:** ✅ Completed  

---

### **Phase 3 — Patch Primitives and Evaluation Harness ✅**
- Implemented patch mechanisms:
  - Retriever parameter tuning (k, BM25 vs dense)
  - Prompt template edits
  - Reranker toggle
- Integrated **RAGAS** + task accuracy evaluation pipeline.
- Established baseline metrics for later patch-selection experiments.  

**Status:** ✅ Completed  

---

### **Phase 4 — Contextual Bandit Patch Selector (In Progress)**
- Develop a bandit-based policy (LinUCB or Thompson Sampling).  
- Optimize factuality, consistency, and latency jointly under a cost budget.  
- Begin detailed logging of patch decisions and outcomes.  

**Status:** 🛠️ In progress  

---

### **Phase 5 — LoRA Micro-Patching (Planned)**
- Add lightweight LoRA adapters for failure shards (PEFT/LoRA).  
- Compare CPU-only vs GPU-assisted patching efficiency.  

**Status:** ⏳ Planned  

---

### **Phase 6 — Multi-hop Evaluation & Failure Taxonomy (Planned)**
- Run tests on a 1k HotpotQA subset to explore multi-hop reasoning.  
- Build taxonomy of failure and patch-response types.  

**Status:** ⏳ Planned  

---

### **Phase 7 (Stretch) — Self-RAG Critique or RAGBench/THELMA Evaluators**
- Experiment with light Self-RAG-style critique or external evaluation frameworks.  

**Status:** ⏳ Optional stretch goal  

---

## 🧪 Deliverables

- **Codebase:** RAG with KG checker, bandit patcher, and evaluation harness  
- **Repro Kit:** Scripts for KG creation, indexing, and patch experiments  
- **Mini-paper:** 4–6 pages detailing cost-bounded patch selection  
- **Ethics Note:** Data/model attribution, license compliance, and safe content handling  

---

## 🎯 Success Criteria

- ≥ **5–10 point gain** on FEVER or HotpotQA accuracy or RAGAS faithfulness vs static RAG.  
- Bandit demonstrates **adaptive patch selection** (distinct strategies per failure type).  

---

## 🧰 Tech Stack

- **Python 3.12+**
- **LangChain / LlamaIndex**
- **FAISS / Chroma / SQLite**
- **SentenceTransformers**
- **Hugging Face PEFT/LoRA**
- **RAGAS**

---

## 📄 License
MIT License — open for academic and research use.

---

## 🧭 Current Progress

✅ Up to **Phase 3 completed** — RAG setup, failure detection, patch primitives, and evaluation harness.  
🛠️ **Phase 4 underway** — contextual-bandit patch selector under latency and cost constraints.

