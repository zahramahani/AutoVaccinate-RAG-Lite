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
| **Failure Detectors** | 1. KG-consistency check  2. FEVER-style entailment (NLI-lite ,facebook/bart-large-mnli)|
| **Patch Set** | - Retriever tuning (k, BM25 vs dense, re-index) <br> - Prompt edits <br> - Reranker on/off <br> - LoRA micro-adapter (PEFT/LoRA) |
| **Patch Selector** | Contextual bandit (LinUCB / Thompson) optimizing factuality↑, KG-consistency↑, latency↓ |
| **Evaluation** | [RAGAS](https://docs.ragas.io/en/stable/) faithfulness & relevancy + task accuracy metrics |

---

## 🧱 Phases

### **Phase 1 — Core RAG Setup ✅**

- Built a minimal RAG pipeline using FEVER development split.  
- Implemented a lightweight Knowledge Graph loader (MetaQA and Wikidata slice).  
- Integrated CPU-friendly embeddings (`all-MiniLM-L6-v2`) with Chroma backend.  

**Status:** ✅ Completed  

---

### **Phase 2 — Failure Detectors ✅**

- Added **KG-consistency checker** to verify entity–relation coherence (use spacy and saved triples in lmdb).  
- Added **FEVER-style NLI entailment detector** for factual verification (facebook/bart-large-mnli)  
- Defined structured failure labels for retrieval vs generation breakdowns.  

**Status:** ✅ Completed  

---

### **Phase 3 — Patch Primitives and Evaluation Harness ✅**

- Implemented patch mechanisms:
  - Retriever parameter tuning (k, BM25 vs dense)
  - Prompt template edits
  - Reranker toggle
- Integrated **RAGAS** + task accuracy evaluation pipeline (faithfulness, nv_accuracy,nv_context_relevance).
- Established baseline metrics for later patch-selection experiments.  

**Status:** ✅ Completed  

---

### **Phase 4 — Contextual Bandit Patch Selector (In Progress)**

- Develop a bandit-based policy (LinUCB or Thompson Sampling).  
- Optimize factuality, consistency, and latency jointly under a cost budget.  
- Begin detailed logging of patch decisions and outcomes.  

**Status:**  ✅ Completed

---

### **Phase 5 — LoRA Micro-Patching (Planned)**

- Add lightweight LoRA adapters for failure shards (PEFT/LoRA).  
- Compare CPU-only vs GPU-assisted patching efficiency.  

**Status:** 🛠️ In progress

---

### **Phase 6 — Multi-hop Evaluation & Failure Taxonomy (Planned)**

- Run tests on a 1k HotpotQA subset to explore multi-hop reasoning.  
- Build taxonomy of failure and patch-response types.  

**Status:**  ✅ Completed

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

- **Python 3.10+**
- **LangChain / LlamaIndex**
- **LMDB / Chroma / SQLite**
- **SentenceTransformers**
- **Hugging Face PEFT/LoRA**
- **RAGAS**

---

## 📄 License

MIT License — open for academic and research use.

---

## 🧭 Current Progress

✅ Up to **Phase 6 completed** — RAG setup, failure detection, patch primitives, and evaluation harness, contextual-bandit patch selector under latency and cost constraints,test on 1000 samples of hotpot-qa 
🛠️ **Phase 5 underway** — Add LoRA micro-patch option.

---

## 🛠️ Setup and Execution Instructions

### 🧭 Prerequisites and Data Acquisition

Download the required datasets for knowledge graph generation:

**MetaQA Dataset**: Obtain the MetaQA dataset from the [official repository](https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g&usp=sharing).

**Wikidata Dataset**: Download the Wikidata5M dataset from the [project page](https://deepgraphlearning.github.io/project/wikidata5m#:~:text=Wikidata5m%20is%20a%20million%2Dscale,link%20prediction%20over%20unseen%20entities.) and obtain the corresponding entity and relation aliases.

**FEVER Dataset**: Acquire the FEVER dataset for retrieval documents (training and development sets) from the [official website](https://fever.ai/dataset/fever.html?utm_source=chatgpt.com).

**HotpotQA Test Data**: Execute `./data/load_hotpotqa.py` to generate a test subset of 1000 HotpotQA samples.

### 🧭 Model Configuration

Configure access to the language model by creating an account on the Mistral AI dashboard to obtain API credentials. A free API key can be obtained from [https://console.mistral.ai/home](https://console.mistral.ai/home). Alternative language models may be substituted if preferred and available.

### 🧭 Environment Setup

Create a Conda environment with Python 3.10, activate it, and install the required dependencies using `pip install -r requirements.txt`.

### 🧭 Knowledge Graph Construction

The system stores triples in SQLite databases for simple queries and LMDB databases for intensive operations.

**SQLite Database Creation**: Execute the following scripts to create SQLite databases for the specified datasets:

- `./data_store/1-kg_query_metaqa_sqlite.py`
- `./data_store/1-kg_query_wikidata_sqlite.py`

**Index Construction**: Run `./data_store/2-build_label_index.py` to create the necessary indexes.

**LMDB Database Creation**: Execute `./build_lmdb_from_sqlite.py` to generate LMDB databases from the SQLite sources.

### 🧭 Retriever Configuration

Construct the document retriever by executing `./1-retriever_builder.py`.

### 🧭 Project Execution

Launch the main application by running `./3-main.py`.
