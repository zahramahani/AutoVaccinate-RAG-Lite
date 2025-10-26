import getpass
import os
import json
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from linker import extract_entities, link_entity
from kg_query import get_kg_triples, facts_to_text
from detector import detect_failures


import time

def safe_invoke(llm, prompt, retries=5, delay=5):
    """
    Retry wrapper for Mistral API calls to handle temporary 429 / capacity errors.
    """
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            msg = str(e)
            if "capacity" in msg.lower() or "429" in msg:
                print(f"⚠️ [Attempt {attempt}/{retries}] Mistral capacity issue, retrying in {delay * attempt}s...")
                time.sleep(delay * attempt)
            else:
                print(f"❌ Mistral error (non-retryable): {e}")
                raise
    print("🚫 Mistral API unavailable after multiple retries — skipping call.")
    return None  # fallback to None if all retries fail

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

llm = init_chat_model("mistral-small", model_provider="mistralai")

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./data_store/chroma_fever_store",
    embedding_function=embeddings
)
# ---------------------------------------------------------------------
# FILM-RELATED CLAIMS (FEVER-like)
# ---------------------------------------------------------------------
queries = [
    "Was Colin Kaepernick a quarterback for the 49ers?",  # control non-film
    "Who directed Moonraker?",
    "Did Marlene Dietrich star in Kismet?",
    "Is Inception a film directed by Christopher Nolan?",
    "Did Leonardo DiCaprio appear in Titanic?",
    "Is Avatar the highest-grossing film directed by James Cameron?",
    "Was The Dark Knight released before 2010?",
    "Was Natalie Portman the main actress in Black Swan?",
    "Was The Godfather written by Mario Puzo?",
    "Was Star Wars released after 1975?"
]

# ---------------------------------------------------------------------
# OUTPUT FILE
# ---------------------------------------------------------------------
output_path = "results_film.jsonl"

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
with open(output_path, "w", encoding="utf-8") as f_out:
    for query in queries:
        print("=" * 80)
        print(f"🎥 QUERY: {query}")
        print("=" * 80)

        # Step 1: Retrieve docs
        retrieved_docs = vectorstore.similarity_search(query, k=5)
        retrieved_texts = [d.page_content for d in retrieved_docs]

        print("\n📘 Retrieved Documents:")
        for i, doc in enumerate(retrieved_texts, 1):
            print(f"  [{i}] {doc[:500]}{'...' if len(doc) > 500 else ''}")

        # Step 2: Extract entities
        ents = extract_entities(query + " " + " ".join(retrieved_texts))
        print("\n🔍 Extracted Entities:")
        for e in ents:
            print(f"  - {e}")

        # Step 3: Link entities
        candidates = {e: link_entity(e) for e in ents}
        print("\n🔗 Linked QIDs:")
        for e, qids in candidates.items():
            print(f"  {e}: {qids}")

        qids = [q for c in candidates.values() for q in c]

        # print("qids are ",qids)
        # Step 4: KG triples (LMDB)
        triples = get_kg_triples(qids, limit_per_q=5)
        kg_text = "\n".join(facts_to_text(triples))

        print("\n🧩 KG Triples (sample):")
        # for t in triples[:10]:
        #     print(f"  {t}")
        print(kg_text)

        
        # Step 5: Build prompt and query LLM
        prompt = rf"""
            Context:
            {kg_text}

            Retrieved passages:
            {"\n\n".join(retrieved_texts)}

            Question: {query}
            Answer (use KG facts if they help; otherwise say NOT ENOUGH INFO):"""

        response = safe_invoke(llm, prompt)

        if response is None:
            print("⚠️ Skipped Mistral call due to capacity issue.")
            response = "DEBUG: Skipped due to Mistral overload."
        answer = response.content.strip()

        print("\n🤖 LLM Response:")
        print(answer)

        # Step 6: Run detectors
        result = detect_failures(query, retrieved_texts, answer)
        print("\n--- DETECTOR RESULTS ---")
        print("NLI:", result.get("nli"))
        print("KG aggregate:", result.get("kg_aggregate"))
        print("Failure label:", result.get("failure_label"))

        # Step 7: Save all
        record = {
            "query": query,
            "retrieved_docs": retrieved_texts,
            "entities": ents,
            "linked_qids": candidates,
            "triples": triples,
            "kg_text": kg_text,
            "llm_answer": answer,
            "detector_results": result
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("\n✅ Saved to results_film.jsonl\n\n")

print(f"\n🎯 Done! All results written to {output_path}")
