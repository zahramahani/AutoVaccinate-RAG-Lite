from langchain_community.embeddings import SentenceTransformerEmbeddings
from retriever import build_retriever
import logging

if __name__ == "__main__":
    embeddings_base = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    retriever = build_retriever(embeddings_base, "dense")# or dense
    test_query = "Was Albert Einstein a physicist?"
    logging.info(f"🧠 Testing retriever on query: {test_query}")
    results = retriever.retrieve(test_query)
    for i, doc in enumerate(results[:3], 1):
        logging.info(f"Result {i}: {doc.page_content[:180]}...")

# 2025-10-20 16:28:42,130 [WARNING] Retrying in 1s [Retry 1/5].
# 2025-10-20 16:29:03,422 [INFO] 📦 RetrieverController initialized: type=bm25, k=5
# 2025-10-20 16:29:03,422 [INFO] 🚀 Building BM25 retriever...
# 2025-10-20 16:29:04,129 [INFO] ✅ BM25 retriever built from 19998 documents.
# 2025-10-20 16:29:04,129 [INFO] ✅ BM25 retriever built and ready to use
# 2025-10-20 16:29:04,129 [INFO] 🎉 Retriever build completed successfully.
# 2025-10-20 16:31:04,689 [INFO] BM25 retrieved 4 docs.
# 2025-10-20 16:31:04,690 [INFO] Result 1: Albert S. Ruddy is a singer....
# 2025-10-20 16:31:04,690 [INFO] Result 2: Albert S. Ruddy is a television actor....
# 2025-10-20 16:31:04,690 [INFO] Result 3: Albert S. Ruddy is only a television producer....