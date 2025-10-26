# reranker.py
from sentence_transformers import CrossEncoder

class RerankerWrapper:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.enabled = False

    def toggle(self, flag: bool):
        """Enable or disable reranking."""
        self.enabled = flag

    def rerank(self, query, docs):
        """Rerank a list of LangChain Document objects."""
        if not self.enabled or not docs:
            return docs

        # Build (query, document_text) pairs
        pairs = [(query, d.page_content) for d in docs]

        # Score all documents
        scores = self.model.predict(pairs)

        # Attach scores and sort
        reranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [d for d, _ in reranked]
