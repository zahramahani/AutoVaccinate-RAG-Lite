import os
import shutil
import logging
from langchain_community.vectorstores import Chroma

class Reindexer:
    def __init__(self, source_dir="./data_store/chroma_fever_store", embed_model=None):
        self.source_dir = source_dir
        self.embed_model = embed_model
        self.version = 1

    def rebuild_index(self, docs=None):
        """
        Rebuild the Chroma index when stale or inconsistent.
        - If docs is provided → re-embed and create new index.
        - Otherwise → clear and rebuild empty index placeholder.
        """
        logging.info(f"🔁 Rebuilding Chroma index at {self.source_dir}")

        # Remove old index
        if os.path.exists(self.source_dir):
            backup_path = f"{self.source_dir}_v{self.version}_backup"
            shutil.move(self.source_dir, backup_path)
            logging.info(f"📦 Old index backed up to {backup_path}")

        # Create new Chroma index
        os.makedirs(self.source_dir, exist_ok=True)
        if docs:
            texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
            Chroma.from_texts(
                texts=texts,
                embedding=self.embed_model,
                persist_directory=self.source_dir,
                metadatas=metadatas,
            )
            logging.info(f"✅ New Chroma index built with {len(texts)} documents.")
        else:
            # Initialize empty Chroma store (placeholder)
            Chroma(
                persist_directory=self.source_dir,
                embedding_function=self.embed_model
            )
            logging.info(f"✅ Empty Chroma index initialized.")

        self.version += 1
        logging.info(f"🆕 Index version now {self.version}")
        return self.version

    def get_index_version(self):
        return f"index_v{self.version}"
