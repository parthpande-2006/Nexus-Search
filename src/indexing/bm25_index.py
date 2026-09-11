import os
import pickle
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.config import settings

class BM25IndexManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BM25IndexManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.index_dir = settings.INDEX_DIR
        self.index_path = os.path.join(self.index_dir, settings.BM25_INDEX_FILENAME)
        
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self.load_index()
        self._initialized = True

    def tokenize(self, text: str) -> List[str]:
        """Convert text to lowercase and split by word characters."""
        if not text:
            return []
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def load_index(self):
        """Load BM25 database from pickle file and construct BM25 index."""
        os.makedirs(self.index_dir, exist_ok=True)
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self.chunks = pickle.load(f)
                self._rebuild_bm25()
                print(f"Loaded BM25 index with {len(self.chunks)} chunks.")
            except Exception as e:
                print(f"Failed to load BM25 index: {e}. Reinitializing...")
                self.chunks = []
                self.bm25 = None
        else:
            self.chunks = []
            self.bm25 = None

    def save_index(self):
        """Persist BM25 database to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Saved BM25 index ({len(self.chunks)} chunks) to disk.")

    def _rebuild_bm25(self):
        """Reinitialize BM25Okapi with the current corpus of tokens."""
        if not self.chunks:
            self.bm25 = None
            return
        corpus_tokens = [chunk["tokens"] for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def add_document(self, doc_id: int, title: str, content: str, chunks: List[str]):
        """Ingest new document chunks into the BM25 database and rebuild the index."""
        self.delete_document(doc_id, save=False)

        for text in chunks:
            tokens = self.tokenize(text)
            self.chunks.append({
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "tokens": tokens
            })
        
        self._rebuild_bm25()
        self.save_index()

    def delete_document(self, doc_id: int, save: bool = True):
        """Remove document chunks from database."""
        initial_count = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk["doc_id"] != doc_id]
        
        if len(self.chunks) != initial_count:
            self._rebuild_bm25()
            if save:
                self.save_index()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform BM25 search on query and return sorted list of matched chunks."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        scored_chunks = []
        for i, score in enumerate(scores):
            if score > 0:
                chunk = self.chunks[i]
                scored_chunks.append({
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "score": float(score)
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
