import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from src.config import settings

class FAISSIndexManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FAISSIndexManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.index_dir = settings.INDEX_DIR
        self.index_path = os.path.join(self.index_dir, settings.FAISS_INDEX_FILENAME)
        self.store_path = os.path.join(self.index_dir, settings.DOCUMENTS_PERSIST_FILENAME)
        
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.id_to_metadata: Dict[int, Dict[str, Any]] = {}
        self.load_index()
        self._initialized = True

    def load_index(self):
        """Load index from disk or create a new one."""
        os.makedirs(self.index_dir, exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.store_path, "rb") as f:
                    self.id_to_metadata = pickle.load(f)
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"Failed to load FAISS index: {e}. Reinitializing...")
                self._initialize_empty_index()
        else:
            self._initialize_empty_index()

    def _initialize_empty_index(self):
        """Create a new FAISS index using Inner Product with ID mapping."""
        flat_index = faiss.IndexFlatIP(self.dimension)  # Cosine similarity (with normalized embeddings)
        self.index = faiss.IndexIDMap(flat_index)
        self.id_to_metadata = {}
        print("Initialized empty FAISS index (Inner Product with ID Map).")

    def save_index(self):
        """Persist index and metadata mapping to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump(self.id_to_metadata, f)
        print(f"Saved FAISS index ({self.index.ntotal} items) to disk.")

    def chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
        """Split document text into smaller chunks for granular vector search."""
        chunks = []
        if not text:
            return chunks
        
        words = text.split()
        current_chunk = []
        current_len = 0
        
        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current_chunk))
                overlap_count = max(1, len(current_chunk) // 5)
                current_chunk = current_chunk[-overlap_count:]
                current_len = sum(len(w) + 1 for w in current_chunk)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def add_document(self, doc_id: int, title: str, content: str):
        """Chunk a document, compute embeddings, and insert into FAISS index."""
        self.delete_document(doc_id)

        chunks = self.chunk_text(content)
        if not chunks:
            return

        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        start_id = 0
        if self.id_to_metadata:
            start_id = max(self.id_to_metadata.keys()) + 1

        new_ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
        self.index.add_with_ids(embeddings, new_ids)

        for i, idx in enumerate(new_ids):
            self.id_to_metadata[int(idx)] = {
                "doc_id": doc_id,
                "title": title,
                "chunk_index": i,
                "text": chunks[i]
            }
        
        self.save_index()

    def delete_document(self, doc_id: int):
        """Remove all chunks associated with a specific doc_id."""
        keys_to_delete = [
            k for k, v in self.id_to_metadata.items() if v["doc_id"] == doc_id
        ]
        if not keys_to_delete:
            return

        for k in keys_to_delete:
            del self.id_to_metadata[k]
        
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild index with current active metadata items."""
        if not self.id_to_metadata:
            self._initialize_empty_index()
            self.save_index()
            return

        chunks_metadata = list(self.id_to_metadata.items())
        texts = [m[1]["text"] for m in chunks_metadata]
        old_ids = [m[0] for m in chunks_metadata]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        flat_index = faiss.IndexFlatIP(self.dimension)
        new_index = faiss.IndexIDMap(flat_index)
        new_ids = np.array(old_ids, dtype=np.int64)
        new_index.add_with_ids(embeddings, new_ids)

        self.index = new_index
        self.save_index()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query vector index and return matched document chunks with metadata and scores."""
        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        scores, ids = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], ids[0]):
            idx = int(idx)
            if idx in self.id_to_metadata:
                metadata = self.id_to_metadata[idx]
                results.append({
                    "doc_id": metadata["doc_id"],
                    "title": metadata["title"],
                    "text": metadata["text"],
                    "chunk_index": metadata["chunk_index"],
                    "score": float(score)  # Cosine similarity score
                })
        return results
