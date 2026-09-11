from typing import List, Dict, Any
from sqlalchemy.orm import Session

from src.indexing.faiss_index import FAISSIndexManager
from src.indexing.bm25_index import BM25IndexManager
from src.security import filter_documents_by_acl
from src.services.llm_client import LLMClient
from src.models import AuditLog

class SearchEngine:
    def __init__(self):
        self.vector_index = FAISSIndexManager()
        self.keyword_index = BM25IndexManager()
        self.llm_client = LLMClient()

    def search(
        self,
        db: Session,
        query: str,
        user_id: int,
        user_role: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute permissions-aware hybrid search (Vector + Keyword) using RRF,
        filters results by ACL permissions, and re-ranks the top results.
        """
        retrieve_limit = max(20, top_k * 4)
        vector_results = self.vector_index.search(query, top_k=retrieve_limit)
        keyword_results = self.keyword_index.search(query, top_k=retrieve_limit)

        # Reciprocal Rank Fusion (RRF)
        rrf_const = 60
        rrf_scores = {}

        for rank, hit in enumerate(vector_results):
            key = (hit["doc_id"], hit["title"], hit["text"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_const + rank + 1))

        for rank, hit in enumerate(keyword_results):
            key = (hit["doc_id"], hit["title"], hit["text"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_const + rank + 1))

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_candidates:
            self._log_audit(db, user_id, "query", query, None, "allowed")
            return []

        # Security Filtering
        candidate_doc_ids = list({key[0] for key, _ in sorted_candidates})
        allowed_doc_ids = filter_documents_by_acl(db, candidate_doc_ids, user_id, user_role)

        filtered_candidates = []
        for key, rrf_score in sorted_candidates:
            doc_id, title, text = key
            if doc_id in allowed_doc_ids:
                filtered_candidates.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "rrf_score": rrf_score
                })

        top_allowed_doc = filtered_candidates[0]["doc_id"] if filtered_candidates else None
        self._log_audit(db, user_id, "query", query, top_allowed_doc, "allowed")

        # Re-ranking
        if rerank and filtered_candidates:
            candidates_to_rerank = filtered_candidates[:15]
            re_ranked = self.llm_client.rerank_chunks(query, candidates_to_rerank, top_k=top_k)
            return re_ranked
        
        return filtered_candidates[:top_k]

    def chat_rag(
        self,
        db: Session,
        query: str,
        user_id: int,
        user_role: str
    ) -> Dict[str, Any]:
        """Secured RAG pipeline with citation answers."""
        top_chunks = self.search(db, query, user_id, user_role, top_k=4, rerank=True)
        
        top_doc_id = top_chunks[0]["doc_id"] if top_chunks else None
        self._log_audit(db, user_id, "rag_chat", query, top_doc_id, "allowed")

        rag_response = self.llm_client.generate_rag_answer(query, top_chunks)
        return {
            "query": query,
            "answer": rag_response["answer"],
            "citations": rag_response["citations"]
        }

    def _log_audit(self, db: Session, user_id: int, action: str, query: str, doc_id: int, status: str):
        try:
            log = AuditLog(
                user_id=user_id,
                action=action,
                query_string=query,
                document_id=doc_id,
                status=status
            )
            db.add(log)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Failed to write audit log: {e}")
