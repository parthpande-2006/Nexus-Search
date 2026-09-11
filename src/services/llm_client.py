import requests
from typing import List, Dict, Any
from src.config import settings

class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.openai_key = settings.OPENAI_API_KEY
        self.anthropic_key = settings.ANTHROPIC_API_KEY
        self.model = settings.LLM_MODEL

    def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        if len(chunks) == 1:
            return chunks[:top_k]

        if self.provider == "mock":
            # Simple word overlap similarity for mock re-ranking
            query_words = set(query.lower().split())
            scored_chunks = []
            for chunk in chunks:
                chunk_words = set(chunk["text"].lower().split())
                overlap = len(query_words.intersection(chunk_words))
                score = overlap + chunk.get("rrf_score", 0.0)
                scored_chunks.append({**chunk, "relevance_score": float(score)})
            
            scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scored_chunks[:top_k]

        elif self.provider == "openai":
            prompt = self._build_rerank_prompt(query, chunks)
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert search relevance grader. Output JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0
                }
                response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    import json
                    parsed = json.loads(content)
                    rankings = {item["index"]: item["score"] for item in parsed.get("rankings", [])}
                    
                    scored_chunks = []
                    for idx, chunk in enumerate(chunks):
                        score = rankings.get(idx, 0.0)
                        scored_chunks.append({**chunk, "relevance_score": float(score)})
                    
                    scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
                    return scored_chunks[:top_k]
            except Exception as e:
                print(f"OpenAI Rerank error: {e}. Falling back to RRF ordering.")
            
        elif self.provider == "anthropic":
            prompt = self._build_rerank_prompt(query, chunks)
            try:
                headers = {
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0
                }
                response = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    content = result["content"][0]["text"]
                    import json
                    start_idx = content.find("{")
                    end_idx = content.rfind("}") + 1
                    parsed = json.loads(content[start_idx:end_idx])
                    rankings = {item["index"]: item["score"] for item in parsed.get("rankings", [])}
                    
                    scored_chunks = []
                    for idx, chunk in enumerate(chunks):
                        score = rankings.get(idx, 0.0)
                        scored_chunks.append({**chunk, "relevance_score": float(score)})
                    
                    scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
                    return scored_chunks[:top_k]
            except Exception as e:
                print(f"Anthropic Rerank error: {e}. Falling back to RRF ordering.")

        return sorted(chunks, key=lambda x: x.get("rrf_score", 0.0), reverse=True)[:top_k]

    def generate_rag_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not context_chunks:
            return {
                "answer": "No relevant documents found that you are authorized to view.",
                "citations": []
            }

        context_str = ""
        citations = []
        for idx, chunk in enumerate(context_chunks):
            doc_id = chunk["doc_id"]
            title = chunk["title"]
            text = chunk["text"]
            ref_num = idx + 1
            context_str += f"Source [{ref_num}]: {title} (ID: {doc_id})\nContent: {text}\n\n"
            citations.append({
                "ref_id": ref_num,
                "doc_id": doc_id,
                "title": title,
                "snippet": text[:150] + "..." if len(text) > 150 else text
            })

        if self.provider == "mock":
            top_chunk = context_chunks[0]
            answer = (
                f"This is a simulated RAG answer. Based on the top source '{top_chunk['title']}' "
                f"[1], the document contains: '{top_chunk['text'][:150]}...'"
            )
            if len(context_chunks) > 1:
                second_chunk = context_chunks[1]
                answer += f" Additionally, source '{second_chunk['title']}' [2] outlines details indicating: '{second_chunk['text'][:100]}...'"
            
            return {"answer": answer, "citations": citations}

        prompt = (
            f"You are an assistant designed to answer questions using search results. Answer the query based ONLY on the sources below.\n"
            f"Always cite the source number (e.g. [1]) when referencing details. If you cannot answer it, say so.\n\n"
            f"--- SOURCES ---\n{context_str}\n"
            f"Query: {query}\n"
            f"Answer:"
        )

        if self.provider == "openai":
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=15)
                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    return {"answer": answer, "citations": citations}
            except Exception as e:
                print(f"OpenAI RAG Generation error: {e}")

        elif self.provider == "anthropic":
            try:
                headers = {
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                response = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=15)
                if response.status_code == 200:
                    answer = response.json()["content"][0]["text"]
                    return {"answer": answer, "citations": citations}
            except Exception as e:
                print(f"Anthropic RAG Generation error: {e}")

        return {
            "answer": "Error generating answer via LLM. Here are the raw sources:\n\n" + 
                      "\n".join([f"- {c['title']}: {c['snippet']}" for c in citations]),
            "citations": citations
        }

    def _build_rerank_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        prompt = (
            f"Grade the relevance of the following document chunks to the search query on a scale of 0.0 (unrelated) to 10.0 (perfect match).\n"
            f"Query: {query}\n\n"
            f"Respond with a JSON object in this format: {{\"rankings\": [ {{\"index\": 0, \"score\": 8.5}}, {{\"index\": 1, \"score\": 4.2}} ] }}\n"
            f"List every document index in order.\n\n"
        )
        for idx, chunk in enumerate(chunks):
            prompt += f"Document Index {idx}:\n{chunk['text']}\n---\n"
        return prompt
