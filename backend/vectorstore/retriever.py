"""
Hybrid retrieval logic with memory, coreference resolution, and context-aware search.
"""

from typing import List, Dict, Optional, Tuple, Callable
from .qdrant_client import QdrantVectorStore
from .bm25_index import BM25Index
from .reranker import ChunkReranker
from ..coref_intent.coref_resolver import CorefResolver
from ..coref_intent.intent_classifier import IntentClassifier
from ..embeddings.nomic_text_embed import NomicTextEmbedder
from ..embeddings.nomic_vision_embed import NomicVisionEmbedder
from ..memory.conversation_memory import ConversationMemory
import numpy as np


class HybridRetriever:
    """Handles hybrid retrieval with memory, coreference, and context awareness."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        text_embedder: NomicTextEmbedder,
        vision_embedder: Optional[NomicVisionEmbedder] = None,
        coref_resolver: Optional[CorefResolver] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[ChunkReranker] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        memory: Optional[ConversationMemory] = None,
        query_rewriter: Optional[Callable[[str, str], str]] = None,
        max_context_chars: int = 12000
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            text_embedder: Text embedding model
            vision_embedder: Optional vision embedding model
            coref_resolver: Optional coreference resolver
            intent_classifier: Optional intent classifier
            bm25_index: Optional BM25 index for lexical search
            reranker: Optional cross-encoder reranker
            top_k: Number of final chunks to return
            similarity_threshold: Minimum similarity score for vector search
            memory: Conversation memory instance
            query_rewriter: Optional LLM-based query rewriter
            max_context_chars: Maximum context characters for LLM
        """
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.vision_embedder = vision_embedder
        self.coref_resolver = coref_resolver
        self.intent_classifier = intent_classifier
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_chars = max_context_chars

        # Memory storage for conversation context
        self.memory = memory or ConversationMemory()
        self.query_rewriter = query_rewriter
    
    def add_to_memory(self, query: str, response: str, context: List[Dict]):
        """Add conversation turn to memory."""
        self.memory.add_turn(query, response, context)
    
    def get_recent_context(self, limit: int = 2) -> List[Dict]:
        """Get recent conversation context."""
        return self.memory.get_recent_turns(limit=limit)

    def get_summary(self) -> str:
        return self.memory.get_summary()
    
    def resolve_coreference(self, query: str) -> str:
        """Resolve coreferences in query using conversation history."""
        if not self.coref_resolver:
            return query
        
        # Build context from recent memory
        context_text = ""
        recent = self.memory.get_recent_turns(limit=3)
        for turn in recent:
            context_text += f"User: {turn['query']}\nAssistant: {turn['response']}\n"
        
        # Resolve coreferences
        resolved_query = self.coref_resolver.resolve(query, context_text)
        return resolved_query
    
    def classify_intent(self, query: str) -> str:
        """Classify query intent."""
        if not self.intent_classifier:
            return "qa"  # Default intent
        
        return self.intent_classifier.classify(query)

    def rewrite_query(self, query: str) -> str:
        """Rewrite query using conversation summary if available."""
        if not self.query_rewriter:
            return query
        summary = self.get_summary()
        if not summary:
            return query
        try:
            return self.query_rewriter(query, summary)
        except Exception:
            return query

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        scores = [r["score"] for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            for r in results:
                r["normalized_score"] = 1.0
            return results
        for r in results:
            r["normalized_score"] = (r["score"] - min_s) / (max_s - min_s)
        return results
    
    def retrieve(
        self,
        query: str,
        use_coref: bool = True,
        use_intent: bool = True,
        file_id: Optional[str] = None,
        use_multimodal: bool = False,
        image_query: Optional["Image.Image"] = None,
        top_k: Optional[int] = None,
        use_history: bool = True
    ) -> Tuple[List[Dict], str, str]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            use_coref: Whether to use coreference resolution
            use_intent: Whether to use intent classification
            file_id: Optional file ID to filter results
        
        Returns:
            Tuple of (retrieved_chunks, resolved_query, intent)
        """
        # Step 1: Resolve coreferences
        resolved_query = query
        if use_coref and self.coref_resolver and use_history:
            resolved_query = self.resolve_coreference(query)

        if use_history:
            resolved_query = self.rewrite_query(resolved_query)
        
        # Step 2: Classify intent
        intent = "qa"
        if use_intent and self.intent_classifier:
            intent = self.classify_intent(resolved_query)
        
        # Step 3: Generate query embedding
        query_embedding = self.text_embedder.encode([resolved_query], show_progress=False, is_query=True)
        query_vector = query_embedding[0].tolist()
        
        # Step 4: Build filter if file_id provided
        filter_condition = None
        if file_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="file_id",
                        match=MatchValue(value=file_id)
                    )
                ]
            )
        
        # Step 5: Search vector store (text)
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        text_filter = Filter(must=[FieldCondition(key="type", match=MatchValue(value="text"))])
        if filter_condition:
            text_filter.must.extend(filter_condition.must)

        limit = top_k or self.top_k
        # Fetch more candidates for RRF fusion (will be narrowed by reranker)
        fetch_k = max(limit * 4, 20)

        search_results = self.vector_store.search(
            query_vector=query_vector,
            limit=fetch_k,
            score_threshold=self.similarity_threshold,
            filter=text_filter
        )
        
        # Step 6: Convert vector results to chunk dicts
        vector_chunks: List[Dict] = []
        for result in search_results:
            chunk = {
                "text": result["payload"].get("text", ""),
                "score": result["score"],
                "metadata": {
                    k: v for k, v in result["payload"].items()
                    if k != "text"
                }
            }
            vector_chunks.append(chunk)

        # Step 7: BM25 lexical search (if available)
        bm25_chunks: List[Dict] = []
        if self.bm25_index and self.bm25_index.is_built:
            bm25_chunks = self.bm25_index.search(
                query=resolved_query,
                top_k=fetch_k,
                file_id=file_id
            )

        # Step 8: RRF fusion (if BM25 results exist)
        if bm25_chunks:
            fused_chunks = self._rrf_fusion(vector_chunks, bm25_chunks)
        else:
            fused_chunks = vector_chunks

        # Step 9: Cross-encoder reranking (if available)
        if self.reranker and fused_chunks:
            text_chunks_for_rerank = [
                c for c in fused_chunks if c.get("text")
            ]
            reranked = self.reranker.rerank(
                query=resolved_query,
                chunks=text_chunks_for_rerank,
                top_k=limit
            )
        else:
            reranked = fused_chunks[:limit]

        # Step 10: Optional image retrieval
        image_items: List[Dict] = []
        if use_multimodal and image_query is not None and self.vision_embedder is not None:
            image_emb = self.vision_embedder.encode([image_query], show_progress=False)
            image_vector = image_emb[0].tolist()

            image_filter = Filter(must=[FieldCondition(key="type", match=MatchValue(value="image"))])
            if filter_condition:
                image_filter.must.extend(filter_condition.must)

            image_results = self.vector_store.search(
                query_vector=image_vector,
                limit=limit,
                score_threshold=self.similarity_threshold,
                filter=image_filter
            )
            for r in image_results:
                image_items.append({
                    "text": "",
                    "score": r["score"],
                    "metadata": {k: v for k, v in r["payload"].items() if k != "text"}
                })

        # Step 11: Merge text + image results
        merged = []
        for item in reranked:
            item["metadata"].setdefault("type", "text")
            merged.append(item)
        for item in image_items:
            item["metadata"].setdefault("type", "image")
            merged.append(item)

        # Deduplicate by chunk_id or image_path
        seen = set()
        deduped = []
        for item in merged:
            metadata = item.get("metadata", {})
            key = metadata.get("chunk_id") or metadata.get("image_path") or id(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        # Store retrieval stats for frontend
        self._last_retrieval_stats = {
            "vector_candidates": len(vector_chunks),
            "bm25_candidates": len(bm25_chunks),
            "fused_candidates": len(fused_chunks) if bm25_chunks else len(vector_chunks),
            "reranked_count": len(reranked),
            "image_candidates": len(image_items),
            "final_count": len(deduped)
        }
        
        return deduped, resolved_query, intent
    
    def get_last_retrieval_stats(self) -> Dict:
        """Return stats from the most recent retrieval call."""
        return getattr(self, '_last_retrieval_stats', {})

    @staticmethod
    def _rrf_fusion(
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion of vector and BM25 results.
        
        RRF score = Σ 1/(k + rank_i) for each retriever i.
        
        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            k: RRF constant (default 60).
        
        Returns:
            Fused and sorted list of chunks.
        """
        scores: Dict[str, float] = {}
        item_map: Dict[str, Dict] = {}

        for rank, item in enumerate(vector_results):
            key = item.get("metadata", {}).get("chunk_id", str(id(item)))
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in item_map:
                item_map[key] = item

        for rank, item in enumerate(bm25_results):
            key = item.get("metadata", {}).get("chunk_id", str(id(item)))
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in item_map:
                item_map[key] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for key, rrf_score in ranked:
            item = item_map[key]
            item["rrf_score"] = rrf_score
            result.append(item)
        return result
    
    def format_context_for_llm(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            if metadata.get("type") == "image":
                continue
            page_no = metadata.get("page_no", "?")
            chunk_id = metadata.get("chunk_id", "")

            block = f"[Context {i} - Page {page_no}, {chunk_id}]\n{text}\n"
            if total_chars + len(block) > self.max_context_chars:
                break
            context_parts.append(block)
            total_chars += len(block)
        
        return "\n".join(context_parts)

