"""
Hybrid retrieval logic with memory, coreference resolution, and context-aware search.
"""

from typing import List, Dict, Optional, Tuple, Callable
from .qdrant_client import QdrantVectorStore
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
        top_k: int = 3,
        similarity_threshold: float = 0.7,
        memory: Optional[ConversationMemory] = None,
        query_rewriter: Optional[Callable[[str, str], str]] = None,
        max_context_chars: int = 12000
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            text_embedder: Text embedding model
            coref_resolver: Optional coreference resolver
            intent_classifier: Optional intent classifier
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.vision_embedder = vision_embedder
        self.coref_resolver = coref_resolver
        self.intent_classifier = intent_classifier
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
        
        # Step 5: Search vector store
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        text_filter = Filter(must=[FieldCondition(key="type", match=MatchValue(value="text"))])
        if filter_condition:
            text_filter.must.extend(filter_condition.must)

        limit = top_k or self.top_k
        search_results = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=self.similarity_threshold,
            filter=text_filter
        )
        
        # Step 6: Add memory context if available
        retrieved_chunks: List[Dict] = []
        for result in search_results:
            chunk = {
                "text": result["payload"].get("text", ""),
                "score": result["score"],
                "metadata": {
                    k: v for k, v in result["payload"].items()
                    if k != "text"
                }
            }
            retrieved_chunks.append(chunk)

        # Optional image retrieval
        image_results: List[Dict] = []
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

        # Normalize scores within each modality
        retrieved_chunks = self._normalize_scores(retrieved_chunks)
        image_items = []
        if image_results:
            image_items = self._normalize_scores([
                {
                    "text": "",
                    "score": r["score"],
                    "metadata": {k: v for k, v in r["payload"].items() if k != "text"}
                }
                for r in image_results
            ])

        # Merge and rerank
        merged = []
        for item in retrieved_chunks:
            item["metadata"].setdefault("type", "text")
            item["hybrid_score"] = item.get("normalized_score", item["score"])
            merged.append(item)
        for item in image_items:
            item["metadata"].setdefault("type", "image")
            item["hybrid_score"] = item.get("normalized_score", item["score"])
            merged.append(item)

        merged.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

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
        
        # Optionally add context from memory
        # This could be enhanced to include relevant past context
        
        return deduped, resolved_query, intent
    
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

