"""
Hybrid retrieval logic with memory, coreference resolution, and context-aware search.
"""

from typing import List, Dict, Optional, Tuple
from .qdrant_client import QdrantVectorStore
from ..coref_intent.coref_resolver import CorefResolver
from ..coref_intent.intent_classifier import IntentClassifier
from ..embeddings.nomic_text_embed import NomicTextEmbedder
import numpy as np


class HybridRetriever:
    """Handles hybrid retrieval with memory, coreference, and context awareness."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        text_embedder: NomicTextEmbedder,
        coref_resolver: Optional[CorefResolver] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        top_k: int = 3,
        similarity_threshold: float = 0.7
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
        self.coref_resolver = coref_resolver
        self.intent_classifier = intent_classifier
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Memory storage for conversation context
        self.memory: List[Dict] = []
    
    def add_to_memory(self, query: str, response: str, context: List[Dict]):
        """Add conversation turn to memory."""
        self.memory.append({
            "query": query,
            "response": response,
            "context": context,
            "timestamp": None  # Could add timestamp
        })
        # Keep only last N turns (prevent memory from growing too large)
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]
    
    def get_recent_context(self, limit: int = 2) -> List[Dict]:
        """Get recent conversation context."""
        return self.memory[-limit:] if self.memory else []
    
    def resolve_coreference(self, query: str) -> str:
        """Resolve coreferences in query using conversation history."""
        if not self.coref_resolver:
            return query
        
        # Build context from recent memory
        context_text = ""
        if self.memory:
            recent = self.memory[-3:]  # Last 3 turns
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
    
    def retrieve(
        self,
        query: str,
        use_coref: bool = True,
        use_intent: bool = True,
        file_id: Optional[str] = None
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
        if use_coref and self.coref_resolver:
            resolved_query = self.resolve_coreference(query)
        
        # Step 2: Classify intent
        intent = "qa"
        if use_intent and self.intent_classifier:
            intent = self.classify_intent(resolved_query)
        
        # Step 3: Generate query embedding
        query_embedding = self.text_embedder.encode([resolved_query], show_progress=False)
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
        search_results = self.vector_store.search(
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.similarity_threshold,
            filter=filter_condition
        )
        
        # Step 6: Add memory context if available
        retrieved_chunks = []
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
        
        # Optionally add context from memory
        # This could be enhanced to include relevant past context
        
        return retrieved_chunks, resolved_query, intent
    
    def format_context_for_llm(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            page_no = metadata.get("page_no", "?")
            chunk_id = metadata.get("chunk_id", "")
            
            context_parts.append(
                f"[Context {i} - Page {page_no}, {chunk_id}]\n{text}\n"
            )
        
        return "\n".join(context_parts)

