"""
Cross-encoder reranker for retrieved chunks.
Uses a lightweight MiniLM-based cross-encoder to re-score (query, chunk) pairs
for improved retrieval precision.
"""

import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional


class ChunkReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading reranker: {model_name} on {self.device}")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank chunks by cross-encoder relevance score.

        Args:
            query: The user query.
            chunks: List of chunk dicts with 'text' key.
            top_k: Number of top results to return.

        Returns:
            Top-k chunks sorted by reranker score (descending),
            each with added 'rerank_score' field.
        """
        if not chunks:
            return []

        # Build (query, chunk_text) pairs
        pairs = [(query, c.get("text", "")) for c in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return ranked[:top_k]
