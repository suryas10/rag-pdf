"""
BM25 lexical search index for hybrid retrieval.
Complements vector search with keyword matching using Okapi BM25.
"""

import re
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


class BM25Index:
    """In-memory BM25 index for lexical search over document chunks."""

    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.documents: List[Dict] = []

    def build(self, chunks: List[Dict]):
        """
        Build BM25 index from chunk dicts.

        Args:
            chunks: List of dicts with at least a 'text' key.
        """
        self.documents = chunks
        tokenized = [self._tokenize(c.get("text", "")) for c in chunks]
        self.index = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        top_k: int = 20,
        file_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search BM25 index with a text query.

        Args:
            query: The search query.
            top_k: Maximum number of results.
            file_id: Optional file_id filter.

        Returns:
            List of chunk dicts with added 'bm25_score' field,
            sorted by descending BM25 score.
        """
        if not self.index or not self.documents:
            return []

        tokens = self._tokenize(query)
        scores = self.index.get_scores(tokens)

        # Pair scores with indices, sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue  # Skip zero-score results
            doc = self.documents[idx]
            # Apply file_id filter if provided
            if file_id:
                doc_file_id = doc.get("metadata", {}).get("file_id")
                if doc_file_id and doc_file_id != file_id:
                    continue
            results.append({**doc, "bm25_score": float(score)})
            if len(results) >= top_k:
                break

        return results

    @property
    def is_built(self) -> bool:
        """Check if index has been built."""
        return self.index is not None and len(self.documents) > 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r'\w+', text.lower())
