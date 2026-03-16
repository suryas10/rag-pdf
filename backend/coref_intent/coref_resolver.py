"""
Lightweight coreference resolution via pronoun detection.
Replaces the heavy biu-nlp/lingmess-coref model (which was a non-functional stub)
with a simple pronoun detector that flags queries for LLM-based rewriting.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class CorefResolver:
    """Lightweight coreference resolver using pronoun detection.
    
    Instead of loading a 400MB model that wasn't functional,
    this detects pronouns/references and flags them for
    the LLM query rewriter to handle.
    """

    # Common pronouns and referential words
    REFERENTIAL_WORDS = {
        "he", "she", "it", "they", "him", "her", "them",
        "his", "its", "their", "this", "that", "these", "those",
        "the same", "such", "former", "latter"
    }

    def __init__(self, model_name: str = None, device: Optional[str] = None):
        """Initialize lightweight coref resolver (no model loading needed)."""
        self.enabled = True
        logger.info("✅ Lightweight CorefResolver initialized (pronoun detection mode)")

    def resolve(self, query: str, context: str = "") -> str:
        """
        Check if query contains referential pronouns.
        
        Returns the original query — actual resolution is handled
        by the LLM query_rewriter which has conversation context.
        
        Args:
            query: The input text possibly containing pronouns.
            context: Optional previous text context.

        Returns:
            The original query (resolution is deferred to LLM rewriter).
        """
        return query

    def has_references(self, query: str) -> bool:
        """Check if query contains pronouns or referential language."""
        words = set(query.lower().split())
        return bool(words & self.REFERENTIAL_WORDS)

    def resolve_batch(self, queries: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        """Resolve coreferences for multiple queries."""
        return queries  # Pass through — LLM rewriter handles resolution
