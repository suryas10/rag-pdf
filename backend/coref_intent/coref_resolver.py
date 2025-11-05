"""
Coreference resolution using Hugging Face's LingMessCoref model.
Replaces spaCy-coref/coreferee dependency for improved compatibility.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Optional
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorefResolver:
    """Resolve coreferences using biu-nlp/lingmess-coref."""

    def __init__(
        self,
        model_name: str = "biu-nlp/lingmess-coref",
        device: Optional[str] = None
    ):
        """
        Initialize LingMessCoref coreference resolver.

        Args:
            model_name: The Hugging Face model name.
            device: 'cuda' or 'cpu'. Auto-detects if not set.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"🔍 Initializing LingMessCoref ({self.model_name}) on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            logger.info("✅ LingMessCoref initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LingMessCoref: {e}")
            self.model = None
            self.tokenizer = None

    def resolve(self, query: str, context: str = "") -> str:
        """
        Resolve coreferences in a query with optional context.

        Args:
            query: The input text possibly containing pronouns.
            context: Optional previous text context.

        Returns:
            The resolved text with pronouns replaced (placeholder for now).
        """
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ Coref model not initialized. Returning original text.")
            return query

        text = f"{context}\n{query}" if context else query
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # NOTE: LingMessCoref produces token classification logits, not directly resolved text.
            # For now, return the original query until decoding logic is added.
            # (You can extend this later to map clusters → replacements.)
            logger.info("ℹ️ Coref model executed successfully, returning original text for now.")
            return query

        except Exception as e:
            logger.error(f"❌ Coref resolution failed: {e}")
            return query

    def resolve_batch(self, queries: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        """
        Resolve coreferences for multiple queries.

        Args:
            queries: List of user inputs.
            contexts: List of context strings (optional).

        Returns:
            List of resolved strings.
        """
        if contexts is None:
            contexts = [""] * len(queries)
        return [self.resolve(q, c) for q, c in zip(queries, contexts)]


if __name__ == "__main__":
    resolver = CorefResolver()
    text = "Alice met Bob. She told him about the project."
    print("Before:", text)
    print("After:", resolver.resolve(text))
