"""
Intent classification for user queries.
Identifies query type (QA, summarize, compare, analyze, etc.)
"""

import re
from typing import Dict, List, Optional


class IntentClassifier:
    """Classify user query intent."""
    
    def __init__(self):
        """Initialize intent classifier with patterns and rules."""
        # Define intent patterns
        self.intent_patterns = {
            "summary": [
                r"summarize",
                r"summary",
                r"overview",
                r"brief",
                r"what.*main.*points",
                r"key.*points",
                r"gist"
            ],
            "definition": [
                r"what.*is",
                r"define",
                r"definition",
                r"meaning.*of",
                r"explain.*what"
            ],
            "fact": [
                r"when",
                r"where",
                r"who",
                r"how.*many",
                r"how.*much",
                r"what.*date",
                r"what.*time"
            ],
            "compare": [
                r"compare",
                r"difference.*between",
                r"versus",
                r"vs",
                r"similar.*to",
                r"different.*from"
            ],
            "analyze": [
                r"analyze",
                r"analysis",
                r"why",
                r"how.*does",
                r"explain.*how",
                r"what.*causes",
                r"what.*leads.*to"
            ],
            "list": [
                r"list",
                r"what.*are",
                r"name.*all",
                r"enumerate",
                r"give.*examples"
            ],
            "contextual": [
                r"according.*to",
                r"in.*document",
                r"based.*on",
                r"what.*document.*says"
            ]
        }
    
    def classify(self, query: str) -> str:
        """
        Classify query intent.
        
        Args:
            query: User query string
        
        Returns:
            Intent label (default: "qa")
        """
        query_lower = query.lower()
        
        # Score each intent
        intent_scores: Dict[str, int] = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score, or default to "qa"
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return "qa"
    
    def get_intent_info(self, intent: str) -> Dict:
        """Get information about an intent."""
        intent_descriptions = {
            "qa": "General question answering",
            "summary": "Summary or overview request",
            "definition": "Definition or explanation request",
            "fact": "Factual information request",
            "compare": "Comparison request",
            "analyze": "Analysis or explanation request",
            "list": "Listing or enumeration request",
            "contextual": "Context-specific query"
        }
        
        return {
            "intent": intent,
            "description": intent_descriptions.get(intent, "Unknown intent")
        }


class AdvancedIntentClassifier(IntentClassifier):
    """Advanced intent classifier using transformer models (optional enhancement)."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize advanced classifier.
        
        Args:
            model_name: Optional transformer model for classification
        """
        super().__init__()
        self.model = None
        # Could load a fine-tuned transformer model here
        # For now, uses rule-based approach from parent class
    
    def classify(self, query: str) -> str:
        """
        Classify using advanced model if available, else fallback to rules.
        
        Args:
            query: User query string
        
        Returns:
            Intent label
        """
        if self.model:
            # Use transformer model for classification
            # Implementation would go here
            pass
        
        # Fallback to rule-based classification
        return super().classify(query)

