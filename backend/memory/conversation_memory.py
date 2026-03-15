"""
Conversation memory with optional summarization.
Maintains recent turns and compresses older history when it grows too large.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


@dataclass
class ConversationTurn:
    query: str
    response: str
    context: List[Dict]


@dataclass
class ConversationMemory:
    max_turns: int = 10
    summarize_after: int = 8
    summary_max_chars: int = 1200
    summary: str = ""
    turns: List[ConversationTurn] = field(default_factory=list)
    summarizer: Optional[Callable[[str], str]] = None

    def add_turn(self, query: str, response: str, context: List[Dict]):
        self.turns.append(ConversationTurn(query=query, response=response, context=context))
        if len(self.turns) > self.max_turns:
            self._summarize_and_trim()

    def _summarize_and_trim(self):
        if len(self.turns) <= self.summarize_after:
            return

        older_turns = self.turns[:-self.summarize_after]
        recent_turns = self.turns[-self.summarize_after:]

        raw_text = self._format_turns(older_turns)
        summary = self._summarize_text(raw_text)

        if summary:
            self.summary = self._merge_summary(self.summary, summary)

        self.turns = recent_turns

    def _merge_summary(self, existing: str, new: str) -> str:
        if not existing:
            merged = new
        else:
            merged = f"{existing}\n{new}"
        return merged[: self.summary_max_chars]

    def _summarize_text(self, text: str) -> str:
        if not text:
            return ""
        if self.summarizer:
            return self.summarizer(text)
        # Fallback heuristic summary
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        compact = " | ".join(lines)
        return compact[: self.summary_max_chars]

    def _format_turns(self, turns: List[ConversationTurn]) -> str:
        parts = []
        for t in turns:
            parts.append(f"User: {t.query}")
            parts.append(f"Assistant: {t.response}")
        return "\n".join(parts)

    def get_recent_turns(self, limit: int = 5) -> List[Dict]:
        recent = self.turns[-limit:] if self.turns else []
        return [{"query": t.query, "response": t.response, "context": t.context} for t in recent]

    def get_summary(self) -> str:
        return self.summary

    def clear(self):
        self.turns = []
        self.summary = ""
