"""
SQLite-backed storage for chat sessions and messages.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ChatSession:
    chat_id: str
    document_id: Optional[str]
    vector_index_id: Optional[str]
    document_name: Optional[str]
    created_at: str


class SessionStore:
    def __init__(self, db_path: str = "data/chat_sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    chat_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    vector_index_id TEXT,
                    document_name TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sources_json TEXT,
                    FOREIGN KEY(chat_id) REFERENCES sessions(chat_id) ON DELETE CASCADE
                )
                """
            )

    def create_session(self, chat_id: str, document_name: Optional[str] = None) -> ChatSession:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (chat_id, document_id, vector_index_id, document_name, created_at)
                VALUES (?, NULL, NULL, ?, ?)
                """,
                (chat_id, document_name, created_at)
            )
        return ChatSession(
            chat_id=chat_id,
            document_id=None,
            vector_index_id=None,
            document_name=document_name,
            created_at=created_at
        )

    def attach_document(self, chat_id: str, document_id: str, vector_index_id: str, document_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET document_id = ?, vector_index_id = ?, document_name = ?
                WHERE chat_id = ?
                """,
                (document_id, vector_index_id, document_name, chat_id)
            )

    def list_sessions(self) -> List[ChatSession]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chat_id, document_id, vector_index_id, document_name, created_at FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [ChatSession(**dict(row)) for row in rows]

    def get_session(self, chat_id: str) -> Optional[ChatSession]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT chat_id, document_id, vector_index_id, document_name, created_at FROM sessions WHERE chat_id = ?",
                (chat_id,)
            ).fetchone()
        return ChatSession(**dict(row)) if row else None

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (chat_id, role, content, timestamp, sources_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, role, content, ts, sources_json)
            )

    def get_history(self, chat_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp, sources_json
                FROM messages
                WHERE chat_id = ?
                ORDER BY id ASC
                """,
                (chat_id,)
            ).fetchall()
        history = []
        for row in rows:
            history.append({
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "sources": json.loads(row["sources_json"] or "[]")
            })
        return history

    def delete_session(self, chat_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))

    def clear_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")
