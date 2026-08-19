"""
SQLite storage for conversations, messages, and future logs.

Swap this module later for Postgres / a vector DB without changing the UI.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DB_PATH = Path(os.environ.get("SQLITE_PATH", "data/cosmic.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    attachments TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_logs_kind
    ON logs(kind, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row_to_conversation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _parse_attachments(raw: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def make_title(message: str, attachments: list[dict[str, Any]] | None = None) -> str:
    text = " ".join((message or "").split())
    if text:
        return text[:42] + ("…" if len(text) > 42 else "")
    attachments = attachments or []
    if attachments:
        name = str(attachments[0].get("name") or "").strip()
        return name[:42] if name else "Attachment"
    return "New Chat"


def list_conversations(limit: int = 40) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_conversation(row) for row in rows]


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        messages = conn.execute(
            "SELECT id, role, content, attachments, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, id ASC",
            (conversation_id,),
        ).fetchall()
    return {
        **_row_to_conversation(row),
        "messages": [
            {
                "id": item["id"],
                "role": item["role"],
                "content": item["content"],
                "attachments": _parse_attachments(item["attachments"]),
                "created_at": item["created_at"],
            }
            for item in messages
        ],
    }


def create_conversation(title: str = "New Chat") -> dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    stamp = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title, stamp, stamp),
        )
    return {
        "id": conversation_id,
        "title": title,
        "created_at": stamp,
        "updated_at": stamp,
    }


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stamp = _now()
    message_id = str(uuid.uuid4())
    payload = json.dumps(attachments or [], ensure_ascii=False)
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if exists is None:
            raise KeyError(conversation_id)
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, attachments, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content, payload, stamp),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (stamp, conversation_id),
        )
    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "attachments": attachments or [],
        "created_at": stamp,
    }


def set_conversation_title(conversation_id: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conversation_id),
        )


def delete_conversation(conversation_id: str) -> bool:
    with get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return cursor.rowcount > 0


def add_log(kind: str, payload: dict[str, Any] | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO logs (id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), kind, json.dumps(payload or {}, ensure_ascii=False), _now()),
        )
