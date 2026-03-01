"""Local user account and analysis history management via SQLite."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join("data", "app_history.db")


def _connect() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role_template TEXT,
                overall_score REAL,
                ats_score REAL,
                data_json TEXT NOT NULL,
                FOREIGN KEY(username) REFERENCES users(username)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> bool:
    init_db()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            return False
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, _hash_password(password), datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> bool:
    init_db()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return False
        return row[0] == _hash_password(password)
    finally:
        conn.close()


def save_analysis(username: str, role_template: str, payload: Dict[str, Any]) -> None:
    init_db()
    conn = _connect()
    try:
        cur = conn.cursor()
        score = float(payload.get("score_details", {}).get("overall", 0.0))
        ats = float(payload.get("ats_details", {}).get("ats_score", 0.0))
        cur.execute(
            """
            INSERT INTO analyses (username, created_at, role_template, overall_score, ats_score, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                datetime.utcnow().isoformat(),
                role_template,
                score,
                ats,
                json.dumps(payload),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_history(username: str, limit: int = 30) -> List[Dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, created_at, role_template, overall_score, ats_score
            FROM analyses
            WHERE username = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (username, limit),
        )
        rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "role_template": row[2],
                "overall_score": row[3],
                "ats_score": row[4],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_history_record(record_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT data_json FROM analyses WHERE id = ?", (record_id,))
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])
    finally:
        conn.close()

