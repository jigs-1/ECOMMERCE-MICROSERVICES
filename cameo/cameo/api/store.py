from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
from threading import Lock
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"{salt.hex()}:{digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    salt_hex, digest_hex = encoded.split(":", 1)
    expected = _hash_password(password, bytes.fromhex(salt_hex)).split(":", 1)[1]
    return hmac.compare_digest(expected, digest_hex)


@dataclass
class SessionUser:
    user_id: int
    username: str
    created_at: str


class DataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    confidence REAL NOT NULL,
                    response_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )

    def create_user(self, username: str, password: str) -> SessionUser:
        normalized = username.strip().lower()
        if len(normalized) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters.")
        created_at = _utc_now()
        password_hash = _hash_password(password)
        try:
            with self.lock, self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (normalized, password_hash, created_at),
                )
                user_id = int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists.") from exc
        return SessionUser(user_id=user_id, username=normalized, created_at=created_at)

    def authenticate_user(self, username: str, password: str) -> Optional[SessionUser]:
        normalized = username.strip().lower()
        with self.lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
                (normalized,),
            ).fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            return None
        return SessionUser(user_id=int(row["id"]), username=row["username"], created_at=row["created_at"])

    def create_session(self, user: SessionUser) -> str:
        token = secrets.token_urlsafe(32)
        with self.lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
                (token, user.user_id, _utc_now()),
            )
        return token

    def get_user_by_token(self, token: str) -> Optional[SessionUser]:
        with self.lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.created_at
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ?
                """,
                (token,),
            ).fetchone()
        if not row:
            return None
        return SessionUser(user_id=int(row["id"]), username=row["username"], created_at=row["created_at"])

    def record_prediction(
        self,
        user_id: int,
        text: str,
        emotion: str,
        intent: str,
        intensity: float,
        confidence: float,
        response_text: str,
    ) -> None:
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions
                (user_id, text, emotion, intent, intensity, confidence, response_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, text, emotion, intent, intensity, confidence, response_text, _utc_now()),
            )

    def get_history(self, user_id: int, limit: int = 12) -> list[dict]:
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, text, emotion, intent, intensity, confidence, response_text, created_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "text": row["text"],
                "emotion": row["emotion"],
                "intent": row["intent"],
                "intensity": float(row["intensity"]),
                "confidence": float(row["confidence"]),
                "response_text": row["response_text"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_trends(self, user_id: int) -> dict:
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT emotion, intent, intensity, confidence, created_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 50
                """,
                (user_id,),
            ).fetchall()
        if not rows:
            return {
                "total_checks": 0,
                "top_emotion": None,
                "top_intent": None,
                "average_intensity": 0.0,
                "average_confidence": 0.0,
                "recent_emotions": [],
                "recent_intents": [],
            }
        emotions = [row["emotion"] for row in rows]
        intents = [row["intent"] for row in rows]
        avg_intensity = sum(float(row["intensity"]) for row in rows) / len(rows)
        avg_confidence = sum(float(row["confidence"]) for row in rows) / len(rows)
        return {
            "total_checks": len(rows),
            "top_emotion": Counter(emotions).most_common(1)[0][0],
            "top_intent": Counter(intents).most_common(1)[0][0],
            "average_intensity": round(avg_intensity, 4),
            "average_confidence": round(avg_confidence, 4),
            "recent_emotions": emotions[:5],
            "recent_intents": intents[:5],
        }
