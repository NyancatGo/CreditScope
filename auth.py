from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from passlib.context import CryptContext


ROOT_DIR = Path(__file__).resolve().parent
_AUTH_DB_OVERRIDE = os.environ.get("CREDITSCOPE_AUTH_DB")
AUTH_DB_PATH = Path(_AUTH_DB_OVERRIDE) if _AUTH_DB_OVERRIDE else ROOT_DIR / "auth.db"
SESSION_COOKIE_NAME = "creditscope_session"
SESSION_MAX_AGE_SECONDS = 8 * 60 * 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_auth_db() -> None:
    with _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_token_hash
                ON sessions (token_hash);

            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
                ON sessions (expires_at);
            """
        )


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def purge_expired_sessions() -> None:
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (_timestamp(_utc_now()),))


def user_count() -> int:
    init_auth_db()
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        return int(row["count"])


def has_users() -> bool:
    return user_count() > 0


def create_first_user(username: str, password: str) -> dict[str, Any]:
    init_auth_db()
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")
    if has_users():
        raise PermissionError("Registration is disabled after the first admin user is created.")

    now = _timestamp(_utc_now())
    with _db() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), now),
        )
        row = conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    init_auth_db()
    with _db() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        return _row_to_dict(row)


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    user = get_user_by_username(username)
    if user is None:
        return None
    if not verify_password(password, str(user["password_hash"])):
        return None
    return {
        "id": user["id"],
        "username": user["username"],
        "created_at": user["created_at"],
    }


def create_session(user_id: int) -> str:
    init_auth_db()
    token = secrets.token_urlsafe(32)
    now = _utc_now()
    expires_at = now + timedelta(seconds=SESSION_MAX_AGE_SECONDS)
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO sessions (user_id, token_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, _hash_token(token), _timestamp(now), _timestamp(expires_at)),
        )
    return token


def get_user_by_session_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    init_auth_db()
    purge_expired_sessions()
    with _db() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username, users.created_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
              AND sessions.expires_at > ?
            """,
            (_hash_token(token), _timestamp(_utc_now())),
        ).fetchone()
        return _row_to_dict(row)


def delete_session(token: str | None) -> None:
    if not token:
        return
    init_auth_db()
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))


init_auth_db()
