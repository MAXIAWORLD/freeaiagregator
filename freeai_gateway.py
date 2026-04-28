import os
import sqlite3
import datetime
import logging
from typing import Optional

import litellm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("FreeIAGateway")

PROVIDERS = [
    {"model": "gemini/gemini-2.0-flash-exp", "key": "google",       "req_limit": 1500, "tok_limit": 1_000_000},
    {"model": "groq/llama-3.1-70b-versatile","key": "groq",         "req_limit": 500,  "tok_limit": 500_000},
    {"model": "mistral/mistral-large-latest", "key": "mistral",      "req_limit": 100,  "tok_limit": 200_000},
    {"model": "huggingface/meta-llama/Llama-3.1-70B-Instruct", "key": "huggingface", "req_limit": 1000, "tok_limit": 500_000},
]


class QuotaManager:
    def __init__(self, db_path: str = "gateway.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quotas (
                    provider TEXT PRIMARY KEY,
                    requests_used INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    date TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def reset_daily_if_needed(self, provider: str):
        today = datetime.date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT date FROM quotas WHERE provider=?", (provider,)).fetchone()
            if row is None or row[0] != today:
                conn.execute("""
                    INSERT OR REPLACE INTO quotas (provider, requests_used, tokens_used, date)
                    VALUES (?, 0, 0, ?)
                """, (provider, today))
                conn.commit()
