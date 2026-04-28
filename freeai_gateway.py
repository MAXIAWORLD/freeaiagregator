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

    def update_usage(self, provider: str, tokens: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE quotas SET requests_used = requests_used + 1,
                                  tokens_used = tokens_used + ?
                WHERE provider = ?
            """, (tokens, provider))
            conn.commit()

    def is_available(self, provider: str, req_limit: int, tok_limit: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT requests_used, tokens_used FROM quotas WHERE provider=?", (provider,)
            ).fetchone()
        if row is None:
            return True
        return row[0] < req_limit and row[1] < tok_limit

    def save_message(self, session_id: str, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            conn.commit()

    def get_history(self, session_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversations WHERE session_id=? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def delete_history(self, session_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
            conn.commit()


class FreeIAGateway:
    def __init__(self, session_id: str = "default", db_path: str = "gateway.db"):
        self.session_id = session_id
        self.quota = QuotaManager(db_path)

    def ask(self, prompt: str, system_prompt: str = "Tu es un assistant utile.") -> str:
        self.quota.save_message(self.session_id, "user", prompt)
        history = self.quota.get_history(self.session_id)
        messages = [{"role": "system", "content": system_prompt}] + history

        for p in PROVIDERS:
            self.quota.reset_daily_if_needed(p["key"])
            if not self.quota.is_available(p["key"], p["req_limit"], p["tok_limit"]):
                logger.info(f"Quota épuisé pour {p['key']}, skip.")
                continue
            try:
                logger.info(f"Appel {p['model']}...")
                resp = litellm.completion(model=p["model"], messages=messages, timeout=30)
                answer = resp.choices[0].message.content
                tokens = resp.usage.total_tokens
                self.quota.update_usage(p["key"], tokens)
                self.quota.save_message(self.session_id, "assistant", answer)
                logger.info(f"Succès {p['key']} ({tokens} tokens)")
                return answer
            except litellm.RateLimitError:
                logger.warning(f"429 {p['key']}, fallback suivant.")
            except litellm.AuthenticationError:
                logger.warning(f"Auth error {p['key']}, fallback suivant.")
            except Exception as e:
                logger.warning(f"Erreur inattendue {p['key']}: {e}")

        raise RuntimeError("Tous les providers ont échoué ou épuisé leurs quotas.")

    def reset_conversation(self, session_id: Optional[str] = None):
        self.quota.delete_history(session_id or self.session_id)

    def get_stats(self) -> list[dict]:
        with sqlite3.connect(self.quota.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM quotas").fetchall()]


if __name__ == "__main__":
    import sys

    print("FreeIAGateway — CLI interactif")
    print("Commandes : /reset  /stats  /quit\n")

    session = input("Session ID [default]: ").strip() or "default"
    gw = FreeIAGateway(session_id=session)
    print(f"Session '{session}' démarrée.\n")

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            sys.exit(0)

        if not user_input:
            continue
        if user_input == "/quit":
            print("Au revoir.")
            break
        if user_input == "/reset":
            gw.reset_conversation()
            print("[Conversation réinitialisée]\n")
            continue
        if user_input == "/stats":
            for s in gw.get_stats():
                print(f"  {s['provider']:15} {s['requests_used']:>4} req  {s['tokens_used']:>8} tok  (date: {s['date']})")
            print()
            continue

        try:
            answer = gw.ask(user_input)
            print(f"\nAssistant : {answer}\n")
        except RuntimeError as e:
            print(f"[ERREUR] {e}\n")
