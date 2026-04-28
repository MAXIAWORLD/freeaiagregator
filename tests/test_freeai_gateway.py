import pytest
import sqlite3
import os
import tempfile
import datetime
from freeai_gateway import QuotaManager


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def qm(db_path):
    return QuotaManager(db_path)


def test_init_creates_tables(db_path):
    qm = QuotaManager(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "quotas" in tables
    assert "conversations" in tables


def test_reset_daily_inserts_row_if_absent(qm, db_path):
    qm.reset_daily_if_needed("google")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT requests_used, tokens_used, date FROM quotas WHERE provider=?", ("google",)).fetchone()
    assert row is not None
    assert row[0] == 0
    assert row[1] == 0
    assert row[2] == datetime.date.today().isoformat()


def test_reset_daily_resets_if_old_date(qm, db_path):
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO quotas (provider, requests_used, tokens_used, date) VALUES (?, 50, 9000, ?)", ("google", yesterday))
        conn.commit()
    qm.reset_daily_if_needed("google")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT requests_used, tokens_used FROM quotas WHERE provider=?", ("google",)).fetchone()
    assert row == (0, 0)


def test_reset_daily_does_not_reset_if_same_day(qm, db_path):
    today = datetime.date.today().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO quotas (provider, requests_used, tokens_used, date) VALUES (?, 10, 500, ?)", ("google", today))
        conn.commit()
    qm.reset_daily_if_needed("google")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT requests_used, tokens_used FROM quotas WHERE provider=?", ("google",)).fetchone()
    assert row == (10, 500)


def test_update_usage_increments_counters(qm, db_path):
    qm.reset_daily_if_needed("groq")
    qm.update_usage("groq", tokens=150)
    qm.update_usage("groq", tokens=50)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT requests_used, tokens_used FROM quotas WHERE provider=?", ("groq",)).fetchone()
    assert row == (2, 200)


def test_is_available_true_when_under_limit(qm):
    qm.reset_daily_if_needed("groq")
    assert qm.is_available("groq", req_limit=500, tok_limit=500_000) is True


def test_is_available_false_when_requests_exceeded(qm, db_path):
    qm.reset_daily_if_needed("groq")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE quotas SET requests_used=500 WHERE provider=?", ("groq",))
        conn.commit()
    assert qm.is_available("groq", req_limit=500, tok_limit=500_000) is False


def test_is_available_false_when_tokens_exceeded(qm, db_path):
    qm.reset_daily_if_needed("groq")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE quotas SET tokens_used=500001 WHERE provider=?", ("groq",))
        conn.commit()
    assert qm.is_available("groq", req_limit=500, tok_limit=500_000) is False
