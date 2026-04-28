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


def test_save_and_get_history(qm):
    qm.save_message("sess1", "user", "Bonjour")
    qm.save_message("sess1", "assistant", "Hello")
    history = qm.get_history("sess1")
    assert history == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_get_history_empty_for_new_session(qm):
    assert qm.get_history("new_session") == []


def test_delete_history_removes_only_target_session(qm):
    qm.save_message("sess1", "user", "message A")
    qm.save_message("sess2", "user", "message B")
    qm.delete_history("sess1")
    assert qm.get_history("sess1") == []
    assert len(qm.get_history("sess2")) == 1


from unittest.mock import MagicMock, patch
from freeai_gateway import FreeIAGateway
import litellm


def _make_response(content="réponse", total_tokens=100):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.total_tokens = total_tokens
    return resp


@pytest.fixture
def gw(db_path):
    return FreeIAGateway(session_id="test", db_path=db_path)


def test_ask_returns_response_from_first_provider(gw):
    with patch("freeai_gateway.litellm.completion", return_value=_make_response("Bonjour")) as mock_completion:
        result = gw.ask("Salut")
    assert result == "Bonjour"
    assert mock_completion.call_count == 1


def test_ask_falls_back_on_rate_limit_error(gw):
    call_count = 0
    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise litellm.RateLimitError(message="429", llm_provider="google", model="gemini")
        return _make_response("fallback")

    with patch("freeai_gateway.litellm.completion", side_effect=side_effect):
        result = gw.ask("test")
    assert result == "fallback"
    assert call_count == 2


def test_ask_raises_when_all_providers_fail(gw):
    with patch("freeai_gateway.litellm.completion", side_effect=litellm.RateLimitError(message="429", llm_provider="x", model="y")):
        with pytest.raises(RuntimeError, match="Tous les providers"):
            gw.ask("test")


def test_ask_saves_history(gw):
    with patch("freeai_gateway.litellm.completion", return_value=_make_response("réponse")):
        gw.ask("ma question")
    history = gw.quota.get_history("test")
    assert history[0] == {"role": "user", "content": "ma question"}
    assert history[1] == {"role": "assistant", "content": "réponse"}


def test_ask_skips_exhausted_provider(gw, db_path):
    qm = gw.quota
    qm.reset_daily_if_needed("google")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE quotas SET requests_used=1500 WHERE provider='google'")
        conn.commit()

    with patch("freeai_gateway.litellm.completion", return_value=_make_response("ok")) as mock_c:
        result = gw.ask("test")
    first_model = mock_c.call_args_list[0][1]["model"]
    assert "groq" in first_model
    assert result == "ok"


def test_get_stats_returns_list(gw):
    with patch("freeai_gateway.litellm.completion", return_value=_make_response()):
        gw.ask("test")
    stats = gw.get_stats()
    assert isinstance(stats, list)
    assert len(stats) >= 1
    assert "provider" in stats[0]
    assert "requests_used" in stats[0]
    assert "tokens_used" in stats[0]


def test_reset_conversation_clears_history(gw):
    with patch("freeai_gateway.litellm.completion", return_value=_make_response()):
        gw.ask("message test")
    gw.reset_conversation()
    assert gw.quota.get_history("test") == []


def test_reset_conversation_other_session(gw):
    gw.quota.save_message("autre", "user", "msg")
    gw.reset_conversation("autre")
    assert gw.quota.get_history("autre") == []
