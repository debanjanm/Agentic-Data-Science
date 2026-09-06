"""Runnable self-check for the Copilot's SQL sandboxing (Phase 3 hardening).

    python -m tests.test_sql_safety
"""

import pandas as pd

from agentic_ds.sql_pipeline import _run_sql


def test_select_allowed():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result, error = _run_sql("SELECT * FROM df WHERE a > 1", df)
    assert error is None
    assert len(result) == 2


def test_with_cte_allowed():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result, error = _run_sql("WITH t AS (SELECT * FROM df) SELECT * FROM t", df)
    assert error is None
    assert len(result) == 3


def test_dangerous_statements_blocked():
    df = pd.DataFrame({"a": [1, 2, 3]})
    for sql in [
        "DROP TABLE df",
        "DELETE FROM df",
        "INSERT INTO df VALUES (99)",
        "UPDATE df SET a = 0",
        "ATTACH DATABASE '/etc/passwd' AS x",
        "PRAGMA table_info(df)",
        "CREATE TABLE evil (x int)",
    ]:
        result, error = _run_sql(sql, df)
        assert result is None, f"expected '{sql}' to be blocked, but it ran"
        assert error is not None


def test_partial_word_not_blocked():
    # Word-boundary regex, not a bare substring match — "dropped" doesn't
    # trigger the "drop" keyword block.
    df = pd.DataFrame({"status": ["dropped", "active"]})
    result, error = _run_sql("SELECT * FROM df WHERE status = 'dropped'", df)
    assert error is None
    assert len(result) == 1


def test_known_false_positive_on_whole_word_in_string_literal():
    # The blocklist is a plain regex over the whole query text, not a real SQL
    # parser — a legitimate SELECT whose *string literal* happens to contain a
    # blocked word as a standalone word still gets rejected. Documenting this
    # as accepted behavior (fails closed, not open) rather than a surprise.
    df = pd.DataFrame({"status": ["drop", "active"]})
    result, error = _run_sql("SELECT * FROM df WHERE status = 'drop'", df)
    assert error is not None
    assert "disallowed keyword" in error


if __name__ == "__main__":
    test_select_allowed()
    test_with_cte_allowed()
    test_dangerous_statements_blocked()
    test_partial_word_not_blocked()
    test_known_false_positive_on_whole_word_in_string_literal()
    print("OK — all SQL-safety self-checks passed.")
