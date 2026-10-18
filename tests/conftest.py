"""Keep the suite off the live Supabase project."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _skip_live_supabase_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "main._historical_competitor_whitelist",
        lambda _domain: set(),
    )

    def _insert(row: dict) -> None:
        row.setdefault("_test_inserted", True)

    monkeypatch.setattr("main._insert_audit_transaction", _insert)
