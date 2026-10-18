"""Supabase audit logging and competitor displacement alerts."""

from __future__ import annotations

import logging

import pytest

import main


def _row(**overrides: object) -> dict:
    payload = {
        "client_domain": "example.com",
        "brand_name": "Example",
        "keyword_queried": "widgets",
        "vibe_share_score": 0.0,
        "competitors_detected": ["Contoso Labs"],
        "openai_status": "success",
        "perplexity_status": "success",
        "perplexity_competitors": ["Contoso Labs"],
    }
    payload.update(overrides)
    return payload


def test_zero_visibility_alert_uses_the_emergency_subject(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    main.dispatch_critical_brand_alert(
        brand_name="Example",
        keyword="widgets",
        vibe_share_score=0.0,
        openai_status="success",
        perplexity_status="success",
        perplexity_competitors=["Contoso Labs"],
        new_competitors=["Contoso Labs"],
    )
    assert (
        "🚨 INDEXGRID ALERT: Example has dropped to 0% Visibility for 'widgets'"
        in caplog.text
    )
    assert "actively recommending competitors instead of the client" in caplog.text
    assert "immediate diagnostic patch" in caplog.text


def test_supabase_timeout_does_not_escape_the_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(_row: dict) -> None:
        raise TimeoutError("supabase timed out")

    monkeypatch.setattr(main, "_insert_audit_transaction", _explode)
    monkeypatch.setattr(main, "_historical_competitor_whitelist", lambda _domain: set())
    main.asyncio.run(main.log_audit_to_supabase([_row()]))


def test_new_perplexity_competitor_alerts_when_score_is_not_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        main,
        "_historical_competitor_whitelist",
        lambda _domain: {"contoso labs"},
    )
    monkeypatch.setattr(main, "_insert_audit_transaction", lambda _row: None)
    caplog.set_level(logging.ERROR)
    main.asyncio.run(
        main.log_audit_to_supabase(
            [
                _row(
                    vibe_share_score=50.0,
                    perplexity_competitors=["Contoso Labs", "Northwind"],
                )
            ]
        )
    )
    assert "Northwind" in caplog.text
    assert "new competitor displacement" in caplog.text


def test_whitelisted_competitor_with_visibility_does_not_alert(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        main,
        "_historical_competitor_whitelist",
        lambda _domain: {"contoso labs"},
    )
    monkeypatch.setattr(main, "_insert_audit_transaction", lambda _row: None)
    caplog.set_level(logging.ERROR)
    main.asyncio.run(
        main.log_audit_to_supabase(
            [_row(vibe_share_score=50.0, perplexity_competitors=["Contoso Labs"])]
        )
    )
    assert "INDEXGRID ALERT" not in caplog.text


def test_transaction_row_uses_the_required_keyset() -> None:
    request = main.AuditRequest(
        client_domain="example.com",
        brand_name="Example",
        keywords=["widgets"],
    )
    metrics = main.calculate_keyword_metrics(
        [
            {
                "provider": "openai",
                "status": "success",
                "cited": False,
                "competitors": [],
            },
            {"provider": "anthropic", "status": "failed", "cited": False},
            {"provider": "gemini", "status": "failed", "cited": False},
            {
                "provider": "perplexity",
                "status": "success",
                "cited": False,
                "competitors": ["Contoso Labs", "Northwind"],
            },
        ]
    )
    batch = main._audit_transaction_batch(request, {"widgets": metrics})
    row = {key: batch[0][key] for key in main._AUDIT_TRANSACTION_KEYS}
    assert set(row) == set(main._AUDIT_TRANSACTION_KEYS)
    assert row["keyword_queried"] == "widgets"
    assert row["vibe_share_score"] == 0.0
    assert row["openai_status"] == "success"
    assert row["perplexity_status"] == "success"
    assert row["competitors_detected"] == ["Contoso Labs", "Northwind"]
