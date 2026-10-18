"""Billing registry, Stripe usage records, and the subscription webhook."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import httpx
import pytest
import stripe
from fastapi.testclient import TestClient

import billing
import main

SECRET = "whsec_test_secret"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def registry(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "billing.sqlite"
    monkeypatch.setenv("BILLING_REGISTRY_PATH", str(path))
    return path


def _events(path) -> list[dict]:
    if not path.exists():
        return []
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'billing_events'"
        ).fetchone()
        if table is None:
            return []
        rows = connection.execute("SELECT * FROM billing_events ORDER BY id").fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


class FakeStripe:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.error = error

    def raw_request(self, method: str, url: str, **params):
        self.calls.append((method, url, params))
        if self.error is not None:
            raise self.error
        return {
            "id": "mbur_123",
            "object": "usage_record",
            "quantity": params["quantity"],
        }


def _subscription_event(event_type: str = "customer.subscription.created") -> bytes:
    payload = {
        "id": "evt_123",
        "object": "event",
        "api_version": "2026-09-30.endive",
        "created": 1_700_000_000,
        "type": event_type,
        "data": {
            "object": {
                "id": "sub_123",
                "object": "subscription",
                "customer": "cus_123",
                "status": "canceled" if event_type.endswith("deleted") else "active",
                "metadata": {},
                "items": {
                    "object": "list",
                    "data": [
                        {
                            "id": "si_licensed",
                            "object": "subscription_item",
                            "price": {
                                "id": "price_flat",
                                "object": "price",
                                "recurring": {
                                    "interval": "month",
                                    "usage_type": "licensed",
                                },
                            },
                        },
                        {
                            "id": "si_metered",
                            "object": "subscription_item",
                            "price": {
                                "id": "price_usage",
                                "object": "price",
                                "recurring": {
                                    "interval": "month",
                                    "usage_type": "metered",
                                },
                            },
                        },
                    ],
                },
            }
        },
    }
    return json.dumps(payload).encode()


def _install_supabase(handler) -> httpx.AsyncClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=10.0)
    billing._supabase_client = client
    billing._supabase_client_owned = False
    return client


def _close_supabase(client: httpx.AsyncClient) -> None:
    _run(client.aclose())
    billing._supabase_client = None
    billing._supabase_client_owned = False


# ---------------------------------------------------------------------------
# Usage reporting
# ---------------------------------------------------------------------------


def test_usage_report_increments_with_idempotency_key(registry, monkeypatch) -> None:
    fake = FakeStripe()
    monkeypatch.setattr(billing, "get_stripe_client", lambda: fake)
    monkeypatch.setenv("STRIPE_API_KEY", "rk_test_restricted")

    _run(billing.report_api_usage_to_stripe("si_metered", 4))

    method, url, params = fake.calls[0]
    assert method == "post"
    assert url == "/v1/subscription_items/si_metered/usage_records"
    assert params["quantity"] == 4
    assert params["action"] == "increment"
    assert params["stripe_version"] == billing.USAGE_RECORD_API_VERSION
    assert params["idempotency_key"].startswith("idxgrid_usage_si_metered_")
    assert isinstance(params["timestamp"], int)
    rows = _events(registry)
    assert rows[0]["status"] == "succeeded"
    assert rows[0]["usage_count"] == 4
    assert rows[0]["idempotency_key"] == params["idempotency_key"]


def test_stripe_error_is_stored_and_not_raised(registry, monkeypatch) -> None:
    fake = FakeStripe(
        error=stripe.error.InvalidRequestError("unknown item", "subscription_item")
    )
    monkeypatch.setattr(billing, "get_stripe_client", lambda: fake)
    monkeypatch.setenv("STRIPE_API_KEY", "rk_test_restricted")

    _run(billing.report_api_usage_to_stripe("si_metered", 2))

    rows = _events(registry)
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_type"] == "InvalidRequestError"
    assert rows[0]["idempotency_key"].startswith("idxgrid_usage_si_metered_")
    assert "rk_test" not in (rows[0]["error_message"] or "")


def test_connection_timeout_is_stored(registry, monkeypatch) -> None:
    fake = FakeStripe(error=stripe.error.APIConnectionError("timed out"))
    monkeypatch.setattr(billing, "get_stripe_client", lambda: fake)
    monkeypatch.setenv("STRIPE_API_KEY", "rk_test_restricted")

    _run(billing.report_api_usage_to_stripe("si_metered", 1))

    rows = _events(registry)
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_type"] == "APIConnectionError"


def test_invalid_usage_does_not_call_stripe(registry, monkeypatch) -> None:
    fake = FakeStripe()
    monkeypatch.setattr(billing, "get_stripe_client", lambda: fake)

    _run(billing.report_api_usage_to_stripe("not valid", 3))
    _run(billing.report_api_usage_to_stripe("si_metered", 0))

    assert fake.calls == []
    rows = _events(registry)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["operation"] == "usage_report"


def test_replay_reuses_the_original_idempotency_key(registry, monkeypatch) -> None:
    failing = FakeStripe(error=stripe.error.APIConnectionError("timed out"))
    monkeypatch.setattr(billing, "get_stripe_client", lambda: failing)
    monkeypatch.setenv("STRIPE_API_KEY", "rk_test_restricted")
    _run(billing.report_api_usage_to_stripe("si_metered", 6))
    original = _events(registry)[0]

    succeeding = FakeStripe()
    monkeypatch.setattr(billing, "get_stripe_client", lambda: succeeding)
    replayed = _run(billing.replay_failed_usage_reports())

    assert replayed == 1
    _, _, params = succeeding.calls[0]
    assert params["idempotency_key"] == original["idempotency_key"]
    assert params["quantity"] == 6
    assert params["timestamp"] == original["usage_unix_timestamp"]
    assert _events(registry)[0]["status"] == "succeeded"


def test_audit_schedules_billing_for_processed_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    async def fake_bill(client_domain: str, usage_count: int) -> None:
        calls.append((client_domain, usage_count))

    monkeypatch.setenv("VISIBILITY_PROVIDER_MODE", "mock")
    monkeypatch.setattr(main, "bill_completed_audit", fake_bill)
    with TestClient(main.app) as client:
        response = client.post(
            "/v1/audit",
            json={
                "client_domain": "https://www.Example.com/pricing",
                "brand_name": "Example",
                "keywords": ["example plans", "example plans", "other query"],
            },
        )
    assert response.status_code == 200
    assert calls == [("example.com", 2)]


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


def test_webhook_rejects_a_missing_or_bad_signature(registry, monkeypatch) -> None:
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", SECRET)
    with TestClient(main.app) as client:
        missing = client.post("/v1/webhooks/stripe", content=b"{}")
        forged = client.post(
            "/v1/webhooks/stripe",
            content=b"{}",
            headers={"stripe-signature": "t=1,v1=deadbeef"},
        )
    assert missing.status_code == 400
    assert forged.status_code == 400
    assert _events(registry) == []


def test_webhook_updates_the_metered_subscription_item(registry, monkeypatch) -> None:
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "service-role-test")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=[
                {
                    "id": "4f5c2c4e-3b2a-4d1e-9c0a-a1b2c3d4e5f6",
                    "stripe_customer_id": "cus_123",
                    "stripe_subscription_item_id": "si_metered",
                }
            ],
        )

    supabase = _install_supabase(handler)
    payload = _subscription_event("customer.subscription.updated")
    header = stripe.WebhookSignature.generate_signature_header(payload.decode(), SECRET)
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/v1/webhooks/stripe",
                content=payload,
                headers={"stripe-signature": header},
            )
    finally:
        _close_supabase(supabase)

    assert response.status_code == 200
    body = response.json()
    assert body["handled"] is True
    assert body["account_id"] == "4f5c2c4e-3b2a-4d1e-9c0a-a1b2c3d4e5f6"
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "PATCH"
    assert request.url.path == "/rest/v1/accounts"
    assert request.url.params["stripe_customer_id"] == "eq.cus_123"
    assert request.headers["apikey"] == "service-role-test"
    written = json.loads(request.content)
    assert written["stripe_customer_id"] == "cus_123"
    assert written["stripe_subscription_id"] == "sub_123"
    assert written["stripe_subscription_item_id"] == "si_metered"
    assert written["subscription_status"] == "active"
    assert _events(registry) == []


def test_webhook_ignores_unhandled_events_and_retries_a_missing_account(
    registry, monkeypatch
) -> None:
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "service-role-test")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return httpx.Response(200, json=[])

    supabase = _install_supabase(handler)
    ignored = json.dumps(
        {
            "id": "evt_other",
            "object": "event",
            "type": "invoice.paid",
            "data": {"object": {}},
        }
    ).encode()
    created = _subscription_event("customer.subscription.created")
    try:
        with TestClient(main.app) as client:
            ignored_response = client.post(
                "/v1/webhooks/stripe",
                content=ignored,
                headers={
                    "stripe-signature": stripe.WebhookSignature.generate_signature_header(
                        ignored.decode(), SECRET
                    )
                },
            )
            missing = client.post(
                "/v1/webhooks/stripe",
                content=created,
                headers={
                    "stripe-signature": stripe.WebhookSignature.generate_signature_header(
                        created.decode(), SECRET
                    )
                },
            )
    finally:
        _close_supabase(supabase)

    assert ignored_response.status_code == 200
    assert ignored_response.json()["handled"] is False
    assert missing.status_code == 500
    assert seen == ["PATCH"]
    rows = _events(registry)
    assert rows[0]["operation"] == "webhook_account_update"
    assert rows[0]["status"] == "failed"
    assert rows[0]["stripe_customer_id"] == "cus_123"
    assert rows[0]["stripe_subscription_item_id"] == "si_metered"
