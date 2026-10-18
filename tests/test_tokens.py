"""Admin token issuance. Supabase writes are stubbed."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

import main
from routes.tokens import generate_agency_token, hash_agency_token


class _TokenHarness:
    def __init__(self, client: TestClient, inserted: list[dict]) -> None:
        self.client = client
        self.inserted = inserted


@pytest.fixture
def token_client(monkeypatch: pytest.MonkeyPatch) -> _TokenHarness:
    monkeypatch.setenv("INDEXGRID_MASTER_KEY", "test-master-key")
    inserted: list[dict] = []

    def _insert(row: dict) -> None:
        inserted.append(row)

    monkeypatch.setattr("routes.tokens._insert_agency_token", _insert)
    with TestClient(main.app) as client:
        yield _TokenHarness(client, inserted)


def test_generate_requires_the_admin_secret(token_client: _TokenHarness) -> None:
    missing = token_client.client.post(
        "/v1/tokens/generate",
        json={"client_email": "ada@example.com", "agency_name": "Northwind"},
    )
    wrong = token_client.client.post(
        "/v1/tokens/generate",
        headers={"X-Admin-Secret": "nope"},
        json={"client_email": "ada@example.com", "agency_name": "Northwind"},
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert token_client.inserted == []


def test_generate_returns_a_live_trial_token_and_stores_only_the_hash(
    token_client: _TokenHarness,
) -> None:
    response = token_client.client.post(
        "/v1/tokens/generate",
        headers={"X-Admin-Secret": "test-master-key"},
        json={
            "client_email": "Ada@Example.com",
            "agency_name": "  Northwind   Labs  ",
        },
    )
    assert response.status_code == 200
    assert response.content.endswith(b"\n")
    body = response.json()
    token = body["token"]
    assert token.startswith("ig_live_trial_")
    assert len(token.removeprefix("ig_live_trial_")) == 24
    assert body["client_email"] == "ada@example.com"
    assert body["agency_name"] == "Northwind Labs"
    assert body["allotted_credits"] == 1000
    stored = token_client.inserted[0]
    assert stored["token_hash"] == hashlib.sha256(token.encode()).hexdigest()
    assert token not in str(stored)
    assert stored["allotted_credits"] == 1000
    assert stored["metadata"]["client_email"] == "ada@example.com"
    assert stored["metadata"]["agency_name"] == "Northwind Labs"


def test_token_generator_uses_24_hex_characters() -> None:
    token = generate_agency_token()
    suffix = token.removeprefix("ig_live_trial_")
    assert len(suffix) == 24
    assert hash_agency_token(token) == hashlib.sha256(token.encode()).hexdigest()
