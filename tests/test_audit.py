"""Contract tests for the visibility audit pipeline.

Live-provider cases use httpx.MockTransport. No test contacts OpenAI,
Anthropic, Gemini, or Perplexity.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import main


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VISIBILITY_PROVIDER_MODE", "mock")
    with TestClient(main.app) as client:
        yield client


def _install_transport(handler) -> httpx.AsyncClient:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=httpx.Timeout(15.0),
    )
    main._http_client = client
    return client


def _close_transport(client: httpx.AsyncClient) -> None:
    _run(client.aclose())
    main._http_client = None


def _live_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISIBILITY_PROVIDER_MODE", "live")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-gemini")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")


def _openai_body(
    cited: bool, competitors: list[str] | None = None, raw: str | None = None
) -> dict:
    content = raw
    if content is None:
        content = json.dumps({"cited": cited, "competitors": competitors or []})
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _anthropic_body(cited: bool, competitors: list[str] | None = None) -> dict:
    text = json.dumps({"cited": cited, "competitors": competitors or []})
    return {
        "content": [
            {"type": "thinking", "thinking": "ignored"},
            {"type": "text", "text": text},
        ]
    }


def _gemini_body(cited: bool, competitors: list[str] | None = None) -> dict:
    text = json.dumps({"cited": cited, "competitors": competitors or []})
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "reasoning trace"},
                        {"text": text},
                    ]
                }
            }
        ]
    }


def _perplexity_body(
    cited: bool,
    competitors: list[str] | None = None,
    urls: list[str] | None = None,
) -> dict:
    text = json.dumps({"cited": cited, "competitors": competitors or []})
    output: list[dict[str, Any]] = []
    if urls is not None:
        output.append(
            {
                "type": "search_results",
                "results": [
                    {"id": index, "url": url, "title": "source"}
                    for index, url in enumerate(urls, start=1)
                ],
            }
        )
    output.append(
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }
    )
    return {"status": "completed", "output": output}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_vibe_share_rounds_half_up_and_ignores_failures() -> None:
    metrics = main.calculate_keyword_metrics(
        [
            {
                "provider": "openai",
                "status": "success",
                "cited": True,
                "competitors": ["Contoso Labs"],
            },
            {
                "provider": "anthropic",
                "status": "success",
                "cited": False,
                "competitors": ["Fabrikam Cloud"],
            },
            {"provider": "gemini", "status": "failed", "cited": False},
            {
                "provider": "perplexity",
                "status": "success",
                "cited": False,
                "competitors": ["Contoso Labs"],
            },
        ]
    )
    # 1 citation / 3 successful responses = 33.333... -> 33.33
    assert metrics.vibe_share_score == 33.33
    assert metrics.successful_citations == 1
    assert metrics.total_successful_provider_responses == 3
    assert metrics.competitors == ["Contoso Labs", "Fabrikam Cloud"]
    assert metrics.optimization_payload == (
        "Action Required: Brand is currently missing from Anthropic, Gemini, "
        "Perplexity search index indices. Review crawl structure."
    )


def test_score_of_exactly_50_has_no_optimization_payload() -> None:
    metrics = main.calculate_keyword_metrics(
        [
            {
                "provider": "openai",
                "status": "success",
                "cited": True,
                "competitors": [],
            },
            {
                "provider": "anthropic",
                "status": "success",
                "cited": False,
                "competitors": [],
            },
            {"provider": "gemini", "status": "failed", "cited": False},
            {"provider": "perplexity", "status": "failed", "cited": False},
        ]
    )
    assert metrics.vibe_share_score == 50.0
    assert metrics.optimization_payload is None


def test_all_providers_failed_scores_zero_and_names_every_index() -> None:
    metrics = main.calculate_keyword_metrics(
        [main._failed_result(provider) for provider in main.PROVIDERS]
    )
    assert metrics.vibe_share_score == 0.0
    assert metrics.total_successful_provider_responses == 0
    assert metrics.optimization_payload == (
        "Action Required: Brand is currently missing from OpenAI, Anthropic, "
        "Gemini, Perplexity search index indices. Review crawl structure."
    )


def test_partial_provider_failures_do_not_zero_a_real_citation() -> None:
    metrics = main.calculate_keyword_metrics(
        [
            {
                "provider": "openai",
                "status": "success",
                "cited": True,
                "competitors": [],
            },
            {
                "provider": "anthropic",
                "status": "failed",
                "cited": False,
                "failure_reason": "workspace_required",
            },
            {
                "provider": "gemini",
                "status": "failed",
                "cited": False,
                "failure_reason": "schema_validation",
            },
            {
                "provider": "perplexity",
                "status": "success",
                "cited": False,
                "competitors": [f"Competitor {index}" for index in range(8)],
            },
        ]
    )
    assert metrics.total_successful_provider_responses == 2
    assert metrics.successful_citations == 1
    assert metrics.vibe_share_score == 50.0
    assert len(metrics.competitors) == 8


def test_measured_zero_keeps_competitor_evidence() -> None:
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
                "competitors": [f"Competitor {index}" for index in range(8)],
            },
        ]
    )
    assert metrics.total_successful_provider_responses == 2
    assert metrics.vibe_share_score == 0.0
    assert len(metrics.competitors) == 8
    assert metrics.providers[1].failure_reason is None


def test_anthropic_workspace_header_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    _url, headers, _body = main._anthropic_request("system", "user")
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "regular-workspaces-preview"
    assert headers["anthropic-workspace-id"] == "wrkspc_test"
    assert headers["content-type"] == "application/json"


def test_provider_401_is_a_fastapi_unauthorized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _live_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.anthropic.com":
            return httpx.Response(
                401,
                json={
                    "error": {
                        "type": "authentication_error",
                        "message": "invalid x-api-key",
                    }
                },
            )
        if host == "api.openai.com":
            return httpx.Response(200, json=_openai_body(False, []))
        if host == "generativelanguage.googleapis.com":
            return httpx.Response(200, json=_gemini_body(False, []))
        if host == "api.perplexity.ai":
            return httpx.Response(200, json=_perplexity_body(False, []))
        raise AssertionError(host)

    client = _install_transport(handler)
    try:
        with TestClient(main.app) as api:
            response = api.post(
                "/v1/audit",
                json={
                    "client_domain": "example.com",
                    "brand_name": "Example",
                    "keywords": ["widgets"],
                },
            )
    finally:
        _close_transport(client)
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"
    assert response.content.endswith(b"\n")


def test_api_key_middle_redaction_hides_fifteen_characters() -> None:
    secret = "sk-ant-abcdefghijklmnopqrstuvwxyz"
    masked = main._redact_api_key_middle(secret)
    assert secret not in masked
    assert masked.count("*") == 15
    assert len(masked) == len(secret)
    assert masked.startswith(secret[: (len(secret) - 15) // 2])


def test_provider_error_classes() -> None:
    assert main._provider_error_class(401, "Unauthorized") == "unauthorized"
    assert (
        main._provider_error_class(403, "Your project has been denied access.")
        == "forbidden"
    )
    assert (
        main._provider_error_class(
            400,
            "this request must include the anthropic-workspace-id header",
        )
        == "workspace_required"
    )
    assert (
        main._provider_error_class(
            400,
            'Invalid value at generation_config.response_format.text.mime_type, "application/json"',
        )
        == "schema_validation"
    )


def test_audit_json_ends_with_a_newline(mock_client: TestClient) -> None:
    response = mock_client.post(
        "/v1/audit",
        json={
            "client_domain": "example.com",
            "brand_name": "Example",
            "keywords": ["example pricing"],
        },
    )
    assert response.status_code == 200
    assert response.content.endswith(b"\n")


def test_percentage_quantizes_repeating_fractions() -> None:
    assert main._percentage(1, 6) == 16.67
    assert main._percentage(2, 3) == 66.67
    assert main._percentage(1, 8) == 12.5
    assert main._percentage(0, 4) == 0.0
    assert main._percentage(1, 0) == 0.0


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_audit_normalizes_domain_and_dedupes_keywords(mock_client: TestClient) -> None:
    response = mock_client.post(
        "/v1/audit",
        json={
            "client_domain": "https://www.Example.com/pricing?q=1",
            "brand_name": "  Example   Brand  ",
            "keywords": ["example brand plans", "example brand plans", "  "],
        },
    )
    assert response.status_code == 422

    response = mock_client.post(
        "/v1/audit",
        json={
            "client_domain": "https://www.Example.com/pricing?q=1",
            "brand_name": "  Example   Brand  ",
            "keywords": ["example brand plans", "example brand plans", "other query"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["keywords_processed"] == 2
    assert list(body["results"]) == ["example brand plans", "other query"]
    plans = body["results"]["example brand plans"]
    assert plans["vibe_share_score"] == 100.0
    assert plans["optimization_payload"] is None
    assert all(
        item["evaluation_source"] == "local_auditor" for item in plans["providers"]
    )


def test_invalid_domain_and_unknown_field_are_rejected(mock_client: TestClient) -> None:
    invalid_domain = mock_client.post(
        "/v1/audit",
        json={
            "client_domain": "not a host",
            "brand_name": "Example",
            "keywords": ["widgets"],
        },
    )
    assert invalid_domain.status_code == 422

    unknown = mock_client.post(
        "/v1/audit",
        json={
            "client_domain": "example.com",
            "brand_name": "Example",
            "keywords": ["widgets"],
            "extra": True,
        },
    )
    assert unknown.status_code == 422


def test_health_reports_mock_mode_without_keys(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "PERPLEXITY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    response = mock_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["provider_mode"] == "mock"
    assert body["http_client_ready"] is True
    assert body["providers_configured"] == {
        "openai": False,
        "anthropic": False,
        "gemini": False,
        "perplexity": False,
    }


def test_mock_audit_keeps_score_consistent_with_provider_rows(
    mock_client: TestClient,
) -> None:
    response = mock_client.post(
        "/v1/audit",
        json={
            "client_domain": "example.com",
            "brand_name": "Example",
            "keywords": ["best widget platforms", "example migration guide"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["keywords_processed"] == 2
    assert "T" in body["timestamp"]
    for metrics in body["results"].values():
        providers = metrics["providers"]
        assert [item["provider"] for item in providers] == list(main.PROVIDERS)
        successes = [item for item in providers if item["status"] == "success"]
        citations = [item for item in successes if item["cited"]]
        assert metrics["total_successful_provider_responses"] == len(successes)
        assert metrics["successful_citations"] == len(citations)
        assert metrics["vibe_share_score"] == main._percentage(
            len(citations), len(successes)
        )
        if metrics["vibe_share_score"] < 50:
            assert metrics["optimization_payload"].startswith("Action Required:")
        else:
            assert metrics["optimization_payload"] is None


# ---------------------------------------------------------------------------
# Provider failures and payload parsing
# ---------------------------------------------------------------------------


def test_timeout_429_and_malformed_json_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _live_keys(monkeypatch)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        host = request.url.host
        if host == "api.openai.com":
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        if host == "api.anthropic.com":
            raise httpx.ReadTimeout("timed out")
        if host == "generativelanguage.googleapis.com":
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]},
            )
        if host == "api.perplexity.ai":
            return httpx.Response(
                200,
                json=_perplexity_body(
                    False, ["Contoso Labs"], ["https://news.example.net/story"]
                ),
            )
        raise AssertionError(host)

    client = _install_transport(handler)
    try:
        result = _run(
            main._audit_single_keyword(
                "example.com", "Example", "best widget platforms"
            )
        )
    finally:
        _close_transport(client)

    assert set(seen) == {
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.perplexity.ai",
    }
    by_provider = {item.provider: item for item in result.providers}
    assert by_provider["openai"].status == "failed"
    assert by_provider["openai"].cited is False
    assert by_provider["anthropic"].status == "failed"
    assert by_provider["gemini"].status == "failed"
    assert by_provider["perplexity"].status == "success"
    assert by_provider["perplexity"].cited is False
    assert by_provider["perplexity"].competitors == ["Contoso Labs"]
    assert result.vibe_share_score == 0.0
    assert result.total_successful_provider_responses == 1
    # Perplexity answered, but it did not cite the brand, so it is named too.
    assert result.optimization_payload == (
        "Action Required: Brand is currently missing from OpenAI, Anthropic, "
        "Gemini, Perplexity search index indices. Review crawl structure."
    )


def test_live_requests_send_the_system_prompt_and_hide_the_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _live_keys(monkeypatch)
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured[request.url.host] = request
        host = request.url.host
        if host == "api.openai.com":
            return httpx.Response(200, json=_openai_body(True, ["Contoso Labs"]))
        if host == "api.anthropic.com":
            return httpx.Response(200, json=_anthropic_body(False, ["Fabrikam Cloud"]))
        if host == "generativelanguage.googleapis.com":
            return httpx.Response(200, json=_gemini_body(True, ["Adventure Works"]))
        if host == "api.perplexity.ai":
            return httpx.Response(
                200,
                json=_perplexity_body(
                    False,
                    ["Litware Digital"],
                    ["https://www.example.com/pricing", "https://contoso.example/alt"],
                ),
            )
        raise AssertionError(host)

    client = _install_transport(handler)
    try:
        with TestClient(main.app) as api:
            response = api.post(
                "/v1/audit",
                json={
                    "client_domain": "example.com",
                    "brand_name": "Example",
                    "keywords": ["best widget platforms"],
                },
            )
    finally:
        _close_transport(client)

    assert response.status_code == 200
    body = response.json()
    metrics = body["results"]["best widget platforms"]
    # Perplexity's model said cited=false, but a retrieved URL is on example.com.
    perplexity = next(
        item for item in metrics["providers"] if item["provider"] == "perplexity"
    )
    assert perplexity["cited"] is True
    assert perplexity["evaluation_source"] == "live"
    assert metrics["successful_citations"] == 3
    assert metrics["vibe_share_score"] == 75.0
    assert metrics["optimization_payload"] is None

    openai_request = captured["api.openai.com"]
    openai_body = json.loads(openai_request.content)
    assert openai_body["messages"][0]["role"] == "system"
    assert "Example" in openai_body["messages"][0]["content"]
    assert "example.com" in openai_body["messages"][0]["content"]
    assert "best widget platforms" in openai_body["messages"][0]["content"]
    assert "JSON" in openai_body["messages"][0]["content"]
    assert openai_body["response_format"]["type"] == "json_schema"
    assert openai_request.headers["authorization"] == "Bearer sk-test-openai"

    anthropic_body = json.loads(captured["api.anthropic.com"].content)
    assert anthropic_body["system"].startswith(
        "You are an AI search-visibility auditor."
    )
    assert "temperature" not in anthropic_body
    assert anthropic_body["output_config"]["effort"] == "low"
    assert captured["api.anthropic.com"].headers["x-api-key"] == "sk-test-anthropic"

    gemini_request = captured["generativelanguage.googleapis.com"]
    assert gemini_request.url.host == "generativelanguage.googleapis.com"
    assert gemini_request.url.params["key"] == "sk-test-gemini"
    assert "x-goog-api-key" not in gemini_request.headers
    gemini_body = json.loads(gemini_request.content)
    assert "temperature" not in gemini_body["generationConfig"]
    assert gemini_body["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseFormat" not in gemini_body["generationConfig"]
    assert gemini_body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "Example" in gemini_body["system_instruction"]["parts"][0]["text"]

    perplexity_body = json.loads(captured["api.perplexity.ai"].content)
    assert perplexity_body["tools"] == [{"type": "web_search"}]
    assert perplexity_body["tool_choice"] == {"type": "web_search"}
    assert "best widget platforms" in perplexity_body["input"]
    assert "Example" in perplexity_body["instructions"]


def test_markdown_fenced_json_and_http_500(monkeypatch: pytest.MonkeyPatch) -> None:
    _live_keys(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.openai.com":
            fenced = (
                'Sure\n```json\n{"cited": false, "competitors": ["Contoso Labs"]}\n```'
            )
            return httpx.Response(200, json=_openai_body(False, raw=fenced))
        if host == "api.anthropic.com":
            return httpx.Response(500, json={"error": {"message": "unavailable"}})
        if host == "generativelanguage.googleapis.com":
            return httpx.Response(
                200, json=_gemini_body(False, ["Fabrikam Cloud", "Example"])
            )
        if host == "api.perplexity.ai":
            return httpx.Response(200, json={"status": "failed", "output": []})
        raise AssertionError(host)

    client = _install_transport(handler)
    try:
        outcome = _run(
            main.fetch_model_visibility("example.com", "Example", "widgets", "openai")
        )
        gemini = _run(
            main.fetch_model_visibility("example.com", "Example", "widgets", "gemini")
        )
        anthropic = _run(
            main.fetch_model_visibility(
                "example.com", "Example", "widgets", "anthropic"
            )
        )
        perplexity = _run(
            main.fetch_model_visibility(
                "example.com", "Example", "widgets", "perplexity"
            )
        )
    finally:
        _close_transport(client)

    assert outcome == {
        "provider": "openai",
        "status": "success",
        "cited": False,
        "competitors": ["Contoso Labs"],
        "evaluation_source": "live",
    }
    assert gemini["competitors"] == ["Fabrikam Cloud"]
    assert anthropic["provider"] == "anthropic"
    assert anthropic["status"] == "failed"
    assert anthropic["cited"] is False
    assert anthropic["failure_reason"] == "http_error"
    assert perplexity == {
        "provider": "perplexity",
        "status": "failed",
        "cited": False,
        "failure_reason": "schema_validation",
    }


def test_provider_exception_does_not_fail_the_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def flaky(
        client_domain: str, brand_name: str, keyword: str, provider: str
    ) -> dict:
        if provider == "anthropic":
            raise RuntimeError("boom")
        return {
            "provider": provider,
            "status": "success",
            "cited": True,
            "competitors": ["Other Co"],
            "evaluation_source": "live",
        }

    monkeypatch.setattr(main, "fetch_model_visibility", flaky)
    metrics = _run(main._audit_single_keyword("example.com", "Example", "widgets"))
    by_provider = {item.provider: item for item in metrics.providers}
    assert by_provider["anthropic"].status == "failed"
    assert metrics.successful_citations == 3
    assert metrics.total_successful_provider_responses == 3
    assert metrics.vibe_share_score == 100.0
    assert metrics.optimization_payload is None


def test_unknown_provider_and_missing_live_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISIBILITY_PROVIDER_MODE", "live")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    unknown = _run(
        main.fetch_model_visibility("example.com", "Example", "widgets", "bing")
    )
    assert unknown == {"provider": "bing", "status": "failed", "cited": False}

    client = _install_transport(lambda request: httpx.Response(200, json={}))
    try:
        missing = _run(
            main.fetch_model_visibility("example.com", "Example", "widgets", "openai")
        )
    finally:
        _close_transport(client)
    assert missing["provider"] == "openai"
    assert missing["status"] == "failed"
    assert missing["cited"] is False
    assert missing["failure_reason"] == "missing_api_key"


def test_local_auditor_is_stable_and_cites_direct_brand_mentions() -> None:
    first = main._local_visibility_decision(
        "gemini", "example.com", "Example", "best widget platforms"
    )
    second = main._local_visibility_decision(
        "gemini", "example.com", "Example", "best widget platforms"
    )
    assert first == second
    assert first["evaluation_source"] == "local_auditor"
    direct = main._local_visibility_decision(
        "openai", "example.com", "Example", "example pricing"
    )
    assert direct["cited"] is True
    assert "Example" not in direct["competitors"]
