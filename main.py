#!/usr/bin/env python3
"""AI Visibility Indexing API.

A standalone FastAPI service that estimates how often major answer engines
cite a brand for a set of keywords. For every keyword the service fans out
to OpenAI, Anthropic, Gemini, and Perplexity at the same time, then reduces
those four observations into a Vibe-Share Score.

Live calls share one httpx.AsyncClient with a hard 15-second timeout. A
provider that times out, rate-limits, or returns anything other than a
usable citation judgment is recorded as failed and does not abort the rest
of the audit. When a provider key is absent, that provider is scored by a
deterministic local auditor so the process can be exercised without secrets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal
from urllib.parse import quote_plus, urlparse

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from billing import (
    bill_completed_audit,
    billing_status,
    start_billing,
    stop_billing,
)
from billing import (
    router as billing_router,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "1.0.0"
USER_AGENT = f"AI-Visibility-Index/{VERSION}"
REQUEST_TIMEOUT_SECONDS = 15.0

# Stable fan-out order. Metrics, logs, and the optimization sentence all
# follow this sequence so repeated audits are diffable.
PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini", "perplexity")

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "perplexity": "Perplexity",
}

PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}

# The local auditor only names these fictional companies. They are a stable
# stand-in for "other brands a model might mention", never live web data.
_LOCAL_COMPETITOR_CATALOG: tuple[str, ...] = (
    "Northwind Analytics",
    "Contoso Labs",
    "Fabrikam Cloud",
    "Adventure Works",
    "Litware Digital",
    "Wide World Importers",
)

_HOSTNAME_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOSTNAME_RE = re.compile(rf"^(?:{_HOSTNAME_LABEL}\.)+{_HOSTNAME_LABEL}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

OPTIMIZATION_TEMPLATE = (
    "Action Required: Brand is currently missing from {providers} "
    "search index indices. Review crawl structure."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per line without pulling in a logging extra."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event_fields = getattr(record, "event_fields", None)
        if isinstance(event_fields, dict):
            payload.update(event_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


LOGGER = logging.getLogger("visibility_index")


def configure_logging() -> None:
    """Attach a single JSON handler to this service logger.

    The handler is installed on a named logger with propagate disabled so
    uvicorn's own configuration cannot drop these events or print secrets
    that were never meant to reach a formatter.
    """

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        LOGGER.addHandler(handler)


def _log(level: int, message: str, **fields: Any) -> None:
    """Log a structured event. Field values must never include credentials."""

    LOGGER.log(level, message, extra={"event_fields": fields})


configure_logging()

# ---------------------------------------------------------------------------
# Process-wide outbound client
# ---------------------------------------------------------------------------

# One client for every provider. Created in the FastAPI lifespan and closed
# on shutdown. Tests may assign a transport-backed client before startup;
# the lifespan will not replace an existing instance.
_http_client: httpx.AsyncClient | None = None
_outbound_semaphore: asyncio.Semaphore | None = None
_outbound_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _flag(name: str, default: str = "false") -> bool:
    return _env(name, default).lower() in {"1", "true", "yes", "on"}


def _bounded_int(name: str, default: int, lower: int, upper: int) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(value, lower), upper)


def _provider_mode() -> Literal["auto", "live", "mock"]:
    mode = _env("VISIBILITY_PROVIDER_MODE", "auto").lower()
    if mode == "live":
        return "live"
    if mode == "mock":
        return "mock"
    return "auto"


def _api_key(provider: str) -> str:
    """Return the configured key for a provider, or an empty string.

    Gemini also accepts GOOGLE_API_KEY because that is the name Google
    documents for the same credential. The value is only placed on an
    outbound header, never logged.
    """

    key = _env(PROVIDER_KEY_ENV[provider])
    if key:
        return key
    if provider == "gemini":
        return _env("GOOGLE_API_KEY")
    return ""


def _use_local_auditor(provider: str) -> bool:
    """Decide whether this call should skip the network.

    ``mock`` forces the local auditor for every provider. ``live`` never
    uses it, so a missing key surfaces as a failed provider result. ``auto``
    calls a provider only when its key is present.
    """

    mode = _provider_mode()
    if mode == "mock":
        return True
    if mode == "live":
        return False
    return _api_key(provider) == ""


def _build_http_client() -> httpx.AsyncClient:
    """Create the shared client.

    The timeout covers connect, write, read, and pool wait. Fifteen seconds
    is intentional and is not read from the environment: a slow provider
    must become a failed observation rather than stall the audit. Redirects
    are disabled so an Authorization header cannot be replayed to another
    host. Environment proxy variables are ignored unless explicitly opted
    into, for the same reason.
    """

    return httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
        follow_redirects=False,
        trust_env=_flag("VISIBILITY_TRUST_ENV", "false"),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )


def _require_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError(
            "HTTP client is not initialized. Start the application so the lifespan can open it."
        )
    return _http_client


def _semaphore_for_current_loop() -> asyncio.Semaphore:
    """Bound in-flight provider calls on the running event loop.

    A semaphore is bound to the loop that created it. Recreate it when the
    loop changes (tests call ``asyncio.run`` more than once) so a later
    loop does not trip "bound to a different event loop".
    """

    global _outbound_semaphore, _outbound_semaphore_loop
    loop = asyncio.get_running_loop()
    if _outbound_semaphore is None or _outbound_semaphore_loop is not loop:
        _outbound_semaphore = asyncio.Semaphore(
            _bounded_int("MAX_OUTBOUND_CONCURRENCY", 16, 1, 200)
        )
        _outbound_semaphore_loop = loop
    return _outbound_semaphore


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _max_keywords() -> int:
    return _bounded_int("MAX_KEYWORDS", 15, 1, 100)


def _normalize_domain(value: str) -> str:
    """Reduce a user-supplied domain or URL to a canonical hostname."""

    candidate = value.strip().lower()
    candidate = re.sub(r"^[a-z][a-z0-9+.-]*://", "", candidate)
    candidate = candidate.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if "@" in candidate:
        candidate = candidate.rsplit("@", 1)[-1]
    if candidate.startswith("["):
        raise ValueError("client_domain must be a hostname such as example.com")
    if ":" in candidate:
        host, _, port = candidate.rpartition(":")
        if port.isdigit():
            candidate = host
    if candidate.startswith("www.") and candidate.count(".") >= 2:
        candidate = candidate[4:]
    candidate = candidate.strip(".")
    if not candidate:
        raise ValueError("client_domain is empty")
    try:
        candidate = candidate.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("client_domain is not a valid hostname") from exc
    if not _HOSTNAME_RE.fullmatch(candidate):
        raise ValueError("client_domain must be a hostname such as example.com")
    return candidate


def _normalize_brand(value: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError("brand_name is empty")
    if len(cleaned) > 200:
        raise ValueError("brand_name must be 200 characters or fewer")
    return cleaned


def _normalize_keywords(values: list[str]) -> list[str]:
    """Trim keywords, drop exact duplicates, and enforce the configured cap."""

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        keyword = " ".join(raw.split())
        if not keyword:
            raise ValueError("keywords cannot contain blank items")
        if len(keyword) > 200:
            raise ValueError("each keyword must be 200 characters or fewer")
        if any(ord(char) < 32 for char in keyword):
            raise ValueError("keywords cannot contain control characters")
        if keyword in seen:
            continue
        seen.add(keyword)
        cleaned.append(keyword)
    if not cleaned:
        raise ValueError("at least one keyword is required")
    limit = _max_keywords()
    if len(cleaned) > limit:
        raise ValueError(f"at most {limit} keywords are accepted per audit")
    return cleaned


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AuditRequest(BaseModel):
    """Keyword audit requested for one brand and domain."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "client_domain": "example.com",
                    "brand_name": "Example",
                    "keywords": ["example pricing", "best widget platforms"],
                }
            ]
        },
    )

    client_domain: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Hostname to look for in answers. A URL is accepted and reduced to its host.",
    )
    brand_name: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Brand a typical answer would have to name for a citation to count.",
    )
    keywords: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Search phrases to audit. Exact duplicates are removed. The default cap is 15.",
    )

    @field_validator("client_domain")
    @classmethod
    def validate_client_domain(cls, value: str) -> str:
        return _normalize_domain(value)

    @field_validator("brand_name")
    @classmethod
    def validate_brand_name(cls, value: str) -> str:
        return _normalize_brand(value)

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, value: list[str]) -> list[str]:
        return _normalize_keywords(value)


class ProviderResult(BaseModel):
    """One provider's citation judgment for a single keyword."""

    model_config = ConfigDict(extra="ignore")

    provider: str
    status: Literal["success", "failed"]
    cited: bool
    competitors: list[str] = Field(default_factory=list)
    evaluation_source: Literal["live", "local_auditor"] | None = None
    failure_reason: str | None = None


class KeywordMetrics(BaseModel):
    """Reduced visibility metrics for one keyword."""

    model_config = ConfigDict(extra="ignore")

    vibe_share_score: float = Field(..., ge=0, le=100)
    successful_citations: int = Field(..., ge=0)
    total_successful_provider_responses: int = Field(..., ge=0)
    providers: list[ProviderResult]
    competitors: list[str]
    optimization_payload: str | None = None


class AuditResponse(BaseModel):
    """Completed multi-keyword visibility audit."""

    timestamp: str
    keywords_processed: int = Field(..., ge=0)
    results: dict[str, KeywordMetrics]


class HealthResponse(BaseModel):
    """Process health and which providers currently have a configured key."""

    status: Literal["ok"]
    provider_mode: Literal["auto", "live", "mock"]
    providers_configured: dict[str, bool]
    http_client_ready: bool
    billing: dict[str, bool]


class ServiceInfo(BaseModel):
    """Discovery payload for the service root."""

    service: str
    version: str
    audit: str
    health: str
    docs: str
    webhook: str


# ---------------------------------------------------------------------------
# Prompts and response schema
# ---------------------------------------------------------------------------


def _citation_json_schema(*, strict: bool) -> dict[str, Any]:
    """JSON schema every provider is asked to fill.

    OpenAI strict mode requires ``additionalProperties: false``. Gemini's
    structured-output subset rejects that keyword, so callers choose.
    """

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "cited": {
                "type": "boolean",
                "description": (
                    "True only when the target brand or the target domain "
                    "would be named, linked, or recommended in a typical answer."
                ),
            },
            "competitors": {
                "type": "array",
                "description": (
                    "Other brands or domains a typical answer would mention, "
                    "excluding the target brand and target domain."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["cited", "competitors"],
    }
    if strict:
        schema["additionalProperties"] = False
    return schema


def build_system_prompt(client_domain: str, brand_name: str, keyword: str) -> str:
    """Build the system prompt that forces a citation judgment.

    The word JSON is included because OpenAI JSON mode rejects a request
    whose messages never mention JSON. The same instruction is sent to
    every provider so the four judgments are comparable.
    """

    return (
        "You are an AI search-visibility auditor. Judge the answer an AI "
        "assistant would typically give to a person searching for the keyword "
        "below. Base the judgment on what that assistant would actually name, "
        "not on what would be flattering to the target brand.\n"
        f"Target brand name: {brand_name}\n"
        f"Target domain: {client_domain}\n"
        f"Keyword: {keyword}\n"
        "Rules:\n"
        "- Set cited to true only when that typical answer would name the "
        "brand, show the domain, or recommend either one.\n"
        "- Set cited to false when a typical answer would cover the topic "
        "without naming the brand or the domain.\n"
        "- Do not treat a generic description of the category as a citation.\n"
        "- competitors lists other brands or domains that typical answer would "
        "mention. Exclude the target brand and the target domain. Use an empty "
        "array when no other names would be mentioned. Do not invent names you "
        "would not actually expect in the answer.\n"
        "Return one JSON object and nothing else. The object must contain "
        'exactly the keys "cited" (boolean) and "competitors" (array of strings). '
        "Do not wrap the JSON in markdown fences. Do not add a preamble or a closing remark."
    )


def build_user_prompt(client_domain: str, brand_name: str, keyword: str) -> str:
    """Build the user turn that states the concrete search scenario."""

    return (
        "Evaluate this search scenario and answer with the required JSON object only.\n"
        f'A user asks an AI assistant: "{keyword}"\n'
        f"Would {brand_name} ({client_domain}) be cited in a typical answer, "
        "and which competing brands would be cited alongside or instead of it?"
    )


# ---------------------------------------------------------------------------
# Provider request builders
# ---------------------------------------------------------------------------


def _openai_model() -> str:
    return _env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"


def _anthropic_model() -> str:
    return _env("ANTHROPIC_MODEL", "claude-haiku-5-5") or "claude-haiku-5-5"


def _gemini_model() -> str:
    return _env("GEMINI_MODEL", "gemini-3.8-flash") or "gemini-3.8-flash"


def _perplexity_model() -> str:
    return _env("PERPLEXITY_MODEL", "perplexity/sonar") or "perplexity/sonar"


def _openai_request(
    system_prompt: str, user_prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    api_key = _api_key("openai")
    if not api_key:
        raise ValueError("openai api key is not configured")
    url = (
        _env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        + "/chat/completions"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": _openai_model(),
        "temperature": 0,
        "max_completion_tokens": 800,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "visibility_audit",
                "strict": True,
                "schema": _citation_json_schema(strict=True),
            },
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    return url, headers, body


def _anthropic_request(
    system_prompt: str, user_prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build a Claude Messages request.

    Haiku 5.5 rejects non-default sampling parameters, so temperature is
    omitted on purpose. Effort is pinned to low so adaptive thinking cannot
    consume the 15-second budget before the JSON answer is emitted.
    ``max_tokens`` still has to leave room for that thinking.
    """

    api_key = str(os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("anthropic api key is not configured")
    base = _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    url = f"{base}/v1/messages"
    headers = {
        "x-api-key": str(os.environ.get("ANTHROPIC_API_KEY")).strip(),
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "regular-workspaces-preview",
        "anthropic-workspace-id": str(os.environ.get("ANTHROPIC_WORKSPACE_ID")).strip(),
        "content-type": "application/json",
    }
    body: dict[str, Any] = {
        "model": _anthropic_model(),
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "output_config": {
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": _citation_json_schema(strict=True),
            },
        },
    }
    return url, headers, body


def _gemini_generation_config() -> dict[str, Any]:
    """Structured-output config for the configured Gemini model.

    Gemini 3.6 and later reject custom temperature values, and Gemini 3.8
    Flash thinks at medium effort unless told otherwise. ``thinkingLevel``
    of ``low`` keeps the classification inside the timeout. Gemini 2.x does
    not accept ``thinkingLevel``, so a zero thinking budget is sent instead
    when an operator points GEMINI_MODEL at that generation.
    """

    # responseFormat.text.mimeType rejects the MIME string "application/json"
    # (it expects the enum APPLICATION_JSON) and the call dies as a 400 before
    # any judgment is returned. responseMimeType is the field that accepts
    # the MIME string alongside responseSchema.
    config: dict[str, Any] = {
        "maxOutputTokens": 1024,
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "object",
            "properties": {
                "cited": {"type": "boolean"},
                "competitors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["cited", "competitors"],
        },
    }
    model = _gemini_model()
    if model.startswith("gemini-2"):
        config["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        level = _env("GEMINI_THINKING_LEVEL", "low") or "low"
        config["thinkingConfig"] = {"thinkingLevel": level}
    return config


def _gemini_request(
    system_prompt: str, user_prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build a Gemini generateContent request.

    The key is the ``key`` query parameter on the standard Generative Language
    host. Callers log the hostname only, never the full URL.
    """

    api_key = _api_key("gemini")
    if not api_key:
        raise ValueError("gemini api key is not configured")
    model = _gemini_model()
    base = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip(
        "/"
    )
    url = f"{base}/v1beta/models/{model}:generateContent?key={quote_plus(api_key)}"
    headers = {
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": _gemini_generation_config(),
    }
    return url, headers, body


def _perplexity_request(
    system_prompt: str, user_prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build a Perplexity Agent API request.

    Sonar chat completions were retired in September 2026. The Agent API
    takes the system prompt in ``instructions`` and the scenario in
    ``input``. Web search is forced because an ungrounded completion is not
    evidence that a domain is in the search index. Unknown fields are
    rejected by that API, so this body only contains documented keys.
    """

    api_key = _api_key("perplexity")
    if not api_key:
        raise ValueError("perplexity api key is not configured")
    base = _env("PERPLEXITY_BASE_URL", "https://api.perplexity.ai").rstrip("/")
    url = f"{base}/v1/agent"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": _perplexity_model(),
        "instructions": system_prompt,
        "input": user_prompt,
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "web_search"},
        "max_output_tokens": 800,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "visibility_audit",
                "strict": True,
                "schema": _citation_json_schema(strict=True),
            },
        },
    }
    return url, headers, body


def build_provider_request(
    provider: str, system_prompt: str, user_prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    if provider == "openai":
        return _openai_request(system_prompt, user_prompt)
    if provider == "anthropic":
        return _anthropic_request(system_prompt, user_prompt)
    if provider == "gemini":
        return _gemini_request(system_prompt, user_prompt)
    if provider == "perplexity":
        return _perplexity_request(system_prompt, user_prompt)
    raise ValueError(f"unknown provider {provider}")


# ---------------------------------------------------------------------------
# Response interpretation
# ---------------------------------------------------------------------------


def _failed_result(provider: str, failure_reason: str | None = None) -> dict[str, Any]:
    """Failure contract returned to the metrics engine.

    provider, status, and cited are always present. failure_reason is set
    only when the call failed for a classified cause such as unauthorized
    or schema_validation.
    """

    result: dict[str, Any] = {"provider": provider, "status": "failed", "cited": False}
    if failure_reason:
        result["failure_reason"] = failure_reason
    return result


_SECRET_TEXT = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]+|sk-[A-Za-z0-9_\-]+|pplx-[A-Za-z0-9_\-]+|"
    r"AIza[0-9A-Za-z_\-]+|AQ\.[A-Za-z0-9_\-]+|sb_secret_[A-Za-z0-9_\-]+)"
)


def _redact_provider_text(value: str) -> str:
    return _SECRET_TEXT.sub("[redacted]", value)[:300]


def _redact_api_key_middle(value: str | None) -> str:
    """Hide the middle 15 characters of an API key before it reaches a print log."""

    secret = value or ""
    if len(secret) <= 15:
        return "*" * len(secret)
    start = (len(secret) - 15) // 2
    hidden = secret[start : start + 15]
    return secret.replace(hidden, "*" * 15, 1)


def _provider_error_message(response: httpx.Response) -> str:
    """Pull a short provider error string. Credentials are stripped."""

    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None
    message = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("status") or "")
        elif isinstance(error, str):
            message = error
        elif isinstance(payload.get("message"), str):
            message = payload["message"]
    if not message:
        message = response.text[:300]
    return _redact_provider_text(" ".join(message.split()))


def _provider_error_class(status_code: int, message: str) -> str:
    """Classify Anthropic and Gemini failures the audit log needs to name."""

    lowered = message.lower()
    if status_code == 401 or "unauthorized" in lowered:
        return "unauthorized"
    if (
        status_code == 403
        or "permission_denied" in lowered
        or "denied access" in lowered
        or "forbidden" in lowered
    ):
        return "forbidden"
    if "anthropic-workspace-id" in lowered or "not scoped to a workspace" in lowered:
        return "workspace_required"
    if status_code == 400 or any(
        token in lowered
        for token in ("schema", "invalid", "mime", "validation", "json")
    ):
        return "schema_validation"
    if status_code == 429:
        return "rate_limited"
    return "http_error"


def _success_result(
    provider: str,
    cited: bool,
    competitors: list[str],
    evaluation_source: Literal["live", "local_auditor"],
) -> dict[str, Any]:
    return {
        "provider": provider,
        "status": "success",
        "cited": cited,
        "competitors": competitors,
        "evaluation_source": evaluation_source,
    }


def _filter_competitors(
    competitors: list[str], brand_name: str, client_domain: str
) -> list[str]:
    """Drop blanks, the target itself, and case-insensitive duplicates."""

    banned = {brand_name.lower(), client_domain.lower()}
    domain_label = client_domain.split(".", 1)[0].lower()
    if domain_label:
        banned.add(domain_label)
    unique: list[str] = []
    seen: set[str] = set()
    for name in competitors:
        cleaned = " ".join(name.split())
        if not cleaned:
            continue
        clipped = cleaned[:120]
        key = clipped.lower()
        if key in banned or key in seen:
            continue
        seen.add(key)
        unique.append(clipped)
        if len(unique) >= 10:
            break
    return unique


def _coerce_text(content: Any) -> str:
    """Pull assistant text out of a string or a list of content blocks."""

    if isinstance(content, str):
        text = content.strip()
        if not text:
            raise ValueError("provider text was empty")
        return text
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Gemini marks reasoning traces with thought=true. Those are
                # not the JSON judgment and must not be parsed as one.
                if item.get("thought") is True:
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "\n".join(part.strip() for part in parts if part and part.strip())
        if not joined:
            raise ValueError("provider text was empty")
        return joined
    raise ValueError("provider text had an unsupported shape")


def _extract_json_object(text: str) -> Any:
    """Parse a JSON object, including a fenced or prose-wrapped payload."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE
        )
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def _parse_citation_document(text: str) -> dict[str, Any]:
    """Validate the model's citation JSON. Ambiguous values are rejected."""

    document = _extract_json_object(text)
    if not isinstance(document, dict):
        raise TypeError("model payload was not a JSON object")
    if "cited" not in document:
        raise ValueError("model payload did not contain cited")
    cited = document["cited"]
    if isinstance(cited, str):
        lowered = cited.strip().lower()
        if lowered in {"true", "yes"}:
            cited = True
        elif lowered in {"false", "no"}:
            cited = False
        else:
            raise ValueError("cited string was not boolean-like")
    if not isinstance(cited, bool):
        raise TypeError("cited was not a boolean")
    raw_competitors = document.get("competitors", [])
    if raw_competitors is None:
        raw_competitors = []
    if not isinstance(raw_competitors, list):
        raise TypeError("competitors was not a list")
    competitors: list[str] = []
    for item in raw_competitors:
        if isinstance(item, str):
            competitors.append(item)
        elif isinstance(item, dict):
            for key in ("name", "brand", "domain"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    competitors.append(value)
                    break
    return {"cited": cited, "competitors": competitors}


def _chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat completion did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise TypeError("chat completion choice did not contain a message")
    return _coerce_text(message.get("content"))


def _anthropic_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return _coerce_text(content)
    if not isinstance(content, list):
        raise TypeError("anthropic response did not contain content")
    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        # Thinking blocks are not the answer. Only text blocks count.
        if block.get("type") not in {None, "text"}:
            continue
        text = block.get("text")
        if isinstance(text, str):
            texts.append(text)
    if not texts:
        raise ValueError("anthropic response did not contain a text block")
    return _coerce_text(texts)


def _gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("gemini response did not contain candidates")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise TypeError("gemini candidate was not an object")
    content = candidate.get("content")
    if not isinstance(content, dict):
        raise TypeError("gemini candidate did not contain content")
    return _coerce_text(content.get("parts"))


def _perplexity_agent_text(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    if status not in {None, "completed"}:
        raise ValueError("perplexity run did not complete")
    output = payload.get("output")
    if not isinstance(output, list):
        raise TypeError("perplexity response did not contain output")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if content is None:
            content = item.get("contents")
        texts.append(_coerce_text(content))
    if not texts:
        raise ValueError("perplexity response did not contain a message")
    return "\n".join(texts)


def _extract_provider_text(provider: str, payload: dict[str, Any]) -> str:
    if provider == "openai":
        return _chat_completion_text(payload)
    if provider == "anthropic":
        return _anthropic_text(payload)
    if provider == "gemini":
        return _gemini_text(payload)
    if provider == "perplexity":
        # Synchronous Sonar requests are still reformulated onto the Agent
        # API. Accept either the Agent output array or a legacy choices body.
        if isinstance(payload.get("output"), list):
            return _perplexity_agent_text(payload)
        if isinstance(payload.get("choices"), list):
            return _chat_completion_text(payload)
        raise ValueError("perplexity response did not contain output or choices")
    raise ValueError(f"unknown provider {provider}")


def _iter_citation_urls(provider: str, payload: dict[str, Any]) -> list[str]:
    """Collect grounded source URLs. Only Perplexity returns those today."""

    if provider != "perplexity" or not isinstance(payload, dict):
        return []
    urls: list[str] = []
    citations = payload.get("citations")
    if isinstance(citations, list):
        urls.extend(item for item in citations if isinstance(item, str))
    search_results = payload.get("search_results")
    if isinstance(search_results, list):
        for item in search_results:
            if isinstance(item, dict) and isinstance(item.get("url"), str):
                urls.append(item["url"])
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "search_results":
                continue
            results = item.get("results")
            if not isinstance(results, list):
                continue
            for result in results:
                if isinstance(result, dict) and isinstance(result.get("url"), str):
                    urls.append(result["url"])
    return urls


def _url_hosts_domain(url: str, domain: str) -> bool:
    """True when the URL host is the domain or a subdomain of it."""

    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    host = host.lower().removeprefix("www.")
    target = domain.lower().removeprefix("www.")
    return bool(host) and (host == target or host.endswith("." + target))


def _interpret_provider_payload(
    provider: str,
    payload: dict[str, Any],
    client_domain: str,
    brand_name: str,
) -> dict[str, Any]:
    """Turn a raw provider body into a success result.

    Perplexity can disagree with itself: the model JSON may say the brand
    was not cited while a retrieved URL is on the client domain. A grounded
    URL is stronger evidence than the model's summary, so the domain match
    promotes cited to true. It never demotes a true judgment, because a
    brand can be named without its canonical host appearing in the sources.
    """

    if not isinstance(payload, dict):
        raise TypeError("provider payload was not a JSON object")
    text = _extract_provider_text(provider, payload)
    parsed = _parse_citation_document(text)
    cited = bool(parsed["cited"])
    if any(
        _url_hosts_domain(url, client_domain)
        for url in _iter_citation_urls(provider, payload)
    ):
        cited = True
    competitors = _filter_competitors(parsed["competitors"], brand_name, client_domain)
    return _success_result(provider, cited, competitors, "live")


# ---------------------------------------------------------------------------
# Local auditor
# ---------------------------------------------------------------------------


def _local_visibility_decision(
    provider: str, client_domain: str, brand_name: str, keyword: str
) -> dict[str, Any]:
    """Deterministic stand-in used when a live provider call is not configured.

    A keyword that already contains the brand or the domain's first label is
    always cited. Otherwise a stable hash bucket, shifted per provider, decides.
    The same inputs always produce the same citation and the same competitors.
    """

    keyword_tokens = set(_TOKEN_RE.findall(keyword.lower()))
    brand_tokens = set(_TOKEN_RE.findall(brand_name.lower()))
    domain_tokens = set(_TOKEN_RE.findall(client_domain.split(".", 1)[0].lower()))
    direct_hit = bool(keyword_tokens & (brand_tokens | domain_tokens))

    digest = hashlib.sha256(
        f"{provider}|{client_domain}|{brand_name}|{keyword}".encode()
    ).hexdigest()
    bucket = int(digest[:8], 16) % 100
    provider_bias = {"openai": 0, "anthropic": 11, "gemini": 23, "perplexity": 5}.get(
        provider, 0
    )
    cited = direct_hit or ((bucket + provider_bias) % 100) < 40

    competitor_count = int(digest[8:10], 16) % 4
    start = int(digest[10:12], 16) % len(_LOCAL_COMPETITOR_CATALOG)
    competitors = [
        _LOCAL_COMPETITOR_CATALOG[(start + offset) % len(_LOCAL_COMPETITOR_CATALOG)]
        for offset in range(competitor_count)
    ]
    return _success_result(
        provider,
        cited,
        _filter_competitors(competitors, brand_name, client_domain),
        "local_auditor",
    )


# ---------------------------------------------------------------------------
# Outbound orchestration
# ---------------------------------------------------------------------------


async def fetch_model_visibility(
    client_domain: str,
    brand_name: str,
    keyword: str,
    provider: str,
) -> dict[str, Any]:
    """Ask one answer engine whether the brand is cited for the keyword.

    Network failures, timeouts, HTTP 429, other non-200 responses, and
    unusable payloads all return ``{"provider", "status": "failed", "cited": False}``.
    This function does not raise for provider errors. The audit loop depends
    on that guarantee: one provider must not cancel the other three.
    """

    if provider not in PROVIDERS:
        _log(logging.ERROR, "unknown provider requested", provider=provider)
        return _failed_result(provider)

    if _use_local_auditor(provider):
        decision = _local_visibility_decision(
            provider, client_domain, brand_name, keyword
        )
        _log(
            logging.INFO,
            "provider evaluated",
            provider=provider,
            keyword=keyword,
            status="success",
            cited=decision["cited"],
            evaluation_source="local_auditor",
        )
        return decision

    started = time.perf_counter()
    try:
        async with _semaphore_for_current_loop():
            system_prompt = build_system_prompt(client_domain, brand_name, keyword)
            user_prompt = build_user_prompt(client_domain, brand_name, keyword)
            url, headers, body = build_provider_request(
                provider, system_prompt, user_prompt
            )
            # Host only. Gemini used to take the key on the query string;
            # the URL is still not logged, in case a base URL contains one.
            _log(
                logging.INFO,
                "provider request started",
                provider=provider,
                keyword=keyword,
                host=urlparse(url).hostname,
            )
            if provider == "anthropic":
                # Built at the send site and passed straight through. The shared
                # client does not replace this mapping; request headers win.
                headers = {
                    "x-api-key": str(os.environ.get("ANTHROPIC_API_KEY")).strip(),
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "regular-workspaces-preview",
                    "anthropic-workspace-id": str(
                        os.environ.get("ANTHROPIC_WORKSPACE_ID")
                    ).strip(),
                    "content-type": "application/json",
                }
                safe_headers = {
                    key: _redact_api_key_middle(value) if key == "x-api-key" else value
                    for key, value in headers.items()
                }
                print(f"anthropic request headers: {safe_headers}", flush=True)
            response = await _require_http_client().post(
                url, headers=headers, json=body
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        if response.status_code == 401:
            error_message = _provider_error_message(response)
            _log(
                logging.WARNING,
                "provider unauthorized",
                provider=provider,
                keyword=keyword,
                status_code=401,
                error_class="unauthorized",
                error_message=error_message,
                elapsed_ms=elapsed_ms,
            )
            raise HTTPException(status_code=401, detail="Unauthorized")
        if response.status_code != 200:
            error_message = _provider_error_message(response)
            error_class = _provider_error_class(response.status_code, error_message)
            # 403 and schema 400s are logged by class so Anthropic and Gemini
            # failures are not indistinguishable from a timeout.
            _log(
                logging.WARNING,
                "provider http error",
                provider=provider,
                keyword=keyword,
                status_code=response.status_code,
                error_class=error_class,
                error_message=error_message,
                elapsed_ms=elapsed_ms,
            )
            return _failed_result(provider, error_class)

        result = _interpret_provider_payload(
            provider,
            response.json(),
            client_domain,
            brand_name,
        )
        _log(
            logging.INFO,
            "provider evaluated",
            provider=provider,
            keyword=keyword,
            status="success",
            cited=result["cited"],
            evaluation_source="live",
            elapsed_ms=elapsed_ms,
        )
        return result
    except httpx.TimeoutException:
        _log(
            logging.WARNING,
            "provider timed out",
            provider=provider,
            keyword=keyword,
            error_type="timeout",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return _failed_result(provider)
    except httpx.HTTPError:
        # Transport errors. The exception text is intentionally discarded:
        # httpx includes the request URL, which is the wrong place for a key
        # if an operator points a base URL at a signed endpoint.
        _log(
            logging.WARNING,
            "provider transport error",
            provider=provider,
            keyword=keyword,
            error_type="transport",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return _failed_result(provider)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        # A 200 body that is not the citation schema is a validation failure.
        # A missing key is raised before the request and must not be labeled
        # as a schema error.
        message = _redact_provider_text(str(exc))
        if "not configured" in message or "api key" in message:
            error_class = "missing_api_key"
        else:
            error_class = "schema_validation"
        _log(
            logging.WARNING,
            "provider payload rejected",
            provider=provider,
            keyword=keyword,
            error_class=error_class,
            error_type=exc.__class__.__name__,
            error_message=message,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return _failed_result(provider, error_class)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - one provider must not cancel the others
        _log(
            logging.ERROR,
            "provider call failed",
            provider=provider,
            keyword=keyword,
            error_type="unexpected",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return _failed_result(provider)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _percentage(numerator: int, denominator: int) -> float:
    """Return a percentage rounded half-up to two decimal places.

    A zero denominator is a defined 0.0, not an error. That happens when
    every provider failed; there is no successful response to share across.
    """

    if denominator <= 0 or numerator <= 0:
        return 0.0
    value = (Decimal(numerator) * Decimal(100)) / Decimal(denominator)
    quantized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(quantized)


def _coerce_provider_result(
    payload: dict[str, Any], fallback_provider: str
) -> ProviderResult:
    provider = payload.get("provider")
    if not isinstance(provider, str) or not provider:
        provider = fallback_provider
    status = payload.get("status")
    cited = payload.get("cited")
    if status not in {"success", "failed"} or not isinstance(cited, bool):
        status = "failed"
        cited = False
    if status == "failed":
        cited = False
    raw_competitors = payload.get("competitors", [])
    competitors: list[str] = []
    if status == "success" and isinstance(raw_competitors, list):
        competitors = [item for item in raw_competitors if isinstance(item, str)]
    source = payload.get("evaluation_source")
    if source not in {"live", "local_auditor"}:
        source = None
    failure_reason = payload.get("failure_reason")
    if status != "failed" or not isinstance(failure_reason, str) or not failure_reason:
        failure_reason = None
    return ProviderResult(
        provider=provider,
        status=status,
        cited=cited,
        competitors=competitors,
        evaluation_source=source,
        failure_reason=failure_reason,
    )


def _optimization_payload(providers: list[ProviderResult]) -> str:
    missing = [
        PROVIDER_DISPLAY_NAMES.get(item.provider, item.provider)
        for item in providers
        if not item.cited
    ]
    if not missing:
        missing = [PROVIDER_DISPLAY_NAMES[name] for name in PROVIDERS]
    return OPTIMIZATION_TEMPLATE.format(providers=", ".join(missing))


def _failed_keyword_metrics() -> KeywordMetrics:
    """Metrics used when a keyword task itself collapses.

    Every provider is recorded as failed, the score is 0, and the
    optimization sentence names the full provider set.
    """

    providers = [
        ProviderResult(provider=name, status="failed", cited=False)
        for name in PROVIDERS
    ]
    return KeywordMetrics(
        vibe_share_score=0.0,
        successful_citations=0,
        total_successful_provider_responses=0,
        providers=providers,
        competitors=[],
        optimization_payload=_optimization_payload(providers),
    )


def calculate_keyword_metrics(provider_results: list[dict[str, Any]]) -> KeywordMetrics:
    """Reduce a provider response array into the Vibe-Share Score.

    Score = (successful citations / successful provider responses) * 100.
    A provider that times out, returns 401/403, or fails schema validation
    is omitted from that ratio. Two live answers and two failures therefore
    score the two answers, not 0/4. The score is the failure sentinel 0 only
    when every provider failed. Zero citations among the engines that did
    answer is a measured 0, and their competitor lists are still kept.
    The optimization sentence is attached only when the score drops below 50.
    A score of exactly 50 does not trigger it.
    """

    by_name: dict[str, dict[str, Any]] = {}
    unnamed: list[dict[str, Any]] = []
    for item in provider_results:
        if not isinstance(item, dict):
            continue
        provider_name = item.get("provider")
        if isinstance(provider_name, str) and provider_name:
            by_name[provider_name] = item
        else:
            unnamed.append(item)

    ordered: list[ProviderResult] = []
    for provider in PROVIDERS:
        payload = by_name.get(provider)
        if payload is None and unnamed:
            payload = unnamed.pop(0)
        if payload is None:
            payload = _failed_result(provider)
        ordered.append(_coerce_provider_result(payload, provider))

    # Failed calls remain in `ordered` for the provider log. The ratio uses
    # only responses whose status is success. 0 citations / 2 successes = 0.0.
    successful = [item for item in ordered if item.status == "success"]
    citations = [item for item in successful if item.cited]
    if not successful:
        score = 0.0
    else:
        score = _percentage(len(citations), len(successful))

    competitors: list[str] = []
    seen: set[str] = set()
    for item in successful:
        for name in item.competitors:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            competitors.append(name)

    optimization: str | None = None
    if score < 50:
        optimization = _optimization_payload(ordered)

    return KeywordMetrics(
        vibe_share_score=score,
        successful_citations=len(citations),
        total_successful_provider_responses=len(successful),
        providers=ordered,
        competitors=competitors,
        optimization_payload=optimization,
    )


async def _audit_single_keyword(
    client_domain: str, brand_name: str, keyword: str
) -> KeywordMetrics:
    """Fan out one keyword across all four providers.

    ``return_exceptions`` keeps a programming error inside one provider
    task from cancelling the siblings. ``fetch_model_visibility`` already
    catches provider failures; this is the second guard.
    """

    raw_results = await asyncio.gather(
        *(
            fetch_model_visibility(client_domain, brand_name, keyword, provider)
            for provider in PROVIDERS
        ),
        return_exceptions=True,
    )
    normalized: list[dict[str, Any]] = []
    for provider, raw in zip(PROVIDERS, raw_results, strict=True):
        if isinstance(raw, HTTPException):
            raise raw
        if isinstance(raw, Exception):
            _log(
                logging.ERROR,
                "provider task raised",
                provider=provider,
                keyword=keyword,
                error_type=type(raw).__name__,
            )
            normalized.append(_failed_result(provider))
            continue
        if not isinstance(raw, dict):
            normalized.append(_failed_result(provider))
            continue
        normalized.append(raw)
    return calculate_keyword_metrics(normalized)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open the shared client for the process and close it on shutdown."""

    global _http_client
    owns_client = _http_client is None
    if owns_client:
        _http_client = _build_http_client()
    configured = {name: bool(_api_key(name)) for name in PROVIDERS}
    _log(
        logging.INFO,
        "visibility api started",
        version=VERSION,
        provider_mode=_provider_mode(),
        providers_configured=configured,
    )
    await start_billing()
    try:
        yield
    finally:
        await stop_billing()
        if owns_client and _http_client is not None:
            await _http_client.aclose()
            _http_client = None
        _log(logging.INFO, "visibility api stopped")


class NewlineJSONResponse(JSONResponse):
    """JSON bodies end with a newline so a curl result does not sit on the shell prompt."""

    def render(self, content: Any) -> bytes:
        rendered = super().render(content)
        if rendered.endswith(b"\n"):
            return rendered
        return rendered + b"\n"


app = FastAPI(
    title="AI Visibility Indexing API",
    summary="Measures how often AI answer engines cite a brand for a keyword.",
    version=VERSION,
    lifespan=_lifespan,
    default_response_class=NewlineJSONResponse,
)
app.include_router(billing_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    _request: Request, exc: HTTPException
) -> NewlineJSONResponse:
    """Keep error JSON on the same newline-terminated response as audit payloads."""

    return NewlineJSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.get("/", response_model=ServiceInfo, tags=["meta"])
async def service_info() -> ServiceInfo:
    return ServiceInfo(
        service="AI Visibility Indexing API",
        version=VERSION,
        audit="/v1/audit",
        health="/health",
        docs="/docs",
        webhook="/v1/webhooks/stripe",
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider_mode=_provider_mode(),
        providers_configured={name: bool(_api_key(name)) for name in PROVIDERS},
        http_client_ready=_http_client is not None,
        billing=billing_status(),
    )


@app.post("/v1/audit", response_model=AuditResponse, tags=["audit"])
async def create_audit(
    request: AuditRequest, background_tasks: BackgroundTasks
) -> AuditResponse:
    """Audit every keyword across OpenAI, Anthropic, Gemini, and Perplexity.

    The loop schedules one task per keyword. Each task gathers the four
    provider calls. Tasks run together, and a failure inside one keyword
    is stored as a zero score instead of failing the HTTP request.
    """

    started = time.perf_counter()
    _log(
        logging.INFO,
        "audit started",
        domain=request.client_domain,
        brand=request.brand_name,
        keywords=len(request.keywords),
        provider_mode=_provider_mode(),
    )
    tasks: list[asyncio.Task[KeywordMetrics]] = []
    for index, keyword in enumerate(request.keywords):
        tasks.append(
            asyncio.create_task(
                _audit_single_keyword(
                    request.client_domain, request.brand_name, keyword
                ),
                name=f"audit-keyword-{index}",
            )
        )
    try:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    results: dict[str, KeywordMetrics] = {}
    for keyword, outcome in zip(request.keywords, completed, strict=True):
        if isinstance(outcome, HTTPException):
            raise outcome
        if isinstance(outcome, KeywordMetrics):
            results[keyword] = outcome
            continue
        _log(
            logging.ERROR,
            "keyword audit failed",
            keyword=keyword,
            error_type=type(outcome).__name__,
        )
        results[keyword] = _failed_keyword_metrics()

    response = AuditResponse(
        timestamp=datetime.now(UTC).isoformat(),
        keywords_processed=len(results),
        results=results,
    )
    _log(
        logging.INFO,
        "audit completed",
        domain=request.client_domain,
        keywords_processed=response.keywords_processed,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        scores={
            keyword: metrics.vibe_share_score
            for keyword, metrics in response.results.items()
        },
    )
    # Billing runs after the response is sent. A Stripe or Supabase failure
    # is recorded by the billing registry and does not change this payload.
    background_tasks.add_task(
        bill_completed_audit,
        request.client_domain,
        response.keywords_processed,
    )
    return response


def _port() -> int:
    raw = _env("PORT", "8741") or "8741"
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"PORT must be an integer, received {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise SystemExit(f"PORT must be between 1 and 65535, received {port}")
    return port


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=_env("HOST", "0.0.0.0") or "0.0.0.0",
        port=_port(),
        reload=_flag("RELOAD", "false"),
    )
