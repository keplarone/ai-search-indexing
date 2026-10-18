"""Stripe metered billing for the IndexGrid visibility pipeline.

A completed audit reports its keyword count as a legacy Stripe usage record
on the account's subscription item, then returns. The report runs as a
FastAPI background task so a Stripe outage cannot fail the audit. Failures
are written to a local SQLite registry before the process forgets them.

stripe-python 16 removed ``stripe.UsageRecord``. Stripe's supported way to
keep calling ``POST /v1/subscription_items/{id}/usage_records`` on that SDK
is ``StripeClient.raw_request`` pinned to API version ``2025-02-24.acacia``,
the last version that accepts a meter-less usage record. This module uses a
``StripeClient`` instance for that call. It does not assign ``stripe.api_key``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import stripe
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# Last Stripe API version that still accepts legacy usage records.
USAGE_RECORD_API_VERSION = "2025-02-24.acacia"
# Version used for every other call this process makes.
STRIPE_API_VERSION = "2026-09-30.endive"

HANDLED_SUBSCRIPTION_EVENTS = frozenset(
    {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }
)
# Statuses that must not keep receiving usage increments.
_CLOSED_SUBSCRIPTION_STATUSES = frozenset({"canceled", "incomplete_expired"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9_]+$")
_SECRET_RE = re.compile(r"\b(?:sk|rk|whsec)_(?:live|test)_[A-Za-z0-9]+")

LOGGER = logging.getLogger("visibility_index.billing")

router = APIRouter(tags=["billing"])

_registry_lock = threading.Lock()
_stripe_client: stripe.StripeClient | None = None
_stripe_client_key: str | None = None
_supabase_client: httpx.AsyncClient | None = None
_supabase_client_owned = False


class BillingRegistryError(Exception):
    """The local failure registry could not be written."""


class SupabaseError(Exception):
    """Supabase rejected or could not complete an accounts update."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def load_local_env(path: Path | None = None) -> bool:
    """Load the workspace ``.env`` into the process environment.

    Existing variables win, so a secret injected by the platform is never
    replaced by the file. Values are not logged.
    """

    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


# Apply the workspace file before any credential lookup.
load_local_env()


def _stripe_api_key() -> str:
    """Stripe secret from the local environment.

    ``STRIPE_API_KEY`` is the canonical name. ``STRIPE_SECRET_KEY`` is the
    name already used in the workspace ``.env``.
    """

    return _env("STRIPE_API_KEY") or _env("STRIPE_SECRET_KEY")


def _redact(value: str) -> str:
    """Remove credential-shaped strings before they reach logs or SQLite."""

    return _SECRET_RE.sub("[redacted]", value)[:500]


def _registry_path() -> Path:
    return Path(_env("BILLING_REGISTRY_PATH", "data/billing_registry.sqlite"))


def _max_usage_count() -> int:
    raw = _env("MAX_BILLABLE_KEYWORDS", "100000")
    try:
        value = int(raw)
    except ValueError:
        return 100_000
    return min(max(value, 1), 1_000_000)


def billing_status() -> dict[str, bool]:
    """Report which billing credentials are present. Values are booleans only."""

    return {
        "stripe_configured": bool(_stripe_api_key()),
        "webhook_configured": bool(_env("STRIPE_WEBHOOK_SECRET")),
        "supabase_configured": bool(_env("SUPABASE_URL") and _supabase_key()),
    }


def _supabase_key() -> str:
    return _env("SUPABASE_SECRET_KEY") or _env("SUPABASE_SERVICE_ROLE_KEY")


def _supabase_configured() -> bool:
    return bool(_env("SUPABASE_URL") and _supabase_key())


# ---------------------------------------------------------------------------
# Failure registry
# ---------------------------------------------------------------------------


def _connect_registry() -> sqlite3.Connection:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            operation TEXT NOT NULL,
            account_id TEXT,
            stripe_customer_id TEXT,
            stripe_subscription_item_id TEXT,
            usage_count INTEGER,
            usage_unix_timestamp INTEGER,
            idempotency_key TEXT,
            error_type TEXT,
            error_message TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )
    return connection


def _insert_event(
    *,
    status: str,
    operation: str,
    payload: dict[str, Any],
    account_id: str | None = None,
    stripe_customer_id: str | None = None,
    stripe_subscription_item_id: str | None = None,
    usage_count: int | None = None,
    usage_unix_timestamp: int | None = None,
    idempotency_key: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> int:
    """Append one billing event and return its row id.

    The write is synchronous and locked. Callers on the event loop should
    reach it through ``asyncio.to_thread``.
    """

    now = datetime.now(UTC).isoformat()
    with _registry_lock:
        connection = _connect_registry()
        try:
            cursor = connection.execute(
                """
                INSERT INTO billing_events (
                    recorded_at, updated_at, status, operation, account_id,
                    stripe_customer_id, stripe_subscription_item_id, usage_count,
                    usage_unix_timestamp, idempotency_key, error_type,
                    error_message, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    now,
                    status,
                    operation,
                    account_id,
                    stripe_customer_id,
                    stripe_subscription_item_id,
                    usage_count,
                    usage_unix_timestamp,
                    idempotency_key,
                    error_type,
                    _redact(error_message) if error_message else None,
                    json.dumps(payload, default=str, sort_keys=True),
                ),
            )
            connection.commit()
            if cursor.lastrowid is None:
                raise BillingRegistryError("billing registry did not return a row id")
            return int(cursor.lastrowid)
        finally:
            connection.close()


def _update_event(
    event_id: int,
    *,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    with _registry_lock:
        connection = _connect_registry()
        try:
            connection.execute(
                """
                UPDATE billing_events
                SET status = ?, updated_at = ?, error_type = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    status,
                    now,
                    error_type,
                    _redact(error_message) if error_message else None,
                    event_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()


def _load_replayable_usage(limit: int) -> list[dict[str, Any]]:
    with _registry_lock:
        connection = _connect_registry()
        try:
            rows = connection.execute(
                """
                SELECT id, stripe_subscription_item_id, usage_count,
                       usage_unix_timestamp, idempotency_key
                FROM billing_events
                WHERE operation = 'usage_report'
                  AND status = 'failed'
                  AND idempotency_key IS NOT NULL
                  AND stripe_subscription_item_id IS NOT NULL
                  AND usage_count IS NOT NULL
                  AND usage_unix_timestamp IS NOT NULL
                ORDER BY id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            connection.close()
    return [dict(row) for row in rows]


def record_billing_failure(
    *,
    operation: str,
    error: BaseException,
    payload: dict[str, Any],
    account_id: str | None = None,
    stripe_customer_id: str | None = None,
    stripe_subscription_item_id: str | None = None,
    usage_count: int | None = None,
    usage_unix_timestamp: int | None = None,
    idempotency_key: str | None = None,
) -> int:
    """Persist a billing failure. This is the registry the audit loop must not skip."""

    message = _redact(str(error) or error.__class__.__name__)
    LOGGER.warning(
        "billing failure recorded",
        extra={
            "event_fields": {
                "operation": operation,
                "error_type": error.__class__.__name__,
                "account_id": account_id,
                "stripe_subscription_item_id": stripe_subscription_item_id,
                "usage_count": usage_count,
                "idempotency_key": idempotency_key,
            }
        },
    )
    return _insert_event(
        status="failed",
        operation=operation,
        payload=payload,
        account_id=account_id,
        stripe_customer_id=stripe_customer_id,
        stripe_subscription_item_id=stripe_subscription_item_id,
        usage_count=usage_count,
        usage_unix_timestamp=usage_unix_timestamp,
        idempotency_key=idempotency_key,
        error_type=error.__class__.__name__,
        error_message=message,
    )


# ---------------------------------------------------------------------------
# Stripe usage records
# ---------------------------------------------------------------------------


def get_stripe_client() -> stripe.StripeClient:
    """Return the process Stripe client.

    The key is read from ``STRIPE_API_KEY`` or, when that is unset, from
    ``STRIPE_SECRET_KEY`` in the local environment. It is never logged.
    Prefer a restricted key (``rk_``) with permission to create usage records.
    """

    global _stripe_client, _stripe_client_key
    api_key = _stripe_api_key()
    if not api_key:
        raise RuntimeError("STRIPE_API_KEY is not configured")
    if _stripe_client is None or _stripe_client_key != api_key:
        _stripe_client = stripe.StripeClient(
            api_key,
            stripe_version=STRIPE_API_VERSION,
            max_network_retries=1,
        )
        _stripe_client_key = api_key
    return _stripe_client


def _usage_record_create(
    client: stripe.StripeClient,
    subscription_item_id: str,
    usage_count: int,
    usage_unix_timestamp: int,
    idempotency_key: str,
) -> Any:
    """Create one legacy usage record.

    ``action=increment`` adds this audit's keyword count to the billing
    period. ``set`` would replace the period total with this request alone
    and drop every earlier audit. The idempotency key is passed on the
    request so Stripe and the SDK's single network retry cannot bill it twice.
    """

    return client.raw_request(
        "post",
        f"/v1/subscription_items/{subscription_item_id}/usage_records",
        stripe_version=USAGE_RECORD_API_VERSION,
        idempotency_key=idempotency_key,
        quantity=usage_count,
        timestamp=usage_unix_timestamp,
        action="increment",
    )


def _idempotency_key(account_id: str, executed_at: datetime) -> str:
    """Build a key from the billing account id and this execution's timestamp.

    The subscription item id is the account id for metered reporting. The
    timestamp includes microseconds so two audits of the same account in the
    same second stay distinct, while a retry of this execution reuses the key.
    """

    stamp = executed_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"idxgrid_usage_{account_id}_{stamp}"


def _validate_usage(stripe_subscription_item_id: str, usage_count: int) -> str:
    account_id = stripe_subscription_item_id.strip()
    if not account_id or not _SAFE_ID.fullmatch(account_id):
        raise ValueError(
            "stripe_subscription_item_id is empty or contains unsafe characters"
        )
    if isinstance(usage_count, bool) or not isinstance(usage_count, int):
        raise TypeError("usage_count must be an integer")
    if usage_count < 0:
        raise ValueError("usage_count cannot be negative")
    if usage_count > _max_usage_count():
        raise ValueError("usage_count exceeds MAX_BILLABLE_KEYWORDS")
    return account_id


async def report_api_usage_to_stripe(
    stripe_subscription_item_id: str,
    usage_count: int,
) -> None:
    """Report keyword usage for one subscription item.

    Designed to run as a FastAPI ``BackgroundTasks`` callback after a
    successful audit. Stripe errors and connection timeouts are stored in
    the billing registry and are not raised, so the worker stays up.
    """

    executed_at = datetime.now(UTC)
    usage_unix_timestamp = int(executed_at.timestamp())
    try:
        account_id = _validate_usage(stripe_subscription_item_id, usage_count)
    except (ValueError, TypeError) as exc:
        await asyncio.to_thread(
            record_billing_failure,
            operation="usage_report",
            error=exc,
            payload={
                "stripe_subscription_item_id": stripe_subscription_item_id,
                "usage_count": usage_count,
            },
            stripe_subscription_item_id=stripe_subscription_item_id,
            usage_count=usage_count if isinstance(usage_count, int) else None,
        )
        return

    if usage_count == 0:
        LOGGER.info(
            "billing usage skipped",
            extra={
                "event_fields": {
                    "account_id": account_id,
                    "usage_count": 0,
                }
            },
        )
        return

    idempotency_key = _idempotency_key(account_id, executed_at)
    payload = {
        "subscription_item": account_id,
        "quantity": usage_count,
        "timestamp": usage_unix_timestamp,
        "action": "increment",
        "idempotency_key": idempotency_key,
        "stripe_version": USAGE_RECORD_API_VERSION,
    }
    event_id: int | None = None
    try:
        event_id = await asyncio.to_thread(
            _insert_event,
            status="pending",
            operation="usage_report",
            payload=payload,
            account_id=account_id,
            stripe_subscription_item_id=account_id,
            usage_count=usage_count,
            usage_unix_timestamp=usage_unix_timestamp,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:  # noqa: BLE001 - disk failure must not drop the usage report
        # The Stripe call still happens. The log line carries the key so the
        # usage can be reconciled if the registry disk is unavailable.
        LOGGER.error(
            "billing registry unavailable before usage report",
            extra={
                "event_fields": {
                    "error_type": exc.__class__.__name__,
                    "account_id": account_id,
                    "usage_count": usage_count,
                    "idempotency_key": idempotency_key,
                }
            },
        )

    try:
        client = get_stripe_client()
        # The Stripe SDK call is synchronous. The executor keeps the event
        # loop free for other audits while this request waits on the network.
        result = await asyncio.to_thread(
            _usage_record_create,
            client,
            account_id,
            usage_count,
            usage_unix_timestamp,
            idempotency_key,
        )
    except stripe.error.StripeError as exc:
        await _mark_usage_failed(
            event_id,
            exc,
            payload,
            account_id,
            usage_count,
            usage_unix_timestamp,
            idempotency_key,
        )
        return
    except (TimeoutError, ConnectionError, OSError) as exc:
        await _mark_usage_failed(
            event_id,
            exc,
            payload,
            account_id,
            usage_count,
            usage_unix_timestamp,
            idempotency_key,
        )
        return
    except Exception as exc:  # noqa: BLE001 - billing must not crash the process
        await _mark_usage_failed(
            event_id,
            exc,
            payload,
            account_id,
            usage_count,
            usage_unix_timestamp,
            idempotency_key,
        )
        return

    try:
        usage_record_id = _as_dict(result).get("id")
    except TypeError:
        usage_record_id = None
    LOGGER.info(
        "billing usage reported",
        extra={
            "event_fields": {
                "account_id": account_id,
                "usage_count": usage_count,
                "idempotency_key": idempotency_key,
                "usage_record_id": usage_record_id,
            }
        },
    )
    if event_id is not None:
        await asyncio.to_thread(_update_event, event_id, status="succeeded")


async def _mark_usage_failed(
    event_id: int | None,
    exc: BaseException,
    payload: dict[str, Any],
    account_id: str,
    usage_count: int,
    usage_unix_timestamp: int,
    idempotency_key: str,
) -> None:
    if event_id is not None:
        await asyncio.to_thread(
            _update_event,
            event_id,
            status="failed",
            error_type=exc.__class__.__name__,
            error_message=_redact(str(exc) or exc.__class__.__name__),
        )
        LOGGER.warning(
            "billing usage report failed",
            extra={
                "event_fields": {
                    "registry_id": event_id,
                    "error_type": exc.__class__.__name__,
                    "account_id": account_id,
                    "usage_count": usage_count,
                    "idempotency_key": idempotency_key,
                }
            },
        )
        return
    await asyncio.to_thread(
        record_billing_failure,
        operation="usage_report",
        error=exc,
        payload=payload,
        account_id=account_id,
        stripe_subscription_item_id=account_id,
        usage_count=usage_count,
        usage_unix_timestamp=usage_unix_timestamp,
        idempotency_key=idempotency_key,
    )


async def replay_failed_usage_reports(limit: int = 50) -> int:
    """Resubmit failed usage reports with their original idempotency keys.

    Reusing the key is what makes replay safe: if Stripe already accepted
    the record and the response was lost, the replay returns that record
    instead of incrementing the period again.
    """

    rows = await asyncio.to_thread(_load_replayable_usage, max(1, min(limit, 500)))
    replayed = 0
    for row in rows:
        try:
            client = get_stripe_client()
            await asyncio.to_thread(
                _usage_record_create,
                client,
                str(row["stripe_subscription_item_id"]),
                int(row["usage_count"]),
                int(row["usage_unix_timestamp"]),
                str(row["idempotency_key"]),
            )
        except (
            stripe.error.StripeError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            await asyncio.to_thread(
                _update_event,
                int(row["id"]),
                status="failed",
                error_type=exc.__class__.__name__,
                error_message=_redact(str(exc) or exc.__class__.__name__),
            )
            continue
        await asyncio.to_thread(_update_event, int(row["id"]), status="succeeded")
        replayed += 1
    return replayed


async def record_unassigned_usage(
    client_domain: str, usage_count: int, reason: str
) -> None:
    """Keep keyword counts that could not be matched to a subscription item."""

    await asyncio.to_thread(
        record_billing_failure,
        operation="usage_unassigned",
        error=RuntimeError(reason),
        payload={
            "client_domain": client_domain,
            "usage_count": usage_count,
            "reason": reason,
        },
        usage_count=usage_count,
    )


async def bill_completed_audit(client_domain: str, usage_count: int) -> None:
    """Background entry point for a finished audit.

    The subscription item is resolved here, after the HTTP response, so a
    slow accounts lookup does not delay the audit. The actual Stripe write
    is ``report_api_usage_to_stripe``.
    """

    try:
        item_id = await resolve_subscription_item_id(client_domain)
    except Exception as exc:  # noqa: BLE001 - lookup failure must not be lost
        LOGGER.warning(
            "billing account lookup failed",
            extra={
                "event_fields": {
                    "client_domain": client_domain,
                    "error_type": exc.__class__.__name__,
                }
            },
        )
        await record_unassigned_usage(
            client_domain, usage_count, "account lookup failed"
        )
        return
    if not item_id:
        await record_unassigned_usage(
            client_domain,
            usage_count,
            "no billable subscription item for this domain",
        )
        return
    await report_api_usage_to_stripe(item_id, usage_count)


# ---------------------------------------------------------------------------
# Supabase accounts
# ---------------------------------------------------------------------------


async def start_billing() -> None:
    """Open the Supabase HTTP client. Safe to call when one was injected."""

    global _supabase_client, _supabase_client_owned
    if _supabase_client is None:
        _supabase_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            trust_env=False,
            headers={"User-Agent": "IndexGrid-Billing/1.0"},
        )
        _supabase_client_owned = True
    LOGGER.info("billing started", extra={"event_fields": billing_status()})


async def stop_billing() -> None:
    """Close the Supabase client this process created."""

    global _supabase_client, _supabase_client_owned
    if _supabase_client_owned and _supabase_client is not None:
        await _supabase_client.aclose()
        _supabase_client = None
        _supabase_client_owned = False


def _require_supabase_client() -> httpx.AsyncClient:
    if _supabase_client is None:
        raise SupabaseError("Supabase client is not initialized")
    return _supabase_client


def _supabase_headers() -> dict[str, str]:
    key = _supabase_key()
    if not key:
        raise SupabaseError("Supabase service role key is not configured")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _supabase_url(table: str) -> str:
    base = _env("SUPABASE_URL").rstrip("/")
    if not base:
        raise SupabaseError("SUPABASE_URL is not configured")
    return f"{base}/rest/v1/{table}"


async def _supabase_select(filters: dict[str, str]) -> list[dict[str, Any]]:
    response = await _require_supabase_client().get(
        _supabase_url("accounts"),
        params=filters,
        headers=_supabase_headers(),
    )
    return _supabase_rows(response)


async def _supabase_patch(
    filters: dict[str, str], body: dict[str, Any]
) -> list[dict[str, Any]]:
    response = await _require_supabase_client().patch(
        _supabase_url("accounts"),
        params=filters,
        headers=_supabase_headers(),
        json=body,
    )
    return _supabase_rows(response)


def _supabase_rows(response: httpx.Response) -> list[dict[str, Any]]:
    if response.status_code >= 400:
        raise SupabaseError(
            f"Supabase accounts request failed with status {response.status_code}",
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise SupabaseError("Supabase accounts response was not JSON") from exc
    if not isinstance(payload, list):
        raise SupabaseError("Supabase accounts response was not a row list")
    return [row for row in payload if isinstance(row, dict)]


async def resolve_subscription_item_id(client_domain: str) -> str | None:
    """Return the metered subscription item for a domain, or None.

    Closed subscriptions are ignored so a canceled account does not keep
    accumulating usage after the webhook has marked it canceled.
    """

    if not _supabase_configured():
        return None
    rows = await _supabase_select(
        {
            "client_domain": f"eq.{client_domain}",
            "select": "id,stripe_subscription_item_id,subscription_status",
            "limit": "1",
        }
    )
    if not rows:
        return None
    status = rows[0].get("subscription_status")
    if isinstance(status, str) and status in _CLOSED_SUBSCRIPTION_STATUSES:
        return None
    item_id = rows[0].get("stripe_subscription_item_id")
    if isinstance(item_id, str) and item_id:
        return item_id
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert a Stripe resource to a plain dict.

    stripe-python 16 resources are not mappings and do not implement ``get``.
    ``to_dict`` is a real method, so it is read from the class.
    """

    if isinstance(value, dict):
        return value
    converter = getattr(type(value), "to_dict", None)
    if callable(converter):
        converted = value.to_dict()
        if isinstance(converted, dict):
            return converted
    raise TypeError(f"{type(value).__name__} is not a mapping")


def _customer_id(subscription: dict[str, Any]) -> str:
    customer = subscription.get("customer")
    if isinstance(customer, str) and _SAFE_ID.fullmatch(customer):
        return customer
    if isinstance(customer, dict):
        nested = customer.get("id")
        if isinstance(nested, str) and _SAFE_ID.fullmatch(nested):
            return nested
    raise ValueError("subscription is missing a Stripe customer id")


def _subscription_items(subscription: dict[str, Any]) -> list[dict[str, Any]]:
    items = subscription.get("items")
    data = items.get("data") if isinstance(items, dict) else None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    raise ValueError("subscription is missing items")


def _is_metered_item(item: dict[str, Any]) -> bool:
    price = item.get("price")
    if not isinstance(price, dict):
        price = item.get("plan")
    if not isinstance(price, dict):
        return False
    recurring = price.get("recurring")
    if not isinstance(recurring, dict):
        return False
    return bool(recurring.get("meter")) or recurring.get("usage_type") == "metered"


def select_subscription_item_id(subscription: dict[str, Any]) -> str:
    """Choose the subscription item that usage records must be attached to.

    A licensed base fee and a metered keyword price often share one
    subscription. Only the metered item accepts usage records. A single-item
    subscription is used as-is when the price was not expanded.
    """

    items = _subscription_items(subscription)
    metered = [item for item in items if _is_metered_item(item)]
    if metered:
        chosen = metered[0]
    elif len(items) == 1:
        chosen = items[0]
    else:
        raise ValueError("subscription has multiple items and none is metered")
    item_id = chosen.get("id")
    if not isinstance(item_id, str) or not _SAFE_ID.fullmatch(item_id):
        raise ValueError("subscription item id is missing")
    return item_id


async def update_account_from_subscription(subscription: Any) -> dict[str, Any]:
    """Write the customer id and metered subscription item onto ``accounts``.

    The row is matched on ``stripe_customer_id``. When checkout stored the
    domain on the subscription and the customer id is not linked yet, a
    second update matches ``client_domain``. Zero matching rows is an error
    so the webhook returns non-200 and Stripe retries.
    """

    record = _as_dict(subscription)
    customer_id = _customer_id(record)
    subscription_id = record.get("id")
    if not isinstance(subscription_id, str) or not _SAFE_ID.fullmatch(subscription_id):
        raise ValueError("subscription id is missing")
    item_id = select_subscription_item_id(record)
    status = record.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("subscription status is missing")
    body = {
        "stripe_customer_id": customer_id,
        "stripe_subscription_id": subscription_id,
        "stripe_subscription_item_id": item_id,
        "subscription_status": status,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    rows = await _supabase_patch({"stripe_customer_id": f"eq.{customer_id}"}, body)
    if rows:
        return rows[0]
    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    )
    domain = metadata.get("client_domain")
    if isinstance(domain, str) and domain:
        rows = await _supabase_patch(
            {
                "client_domain": f"eq.{domain}",
                "stripe_customer_id": "is.null",
            },
            body,
        )
        if rows:
            return rows[0]
    raise SupabaseError(f"no accounts row for Stripe customer {customer_id}")


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


def _client_ip(request: Request) -> str:
    if _env("STRIPE_TRUST_PROXY").lower() in {"1", "true", "yes", "on"}:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    if request.client is None:
        return ""
    return request.client.host or ""


def _ip_allowed(request: Request) -> bool:
    allowlist = [
        item.strip()
        for item in _env("STRIPE_WEBHOOK_IP_ALLOWLIST").split(",")
        if item.strip()
    ]
    if not allowlist:
        return True
    return _client_ip(request) in allowlist


def _webhook_json(status_code: int, payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload)


@router.post("/v1/webhooks/stripe")
async def stripe_webhook_listener_endpoint(request: Request) -> JSONResponse:
    """Receive a Stripe snapshot webhook and sync subscription items.

    The body is read raw. ``stripe.Webhook.construct_event`` checks the
    ``Stripe-Signature`` header against ``STRIPE_WEBHOOK_SECRET`` before any
    field is trusted. ``customer.subscription.created`` and
    ``customer.subscription.updated`` write the customer id and metered
    subscription item id to Supabase ``accounts``. ``customer.subscription.deleted``
    uses the same write so a canceled subscription stops receiving usage.
    """

    if not _ip_allowed(request):
        LOGGER.warning("stripe webhook rejected by ip allowlist")
        return _webhook_json(403, {"detail": "Source address is not allowed"})

    secret = _env("STRIPE_WEBHOOK_SECRET")
    if not secret:
        LOGGER.error("stripe webhook secret is not configured")
        return _webhook_json(500, {"detail": "Stripe webhook is not configured"})

    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    if not signature:
        return _webhook_json(400, {"detail": "Missing Stripe-Signature header"})

    try:
        event = await asyncio.to_thread(
            stripe.Webhook.construct_event,
            payload,
            signature,
            secret,
        )
    except stripe.error.SignatureVerificationError:
        LOGGER.warning("stripe webhook signature rejected")
        return _webhook_json(400, {"detail": "Invalid Stripe signature"})
    except ValueError:
        LOGGER.warning("stripe webhook payload rejected")
        return _webhook_json(400, {"detail": "Invalid Stripe payload"})

    event_data = _as_dict(event)
    event_id = event_data.get("id")
    event_type = event_data.get("type")
    if event_type not in HANDLED_SUBSCRIPTION_EVENTS:
        return _webhook_json(
            200,
            {
                "received": True,
                "handled": False,
                "event_id": event_id,
                "type": event_type,
            },
        )

    if not _supabase_configured():
        await asyncio.to_thread(
            record_billing_failure,
            operation="webhook_account_update",
            error=SupabaseError("Supabase is not configured"),
            payload={"event_id": event_id, "type": event_type},
        )
        return _webhook_json(500, {"detail": "Account store is not configured"})

    data = event_data.get("data")
    subscription = data.get("object") if isinstance(data, dict) else None
    if not isinstance(subscription, dict):
        return _webhook_json(
            400, {"detail": "Stripe event did not contain a subscription"}
        )
    try:
        account = await update_account_from_subscription(subscription)
    except (
        ValueError,
        SupabaseError,
        httpx.TimeoutException,
        httpx.TransportError,
    ) as exc:
        customer_id = None
        item_id = None
        try:
            customer_id = _customer_id(subscription)
            item_id = select_subscription_item_id(subscription)
        except ValueError:
            customer_id = None
            item_id = None
        await asyncio.to_thread(
            record_billing_failure,
            operation="webhook_account_update",
            error=exc,
            payload={"event_id": event_id, "type": event_type},
            stripe_customer_id=customer_id,
            stripe_subscription_item_id=item_id,
        )
        LOGGER.warning(
            "stripe subscription sync failed",
            extra={
                "event_fields": {
                    "event_id": event_id,
                    "event_type": event_type,
                    "error_type": exc.__class__.__name__,
                }
            },
        )
        return _webhook_json(500, {"detail": "Account update failed"})

    LOGGER.info(
        "stripe subscription synced",
        extra={
            "event_fields": {
                "event_id": event_id,
                "event_type": event_type,
                "stripe_customer_id": account.get("stripe_customer_id"),
                "stripe_subscription_item_id": account.get(
                    "stripe_subscription_item_id"
                ),
            }
        },
    )
    return _webhook_json(
        200,
        {
            "received": True,
            "handled": True,
            "event_id": event_id,
            "type": event_type,
            "account_id": account.get("id"),
        },
    )
