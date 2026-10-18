"""Admin issuance of agency API tokens.

The plaintext token is returned once. Supabase stores only its SHA-256 hash.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import re
import secrets
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

router = APIRouter()

_TOKEN_PREFIX = "ig_live_trial_"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class TokenGenerateRequest(BaseModel):
    """Body for POST /v1/tokens/generate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_email: str = Field(..., min_length=3, max_length=320)
    agency_name: str = Field(..., min_length=1, max_length=200)
    allotted_credits: int = Field(default=1000, ge=1, le=1_000_000)

    @field_validator("client_email")
    @classmethod
    def validate_client_email(cls, value: str) -> str:
        email = value.strip().lower()
        if _EMAIL_RE.fullmatch(email) is None:
            raise ValueError("client_email must be an email address")
        return email

    @field_validator("agency_name")
    @classmethod
    def validate_agency_name(cls, value: str) -> str:
        name = " ".join(value.split())
        if not name:
            raise ValueError("agency_name is empty")
        return name


class TokenGenerateResponse(BaseModel):
    """The only response that contains the usable token."""

    token: str
    client_email: str
    agency_name: str
    allotted_credits: int


def _master_key() -> str:
    return str(os.environ.get("INDEXGRID_MASTER_KEY") or "").strip()


def _admin_secret_matches(provided: str | None) -> bool:
    expected = _master_key()
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided, expected)


def generate_agency_token() -> str:
    """Return ig_live_trial_ plus 24 cryptographically random hex characters."""

    return _TOKEN_PREFIX + secrets.token_hex(12)


def hash_agency_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _insert_agency_token(row: dict[str, Any]) -> None:
    from main import _get_supabase_client

    _get_supabase_client().table("agency_tokens").insert(row).execute()


@router.post("/generate", response_model=TokenGenerateResponse)
async def generate_token(
    body: TokenGenerateRequest,
    x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret"),
) -> TokenGenerateResponse:
    """Issue one agency token after the admin secret matches INDEXGRID_MASTER_KEY."""

    if not _admin_secret_matches(x_admin_secret):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = generate_agency_token()
    row = {
        "token_hash": hash_agency_token(token),
        "allotted_credits": body.allotted_credits,
        "metadata": {
            "client_email": body.client_email,
            "agency_name": body.agency_name,
            "token_prefix": _TOKEN_PREFIX,
        },
    }
    try:
        await asyncio.to_thread(_insert_agency_token, row)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Token store is not configured"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Token store is unavailable"
        ) from exc

    return TokenGenerateResponse(
        token=token,
        client_email=body.client_email,
        agency_name=body.agency_name,
        allotted_credits=body.allotted_credits,
    )
