"""Unified, provider-neutral supervisor error taxonomy (Milestone M1).

One canonical category set covers every provider. Anything uncertain FAILS
CLOSED as non-retryable: a wrong "retryable" verdict would let a provider
outage consume a Cline attempt, which the review-retry contract forbids.

This module is free of ``tkinter``/``sqlite3``/``httpx``.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .models import ProviderUsage

# --------------------------------------------------------------------------- #
# Retryable categories: "the provider was temporarily unable to serve us"
# --------------------------------------------------------------------------- #

CATEGORY_QUOTA_WITH_RESET = "quota_with_reset"
CATEGORY_RATE_LIMIT = "rate_limit"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_TRANSPORT = "transport"
CATEGORY_NETWORK = "network"
CATEGORY_PROVIDER_UNAVAILABLE = "provider_unavailable"
CATEGORY_TEMPORARY_5XX = "temporary_5xx"

# --------------------------------------------------------------------------- #
# Non-retryable categories: fail closed
# --------------------------------------------------------------------------- #

CATEGORY_AUTH = "auth"
CATEGORY_INVALID_API_KEY = "invalid_api_key"
CATEGORY_BILLING_DISABLED = "billing_disabled"
CATEGORY_PAYMENT_REQUIRED = "payment_required"
CATEGORY_INVALID_MODEL = "invalid_model"
CATEGORY_INVALID_REQUEST = "invalid_request"
CATEGORY_NOT_FOUND = "not_found"
CATEGORY_CONTRACT = "contract"
CATEGORY_SCHEMA = "schema"
CATEGORY_UNSUPPORTED_FEATURE = "unsupported_feature"
CATEGORY_RESPONSE_TOO_LARGE = "response_too_large"
CATEGORY_UNKNOWN = "unknown"

RETRYABLE_CATEGORIES = frozenset(
    {
        CATEGORY_QUOTA_WITH_RESET,
        CATEGORY_RATE_LIMIT,
        CATEGORY_TIMEOUT,
        CATEGORY_TRANSPORT,
        CATEGORY_NETWORK,
        CATEGORY_PROVIDER_UNAVAILABLE,
        CATEGORY_TEMPORARY_5XX,
    }
)

NON_RETRYABLE_CATEGORIES = frozenset(
    {
        CATEGORY_AUTH,
        CATEGORY_INVALID_API_KEY,
        CATEGORY_BILLING_DISABLED,
        CATEGORY_PAYMENT_REQUIRED,
        CATEGORY_INVALID_MODEL,
        CATEGORY_INVALID_REQUEST,
        CATEGORY_NOT_FOUND,
        CATEGORY_CONTRACT,
        CATEGORY_SCHEMA,
        CATEGORY_UNSUPPORTED_FEATURE,
        CATEGORY_RESPONSE_TOO_LARGE,
        CATEGORY_UNKNOWN,
    }
)

ALL_CATEGORIES = RETRYABLE_CATEGORIES | NON_RETRYABLE_CATEGORIES


def is_retryable_category(category: str) -> bool:
    """True only for a positively classified retryable provider condition."""
    return category in RETRYABLE_CATEGORIES


# --------------------------------------------------------------------------- #
# Secret redaction - every diagnostic passes through here before it is raised,
# logged or persisted.
# --------------------------------------------------------------------------- #

_MAX_DIAGNOSTIC_CHARS = 800

_REDACTIONS = (
    # Anthropic-style keys first (they share the OpenAI-style prefix too).
    (re.compile(r"(?i)\bsk-ant-[A-Za-z0-9_\-]{6,}\b"), "***"),
    (re.compile(r"(?i)\bsk-[A-Za-z0-9_\-]{6,}\b"), "***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{6,}"), "Bearer ***"),
    (
        re.compile(
            r"(?i)(authorization|proxy-authorization|x-api-key|api[_-]?key|apikey"
            r"|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret)"
            r'\b\s*[:=]\s*"?[^\s",}]+'
        ),
        r"\1: ***",
    ),
)


def sanitize_diagnostic(text: Any, max_chars: int = _MAX_DIAGNOSTIC_CHARS) -> str:
    """One-line, credential-redacted diagnostic safe to persist or display.

    Redacts bearer tokens, ``sk-...``/``sk-ant-...`` keys and any
    ``api_key``/``authorization``/``x-api-key`` style ``name=value`` pair.
    """
    value = str(text or "").replace("\x00", " ").strip()
    value = " ".join(value.split())
    for pattern, replacement in _REDACTIONS:
        value = pattern.sub(replacement, value)
    return value[:max_chars]


# --------------------------------------------------------------------------- #
# HTTP status -> category (baseline mapping; adapters may refine it later from
# the provider-specific error body).
# --------------------------------------------------------------------------- #


def category_for_http_status(status: Optional[int]) -> str:
    """Conservative baseline category for an HTTP failure status.

    A 404 is deliberately mapped to the generic ``not_found`` (NOT
    ``invalid_model``): only an adapter that understands a provider's own
    error body may promote it to ``invalid_model``.
    """
    if status in (401, 403):
        return CATEGORY_AUTH
    if status == 402:
        return CATEGORY_PAYMENT_REQUIRED
    if status == 404:
        return CATEGORY_NOT_FOUND
    if status == 429:
        return CATEGORY_RATE_LIMIT
    if status in (500, 502, 503, 504):
        return CATEGORY_PROVIDER_UNAVAILABLE
    if isinstance(status, int) and 400 <= status < 500:
        return CATEGORY_INVALID_REQUEST
    return CATEGORY_UNKNOWN


# --------------------------------------------------------------------------- #
# Error types
# --------------------------------------------------------------------------- #


class SupervisorError(Exception):
    """Base class for all provider-neutral supervisor errors."""


class SupervisorRunError(SupervisorError):
    """A classified supervisor/provider failure.

    ``retryable`` defaults to the category's classification and therefore
    fails closed (``False``) for anything unrecognized. ``str(exc)`` and
    ``to_dict()`` are always sanitized: no API key, bearer token or
    Authorization header can survive them.

    The optional ``origin_*`` fields carry the *originating provider's own*
    raw classification (for example legacy Codex ``"quota"`` plus its return
    code) so audit payloads keep byte-level parity. They are never used for
    routing or retry decisions - those always use the canonical fields.
    """

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        model: Optional[str] = None,
        operation: str = "",
        category: str = CATEGORY_UNKNOWN,
        retryable: Optional[bool] = None,
        http_status: Optional[int] = None,
        provider_error_code: Optional[str] = None,
        sanitized_message: str = "",
        raw_response_ref: Optional[str] = None,
        usage: Optional[ProviderUsage] = None,
        origin_provider: Optional[str] = None,
        origin_category: Optional[str] = None,
        origin_code: Optional[str] = None,
        origin_source: Optional[str] = None,
    ) -> None:
        if retryable is None:
            retryable = is_retryable_category(category)
        safe = sanitize_diagnostic(sanitized_message or message)
        super().__init__(safe or category)
        self.message = safe
        self.provider = provider
        self.model = model
        self.operation = operation
        self.category = category
        self.retryable = bool(retryable)
        self.http_status = http_status
        self.provider_error_code = provider_error_code
        self.sanitized_message = safe
        self.raw_response_ref = raw_response_ref
        self.usage = usage
        # Originating provider's own raw classification (audit parity only).
        self.origin_provider = origin_provider
        self.origin_category = origin_category
        self.origin_code = origin_code
        self.origin_source = origin_source

    def __str__(self) -> str:
        head = "/".join(part for part in (self.operation, self.category) if part)
        head = head or "supervisor"
        via = f" via {self.provider}" if self.provider else ""
        status = f" (HTTP {self.http_status})" if self.http_status else ""
        return f"{head}{via}{status}: {self.sanitized_message}".strip()

    def to_dict(self) -> dict:
        """Sanitized, audit-safe representation (never contains secrets)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "category": self.category,
            "retryable": self.retryable,
            "http_status": self.http_status,
            "provider_error_code": self.provider_error_code,
            "sanitized_message": self.sanitized_message,
            "raw_response_ref": self.raw_response_ref,
            "origin_provider": self.origin_provider,
            "origin_category": self.origin_category,
            "origin_code": self.origin_code,
            "origin_source": self.origin_source,
        }

