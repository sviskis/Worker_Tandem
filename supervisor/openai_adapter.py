"""OpenAI API supervisor adapter (Milestone M6).

API CONTRACT (V1) - deliberately ONE style, documented, isolated here
--------------------------------------------------------------------------
``POST {api_base}/chat/completions`` with
``response_format={"type": "json_object"}``, plus ``GET {api_base}/models`` for
the cheap health probe. Base/model/endpoint live only in this module.

Why chat completions and not the Responses API: its envelope is identical in
shape to DeepSeek's, which (a) proves the shared ``openai_compatible``
abstraction instead of assuming it, (b) gives
``usage.prompt_tokens/completion_tokens/total_tokens`` that normalize 1:1 into
``ProviderUsage``, and (c) reuses the existing transport untouched. Because the
endpoint and body are built only here, migrating to another OpenAI endpoint
later cannot ripple into the controller or the router.

Everything generic (key handling, model preconditions, request body, transport
call, usage extraction, health mapping, error stamping) comes from
:class:`supervisor.openai_compatible.OpenAICompatibleSupervisorAdapter`.
"""

from __future__ import annotations

from typing import Any, Optional

from .errors import (
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_MODEL,
    CATEGORY_NOT_FOUND,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_TEMPORARY_5XX,
    CATEGORY_UNKNOWN,
    SupervisorRunError,
    sanitize_diagnostic,
)
from .models import ProviderConfig
from .openai_compatible import (
    OpenAICompatibleSupervisorAdapter,
    api_endpoint,
    error_haystack,
    extract_message_content,
    matches_hints,
    stamp_run_error,
    strict_json_object,
)

# --------------------------------------------------------------------------- #
# Provider identity + centralized defaults
# --------------------------------------------------------------------------- #

PROVIDER_OPENAI = "openai"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SEC = 120


def default_openai_config(**overrides: Any) -> ProviderConfig:
    """The ONE place OpenAI's default base URL and model are expressed."""
    values: dict[str, Any] = {
        "provider_id": PROVIDER_OPENAI,
        "display_name": "OpenAI",
        "model": DEFAULT_OPENAI_MODEL,
        "enabled": True,
        "api_base": DEFAULT_OPENAI_API_BASE,
        "api_key_source": f"env:{OPENAI_API_KEY_ENV}",
        "timeout_sec": DEFAULT_OPENAI_TIMEOUT_SEC,
        "max_output_tokens": None,
        "temperature": None,
        "supports_structured_output": True,
        "cost_policy": "standard",
        "priority": 2,
    }
    values.update(overrides)
    return ProviderConfig(**values)


def openai_endpoint(api_base: Optional[str], path: str) -> str:
    """Centralized endpoint construction (delegates to the shared helper)."""
    return api_endpoint(api_base, path, default_base=DEFAULT_OPENAI_API_BASE)


# --------------------------------------------------------------------------- #
# OpenAI-specific error classification (evidence first)
# --------------------------------------------------------------------------- #

#: Structured error codes/types meaning "hard billing / no credits": these are
#: NON-retryable even when the HTTP status is 429.
_PAYMENT_CODES = frozenset(
    {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
        "payment_required",
        "insufficient_balance",
    }
)

#: Billing wording - used ONLY when no structured code is present.
_PAYMENT_HINTS = (
    "insufficient_quota",
    "insufficient balance",
    "exceeded your current quota",
    "billing",
    "payment required",
    "no credits",
)

#: Structured codes meaning the configured model is unusable.
_MODEL_CODES = frozenset(
    {"model_not_found", "model_not_available", "model_disabled", "invalid_model"}
)

#: Model wording - applied ONLY to a 404, never as a global rule.
_MODEL_HINTS = (
    "does not exist",
    "model not found",
    "no access to model",
    "unsupported model",
    "invalid model",
)

#: Evidence that a throttled request is a TEMPORARY quota condition. Without
#: this, "quota" wording fails closed as non-retryable billing.
_RESET_HINTS = (
    "try again at",
    "try again in",
    "retry after",
    "resets in",
    "reset at",
    "temporarily",
)


def classify_openai_error(exc: SupervisorRunError) -> Optional[str]:
    """OpenAI-specific refinement, strongest evidence first.

    Returns ``None`` to keep the transport's conservative baseline:

    * 401/403 -> ``auth``, 402 -> ``payment_required``, 429 -> ``rate_limit``,
      400 -> ``invalid_request``, 404 -> ``not_found``,
      500/502/503/504 -> ``provider_unavailable``, timeout/network/transport.
    """
    code = (exc.provider_error_code or "").strip().lower()
    text = error_haystack(exc)

    if code in _PAYMENT_CODES:  # 1. explicit billing code wins outright
        return CATEGORY_PAYMENT_REQUIRED
    if code in _MODEL_CODES:  # 2. explicit model code
        return CATEGORY_INVALID_MODEL
    if exc.category == CATEGORY_RATE_LIMIT and matches_hints(text, _RESET_HINTS):
        return CATEGORY_QUOTA_WITH_RESET  # 3. throttled WITH reset evidence
    if matches_hints(text, _PAYMENT_HINTS):  # 4. billing wording, no reset
        return CATEGORY_PAYMENT_REQUIRED
    if exc.category == CATEGORY_NOT_FOUND and matches_hints(text, _MODEL_HINTS):
        return CATEGORY_INVALID_MODEL  # 5. a 404 that names the model
    if (
        exc.category == CATEGORY_UNKNOWN
        and isinstance(exc.http_status, int)
        and 500 <= exc.http_status < 600
    ):
        return CATEGORY_TEMPORARY_5XX  # 6. any other 5xx is temporary
    return None


def refine_provider_error(
    exc: SupervisorRunError,
    *,
    provider: str,
    model: Optional[str],
    operation: str,
) -> SupervisorRunError:
    """OpenAI-specific refinement plus shared supervisor-context stamping."""
    return stamp_run_error(
        exc,
        provider=provider,
        model=model,
        operation=operation,
        category=classify_openai_error(exc),
    )


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class OpenAISupervisorAdapter(OpenAICompatibleSupervisorAdapter):
    """A real OpenAI API ``SupervisorPort`` implementation.

    Only the provider-specific pieces live here: identity, defaults and the two
    hooks (strict JSON parsing, OpenAI error classification).
    """

    provider_id = PROVIDER_OPENAI
    default_api_base = DEFAULT_OPENAI_API_BASE
    default_model = DEFAULT_OPENAI_MODEL
    default_timeout_sec = DEFAULT_OPENAI_TIMEOUT_SEC
    default_key_env = OPENAI_API_KEY_ENV

    # -- provider-specific hooks ------------------------------------------- #

    def _refine_error(
        self,
        exc: SupervisorRunError,
        *,
        provider: str,
        model: Optional[str],
        operation: str,
    ) -> SupervisorRunError:
        return refine_provider_error(
            exc, provider=provider, model=model, operation=operation
        )

    def _parse_content(self, payload: Any, operation: str) -> dict:
        """content -> JSON object using the STRICT extractor.

        Accepted: a plain JSON object, a single fenced object, surrounding
        whitespace. Rejected: prose before/after the object, two concatenated
        objects, arrays and anything else non-JSON.
        """
        content = extract_message_content(payload)
        if not content:
            raise SupervisorRunError(
                "provider returned no message content",
                provider=self.provider_id,
                model=self.model,
                operation=operation,
                category=CATEGORY_CONTRACT,
                retryable=False,
                sanitized_message="OpenAI response contained no message content",
            )
        parsed = strict_json_object(content)
        if parsed is None:
            raise SupervisorRunError(
                "provider returned malformed JSON",
                provider=self.provider_id,
                model=self.model,
                operation=operation,
                category=CATEGORY_CONTRACT,
                retryable=False,
                sanitized_message=sanitize_diagnostic(
                    f"not a strict JSON object: {content[:200]}"
                ),
            )
        return parsed

