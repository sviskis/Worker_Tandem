"""The provider-neutral ``SupervisorPort`` interface (Milestone M1).

The controller and the future ``SupervisorRouter`` depend ONLY on this
Protocol. Provider-specific HTTP/SDK/subprocess behavior stays inside the
adapters.

The Protocol intentionally declares *methods only* so that
``isinstance(obj, SupervisorPort)`` works against any conforming adapter
(including test doubles). Adapters are expected to also expose
``provider_id`` and ``model`` as plain attributes.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from .errors import (
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_API_KEY,
    SupervisorRunError,
    sanitize_diagnostic,
)
from .models import (
    ProviderHealth,
    SupervisorPlanRequest,
    SupervisorPlanResult,
    SupervisorReviewRequest,
    SupervisorReviewResult,
)
from .schemas import (
    supervisor_plan_schema,
    supervisor_review_schema,
    validate_plan_payload,
    validate_review_payload,
)


@runtime_checkable
class SupervisorPort(Protocol):
    """Advisory PLAN/REVIEW provider. Never authoritative for state."""

    def plan(self, request: SupervisorPlanRequest) -> SupervisorPlanResult:
        ...

    def review(self, request: SupervisorReviewRequest) -> SupervisorReviewResult:
        ...

    def health_check(self) -> ProviderHealth:
        ...


def normalize_plan_payload(
    payload: Any,
    *,
    provider: str = "",
    model: Optional[str] = None,
    operation: str = "PLAN",
) -> dict:
    """Validate a PLAN payload against the canonical schema, failing closed.

    A malformed payload raises ``SupervisorRunError(category="contract")``
    and therefore can never mutate the FSM.
    """
    problems = validate_plan_payload(payload)
    if problems:
        raise SupervisorRunError(
            "provider PLAN payload does not satisfy the canonical schema",
            provider=provider,
            model=model,
            operation=operation,
            category=CATEGORY_CONTRACT,
            sanitized_message="; ".join(problems[:5]),
        )
    return dict(payload)


def normalize_review_payload(
    payload: Any,
    *,
    provider: str = "",
    model: Optional[str] = None,
    operation: str = "REVIEW",
) -> dict:
    """Validate a REVIEW payload against the canonical schema, failing closed."""
    problems = validate_review_payload(payload)
    if problems:
        raise SupervisorRunError(
            "provider REVIEW payload does not satisfy the canonical schema",
            provider=provider,
            model=model,
            operation=operation,
            category=CATEGORY_CONTRACT,
            sanitized_message="; ".join(problems[:5]),
        )
    return dict(payload)


def plan_schema() -> dict:
    """Convenience accessor (kept next to the port for adapter symmetry)."""
    return supervisor_plan_schema()


def review_schema() -> dict:
    return supervisor_review_schema()


# --------------------------------------------------------------------------- #
# API-key resolution (shared by every HTTP provider adapter)
# --------------------------------------------------------------------------- #

#: Reference syntax used by ``ProviderConfig.api_key_source``.
ENV_KEY_PREFIX = "env:"


def api_key_env_name(api_key_source: Optional[str]) -> str:
    """The environment variable NAME referenced by ``api_key_source``.

    ``"env:DEEPSEEK_API_KEY"`` -> ``"DEEPSEEK_API_KEY"``. Only the name is ever
    handled here (and only the name may appear in diagnostics).
    """
    source = str(api_key_source or "").strip()
    if source.lower().startswith(ENV_KEY_PREFIX):
        return source[len(ENV_KEY_PREFIX):].strip()
    return source


def resolve_api_key(
    api_key_source: Optional[str], *, env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Resolve an ``api_key_source`` reference to a secret, or ``None``.

    V1 supports environment variables only. The value is handed straight to the
    caller's request headers and is NEVER persisted, logged, returned in an
    error, or exposed through any ``to_dict``.
    """
    name = api_key_env_name(api_key_source)
    if not name:
        return None
    environ = os.environ if env is None else env
    value = environ.get(name)
    value = str(value).strip() if value else ""
    return value or None


def key_missing_run_error(
    provider: str,
    *,
    model: Optional[str] = None,
    operation: str = "",
    api_key_source: Optional[str] = None,
) -> SupervisorRunError:
    """A locally missing/empty API key: always non-retryable.

    Only the environment-variable NAME is mentioned - never a value.
    """
    name = api_key_env_name(api_key_source) or "(unset)"
    return SupervisorRunError(
        "provider API key is not configured",
        provider=provider,
        model=model,
        operation=operation,
        category=CATEGORY_INVALID_API_KEY,
        retryable=False,
        sanitized_message=sanitize_diagnostic(
            f"missing API key for {provider}; set the environment variable {name}"
        ),
    )
