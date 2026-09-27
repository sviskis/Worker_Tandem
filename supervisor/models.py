"""Provider-neutral supervisor data models (Milestone M1).

These types are the ONLY contract the controller and the future
``SupervisorRouter`` see. Provider specifics (HTTP bodies, SDK objects,
Codex JSONL events) never leak through them.

Python / the FSM stays authoritative for operation identity:

* ``step_no`` and ``attempt`` are carried on the *request*
  (controller -> provider) as context;
* they are never part of a canonical provider *result payload*, and a
  provider that echoes them is rejected by validation (see
  ``supervisor.schemas``).

This module is intentionally free of ``tkinter``/``sqlite3``/``httpx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Operation names
# --------------------------------------------------------------------------- #

PLAN_OPERATION = "PLAN"
REVIEW_OPERATION = "REVIEW"

# --------------------------------------------------------------------------- #
# Canonical PLAN decision + REVIEW verdict vocabularies
# --------------------------------------------------------------------------- #

#: PLAN decisions (unchanged from the Codex PLAN contract).
PLAN_READY = "PLAN_READY"
PLAN_BLOCKED = "BLOCKED"
PLAN_DECISIONS = (PLAN_READY, PLAN_BLOCKED)

#: Canonical REVIEW verdicts (unchanged from the Codex REVIEW contract).
REVIEW_APPROVE = "APPROVE"
REVIEW_REVISE = "REVISE"
REVIEW_BLOCKED = "BLOCKED"
REVIEW_ARCHITECTURE_DECISION_REQUIRED = "ARCHITECTURE_DECISION_REQUIRED"
REVIEW_VERDICTS = (
    REVIEW_APPROVE,
    REVIEW_REVISE,
    REVIEW_BLOCKED,
    REVIEW_ARCHITECTURE_DECISION_REQUIRED,
)

# --------------------------------------------------------------------------- #
# Provider health vocabulary
# --------------------------------------------------------------------------- #

HEALTH_READY = "READY"
HEALTH_KEY_MISSING = "KEY_MISSING"
HEALTH_AUTH_ERROR = "AUTH_ERROR"
HEALTH_RATE_LIMIT = "RATE_LIMIT"
HEALTH_QUOTA_LIMIT = "QUOTA_LIMIT"
HEALTH_NETWORK_ERROR = "NETWORK_ERROR"
HEALTH_PROVIDER_ERROR = "PROVIDER_ERROR"

HEALTH_STATUSES = (
    HEALTH_READY,
    HEALTH_KEY_MISSING,
    HEALTH_AUTH_ERROR,
    HEALTH_RATE_LIMIT,
    HEALTH_QUOTA_LIMIT,
    HEALTH_NETWORK_ERROR,
    HEALTH_PROVIDER_ERROR,
)

# --------------------------------------------------------------------------- #
# Provider configuration / telemetry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderConfig:
    """Static, provider-neutral configuration for one supervisor provider.

    ``api_key_source`` is a *reference* (for example ``"env:OPENAI_API_KEY"``),
    never the secret itself. Secrets are resolved at call time and are never
    persisted, logged or displayed.
    """

    provider_id: str
    display_name: str
    model: Optional[str] = None
    enabled: bool = True
    api_base: Optional[str] = None
    api_key_source: Optional[str] = None
    timeout_sec: int = 180
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    supports_structured_output: bool = False
    cost_policy: str = "standard"
    priority: int = 100


@dataclass(frozen=True)
class ProviderUsage:
    """Token/cost usage reported by a provider.

    Fields are ``None`` when the provider did not report them. Cost is only
    recorded when real pricing metadata is available - never guessed.
    """

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    latency_ms: Optional[int] = None


@dataclass(frozen=True)
class ProviderHealth:
    """Cheap, side-effect-free health of a single provider."""

    provider_id: str
    status: str
    detail: str = ""
    model: Optional[str] = None
    latency_ms: Optional[int] = None

    @property
    def ready(self) -> bool:
        return self.status == HEALTH_READY


@dataclass(frozen=True)
class TelemetryRecord:
    """One provider invocation, safe to audit (no secrets)."""

    provider: str
    model: Optional[str]
    operation: str
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    attempt: int = 0
    fallback_from: Optional[str] = None
    fallback_reason: Optional[str] = None
    success: bool = False
    category: Optional[str] = None
    sanitized_error: Optional[str] = None
    result_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "attempt": self.attempt,
            "fallback_from": self.fallback_from,
            "fallback_reason": self.fallback_reason,
            "success": self.success,
            "category": self.category,
            "sanitized_error": self.sanitized_error,
            "result_hash": self.result_hash,
        }


# --------------------------------------------------------------------------- #
# PLAN / REVIEW requests (controller -> provider) and results (provider ->
# controller). Results never carry provider-owned step identity.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SupervisorPlanRequest:
    step_no: int
    attempt: int
    risk: str
    requires_human: bool
    prompt: str
    context: dict[str, Any]
    schema: dict[str, Any]
    provider_hint: Optional[str] = None


@dataclass(frozen=True)
class SupervisorPlanResult:
    """A normalized, schema-valid PLAN payload.

    ``payload`` carries NO provider-owned ``step_no``/``attempt``: the
    controller injects the authoritative identity after the fact.
    """

    payload: dict[str, Any]
    provider: str
    model: Optional[str] = None
    usage: Optional[ProviderUsage] = None
    fallback_from: Optional[str] = None
    fallback_reason: Optional[str] = None
    #: M8 routing metadata. Provider attempts are separate from project
    #: (Cline) attempts and never consume one.
    provider_attempt: int = 0
    route_chain: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupervisorReviewRequest:
    step_no: int
    attempt: int
    risk: str
    requires_human: bool
    prompt: str
    context: dict[str, Any]
    schema: dict[str, Any]
    provider_hint: Optional[str] = None


@dataclass(frozen=True)
class SupervisorReviewResult:
    """A normalized, schema-valid REVIEW payload (canonical verdicts only)."""

    payload: dict[str, Any]
    provider: str
    model: Optional[str] = None
    usage: Optional[ProviderUsage] = None
    fallback_from: Optional[str] = None
    fallback_reason: Optional[str] = None
    #: M8 routing metadata. Provider attempts are separate from project
    #: (Cline) attempts and never consume one.
    provider_attempt: int = 0
    route_chain: tuple[str, ...] = ()

