"""Deterministic test doubles (Milestone M2).

* :class:`FakeSupervisorAdapter` - a ``SupervisorPort`` conforming adapter that
  returns scripted, already-normalized PLAN/REVIEW payloads or raises a scripted
  ``SupervisorRunError``. It performs NO network I/O. It is the primary tool for
  testing the future ``SupervisorRouter`` (routing, fallback, validation, audit)
  and the FSM integration without any paid/live API call.

* :class:`FakeHttpTransport` - an injectable transport double that satisfies the
  same ``post_json`` / ``get_json`` seam as :class:`supervisor.transport.HttpTransport`.
  It records every request (for assertions) and returns or raises scripted
  responses. It never opens a socket.

Nothing here is imported by production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .errors import CATEGORY_UNKNOWN, SupervisorRunError
from .models import (
    HEALTH_READY,
    PLAN_OPERATION,
    REVIEW_OPERATION,
    ProviderConfig,
    ProviderHealth,
    ProviderUsage,
    SupervisorPlanRequest,
    SupervisorPlanResult,
    SupervisorReviewRequest,
    SupervisorReviewResult,
)
from .transport import TransportResponse

# --------------------------------------------------------------------------- #
# Default, schema-valid canonical payloads
# --------------------------------------------------------------------------- #


def fake_plan_payload(
    *,
    decision: str = "PLAN_READY",
    goal: str = "do the safe thing",
    files: Optional[list[str]] = None,
    changes: Optional[list[str]] = None,
    checks: Optional[list[str]] = None,
    architecture_risk: str = "none",
    open_questions: Optional[list[str]] = None,
    rollback: Optional[list[str]] = None,
    summary: str = "ok",
) -> dict:
    """A canonical-schema-valid PLAN payload (no controller-owned keys)."""
    return {
        "decision": decision,
        "goal": goal,
        "files": list(files if files is not None else ["a.py"]),
        "changes": list(changes if changes is not None else ["edit a.py"]),
        "checks": list(checks if checks is not None else ["run tests"]),
        "architecture_risk": architecture_risk,
        "open_questions": list(open_questions or []),
        "rollback": list(rollback if rollback is not None else ["revert a.py"]),
        "summary": summary,
    }


def fake_review_payload(
    *,
    verdict: str = "APPROVE",
    issues: Optional[list[str]] = None,
    required_changes: Optional[list[str]] = None,
    checks: Optional[list[str]] = None,
    architecture_risk: str = "none",
    summary: str = "looks good",
) -> dict:
    """A canonical-schema-valid REVIEW payload (no controller-owned keys)."""
    return {
        "verdict": verdict,
        "issues": list(issues or []),
        "required_changes": list(required_changes or []),
        "checks": list(checks if checks is not None else ["tests green"]),
        "architecture_risk": architecture_risk,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Injected transport double
# --------------------------------------------------------------------------- #


@dataclass
class FakeHttpResponse:
    """A scripted transport outcome (a successful response or a raised error)."""

    status_code: int = 200
    payload: Any = None
    text: str = ""
    error: Optional[Exception] = None


class FakeHttpTransport:
    """Injectable transport double; records requests, never touches the network."""

    def __init__(
        self,
        responses: Optional[list[FakeHttpResponse]] = None,
        default: Optional[FakeHttpResponse] = None,
    ) -> None:
        self.responses = list(responses or [])
        self.default = default or FakeHttpResponse(status_code=200, payload={"ok": True})
        self.requests: list[dict] = []

    def _next(self) -> FakeHttpResponse:
        if self.responses:
            return self.responses.pop(0)
        return self.default

    def _record(self, method: str, url: str, headers, json_body, timeout_sec) -> None:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "json": json_body,
                "timeout_sec": timeout_sec,
            }
        )

    def _respond(self, method: str, url: str, headers, json_body, timeout_sec):
        self._record(method, url, headers, json_body, timeout_sec)
        response = self._next()
        if response.error is not None:
            raise response.error
        payload = response.payload
        text = response.text
        raw_bytes = len(text) if text else 0
        return TransportResponse(
            status_code=response.status_code,
            payload=payload,
            text=text,
            raw_bytes=raw_bytes,
        )

    def post_json(self, url, *, headers=None, json_body=None, timeout_sec=None):
        return self._respond("POST", url, headers, json_body, timeout_sec)

    def get_json(self, url, *, headers=None, timeout_sec=None):
        return self._respond("GET", url, headers, None, timeout_sec)


# --------------------------------------------------------------------------- #
# Scripted adapter outcomes
# --------------------------------------------------------------------------- #


@dataclass
class FakeOutcome:
    """One scripted adapter outcome: a payload OR an error to raise."""

    payload: Optional[dict] = None
    error: Optional[Exception] = None
    usage: Optional[ProviderUsage] = None


class FakeSupervisorAdapter:
    """A ``SupervisorPort`` conforming, fully deterministic, offline adapter.

    Each operation consumes the next scripted :class:`FakeOutcome`; when only
    one outcome remains it is repeated (so repeated calls are stable). Usage,
    health and payloads are all configurable.
    """

    def __init__(
        self,
        provider_id: str = "fake",
        model: str = "fake-model",
        *,
        plan_outcomes: Optional[list[FakeOutcome]] = None,
        review_outcomes: Optional[list[FakeOutcome]] = None,
        health: Optional[ProviderHealth] = None,
        config: Optional[ProviderConfig] = None,
    ) -> None:
        self.provider_id = provider_id
        self.model = model
        self.config = config or ProviderConfig(
            provider_id=provider_id, display_name=provider_id, model=model
        )
        self._plan_outcomes = list(plan_outcomes) if plan_outcomes else [
            FakeOutcome(payload=fake_plan_payload())
        ]
        self._review_outcomes = list(review_outcomes) if review_outcomes else [
            FakeOutcome(payload=fake_review_payload())
        ]
        self._health = health if health is not None else ProviderHealth(
            provider_id=provider_id, status=HEALTH_READY, model=model
        )
        #: Every (operation, request) this fake was asked to perform.
        self.calls: list[tuple[str, Any]] = []

    @staticmethod
    def _pop(outcomes: list[FakeOutcome]) -> FakeOutcome:
        if len(outcomes) > 1:
            return outcomes.pop(0)
        return outcomes[0]

    def plan(self, request: SupervisorPlanRequest) -> SupervisorPlanResult:
        self.calls.append((PLAN_OPERATION, request))
        outcome = self._pop(self._plan_outcomes)
        if outcome.error is not None:
            raise outcome.error
        return SupervisorPlanResult(
            payload=dict(outcome.payload or {}),
            provider=self.provider_id,
            model=self.model,
            usage=outcome.usage,
        )

    def review(self, request: SupervisorReviewRequest) -> SupervisorReviewResult:
        self.calls.append((REVIEW_OPERATION, request))
        outcome = self._pop(self._review_outcomes)
        if outcome.error is not None:
            raise outcome.error
        return SupervisorReviewResult(
            payload=dict(outcome.payload or {}),
            provider=self.provider_id,
            model=self.model,
            usage=outcome.usage,
        )

    def health_check(self) -> ProviderHealth:
        return self._health

    # -- convenience constructors ------------------------------------------ #

    @classmethod
    def healthy(
        cls,
        provider_id: str = "fake",
        *,
        model: str = "fake-model",
        plan_payload: Optional[dict] = None,
        review_payload: Optional[dict] = None,
        health: Optional[ProviderHealth] = None,
    ) -> "FakeSupervisorAdapter":
        return cls(
            provider_id,
            model,
            plan_outcomes=[
                FakeOutcome(payload=dict(plan_payload or fake_plan_payload()))
            ],
            review_outcomes=[
                FakeOutcome(payload=dict(review_payload or fake_review_payload()))
            ],
            health=health,
        )

    @classmethod
    def with_plan_success(
        cls,
        provider_id: str = "fake",
        *,
        model: str = "fake-model",
        plan_payload: Optional[dict] = None,
    ) -> "FakeSupervisorAdapter":
        return cls.healthy(provider_id, model=model, plan_payload=plan_payload)

    @classmethod
    def with_review_verdict(
        cls, verdict: str, provider_id: str = "fake", *, model: str = "fake-model"
    ) -> "FakeSupervisorAdapter":
        return cls(
            provider_id,
            model,
            review_outcomes=[
                FakeOutcome(payload=fake_review_payload(verdict=verdict))
            ],
        )

    @classmethod
    def with_plan_error(
        cls,
        error: SupervisorRunError,
        provider_id: str = "fake",
        *,
        model: str = "fake-model",
    ) -> "FakeSupervisorAdapter":
        return cls(provider_id, model, plan_outcomes=[FakeOutcome(error=error)])

    @classmethod
    def with_retryable_plan_error(
        cls,
        category: str = "rate_limit",
        provider_id: str = "fake",
        *,
        model: str = "fake-model",
    ) -> "FakeSupervisorAdapter":
        error = SupervisorRunError(
            "provider temporarily unavailable",
            provider=provider_id,
            model=model,
            operation=PLAN_OPERATION,
            category=category,
        )
        return cls.with_plan_error(error, provider_id, model=model)

    @classmethod
    def with_non_retryable_plan_error(
        cls,
        category: str = "auth",
        provider_id: str = "fake",
        *,
        model: str = "fake-model",
    ) -> "FakeSupervisorAdapter":
        error = SupervisorRunError(
            "provider rejected the request",
            provider=provider_id,
            model=model,
            operation=PLAN_OPERATION,
            category=category,
        )
        return cls.with_plan_error(error, provider_id, model=model)

    @classmethod
    def with_malformed_plan(
        cls, provider_id: str = "fake", *, model: str = "fake-model"
    ) -> "FakeSupervisorAdapter":
        """A PLAN outcome that is intentionally NOT canonical-schema-valid."""
        return cls(
            provider_id,
            model,
            plan_outcomes=[FakeOutcome(payload={"decision": "PLAN_READY"})],
        )

    @classmethod
    def with_unknown_category_error(
        cls, provider_id: str = "fake", *, model: str = "fake-model"
    ) -> "FakeSupervisorAdapter":
        """A failure with no positive classification: must fail closed."""
        error = SupervisorRunError(
            "unclassified provider failure",
            provider=provider_id,
            model=model,
            operation=REVIEW_OPERATION,
            category=CATEGORY_UNKNOWN,
        )
        return cls(provider_id, model, review_outcomes=[FakeOutcome(error=error)])


