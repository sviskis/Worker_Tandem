"""Supervisor router tests (milestone M4).

Pins the M4 routing contract: MANUAL selection of a single registered provider
(``codex_legacy``), fail-closed selection errors, provider metadata, health
collection and - importantly - that NO automatic fallback exists yet.

No test performs a network call or spawns a process; adapters are the M2
deterministic fakes.
"""

from __future__ import annotations

import socket
import subprocess

import pytest

from supervisor import (
    DEFAULT_LEGACY_PROVIDER,
    POLICY_MODE_MANUAL,
    PROVIDER_CODEX_LEGACY,
    SUPERVISOR_ERROR_EVENT,
    SUPERVISOR_RESULT_EVENT,
    SUPERVISOR_SELECTED_EVENT,
    FakeSupervisorAdapter,
    FakeOutcome,
    ProviderConfig,
    ProviderHealth,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorReviewRequest,
    SupervisorRouter,
    SupervisorRunError,
    fake_plan_payload,
    fake_review_payload,
    supervisor_plan_schema,
    supervisor_review_schema,
)
from supervisor.errors import CATEGORY_INVALID_REQUEST
from supervisor.models import (
    HEALTH_KEY_MISSING,
    HEALTH_PROVIDER_ERROR,
    HEALTH_READY,
    PLAN_OPERATION,
    REVIEW_OPERATION,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def plan_request(step_no: int = 1, attempt: int = 1, risk: str = "LOW"):
    return SupervisorPlanRequest(
        step_no=step_no,
        attempt=attempt,
        risk=risk,
        requires_human=False,
        prompt="PLAN prompt",
        context={},
        schema=supervisor_plan_schema(),
    )


def review_request(step_no: int = 1, attempt: int = 1, risk: str = "LOW"):
    return SupervisorReviewRequest(
        step_no=step_no,
        attempt=attempt,
        risk=risk,
        requires_human=False,
        prompt="REVIEW prompt",
        context={},
        schema=supervisor_review_schema(),
    )


def legacy_adapter(
    provider_id: str = PROVIDER_CODEX_LEGACY, *, enabled: bool = True, **kwargs
):
    """A deterministic, enabled stand-in for the legacy adapter."""
    return FakeSupervisorAdapter(
        provider_id,
        "codex-model",
        config=ProviderConfig(
            provider_id=provider_id, display_name=provider_id, enabled=enabled
        ),
        **kwargs,
    )


def router_with(adapters: dict, **policy_kwargs) -> SupervisorRouter:
    return SupervisorRouter(adapters, SupervisorPolicy(**policy_kwargs))


# --------------------------------------------------------------------------- #
# 1-2  Selection
# --------------------------------------------------------------------------- #


def test_plan_selects_codex_legacy():
    adapter = legacy_adapter()
    router = router_with({PROVIDER_CODEX_LEGACY: adapter})

    result = router.plan(plan_request(), risk="LOW", requires_human=False)

    assert result.provider == PROVIDER_CODEX_LEGACY
    assert result.payload["decision"] == "PLAN_READY"
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] == PLAN_OPERATION
    assert router.selected_provider(PLAN_OPERATION) == PROVIDER_CODEX_LEGACY


def test_review_selects_codex_legacy():
    adapter = legacy_adapter()
    router = router_with({PROVIDER_CODEX_LEGACY: adapter})

    result = router.review(review_request(), risk="LOW", requires_human=False)

    assert result.provider == PROVIDER_CODEX_LEGACY
    assert result.payload["verdict"] == "APPROVE"
    assert adapter.calls[0][0] == REVIEW_OPERATION


# --------------------------------------------------------------------------- #
# 3-4  Selection failures fail closed
# --------------------------------------------------------------------------- #


def test_missing_provider_fails_closed():
    router = router_with(
        {PROVIDER_CODEX_LEGACY: legacy_adapter()}, plan_provider="nope"
    )

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False
    assert info.value.operation == PLAN_OPERATION


def test_empty_provider_fails_closed():
    router = router_with({}, plan_provider="", review_provider="")

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False


def test_disabled_provider_fails_closed():
    adapter = legacy_adapter(enabled=False)
    router = router_with({PROVIDER_CODEX_LEGACY: adapter})

    with pytest.raises(SupervisorRunError) as info:
        router.review(review_request())

    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False
    assert adapter.calls == []  # never executed


# --------------------------------------------------------------------------- #
# 5  PLAN and REVIEW may use different providers
# --------------------------------------------------------------------------- #


def test_plan_and_review_providers_may_differ():
    planner = legacy_adapter("planner")
    reviewer = legacy_adapter("reviewer")
    router = router_with(
        {"planner": planner, "reviewer": reviewer},
        plan_provider="planner",
        review_provider="reviewer",
    )

    plan = router.plan(plan_request())
    review = router.review(review_request())

    assert plan.provider == "planner"
    assert review.provider == "reviewer"
    assert len(planner.calls) == 1 and len(reviewer.calls) == 1


def test_empty_review_provider_falls_back_to_plan_provider():
    policy = SupervisorPolicy(plan_provider="planner", review_provider="")
    assert policy.provider_for(REVIEW_OPERATION) == "planner"
    assert policy.provider_for(PLAN_OPERATION) == "planner"


# --------------------------------------------------------------------------- #
# 6  Health collection
# --------------------------------------------------------------------------- #


def test_health_collection():
    healthy = legacy_adapter("healthy")
    missing = FakeSupervisorAdapter(
        "missing",
        "m",
        health=ProviderHealth(
            provider_id="missing", status=HEALTH_KEY_MISSING, detail="no key"
        ),
    )
    router = router_with({"healthy": healthy, "missing": missing})

    health = router.health()

    assert set(health) == {"healthy", "missing"}
    assert health["healthy"].status == HEALTH_READY
    assert health["missing"].status == HEALTH_KEY_MISSING


def test_health_failure_is_contained():
    class Broken:
        model = None

        def plan(self, request):
            raise AssertionError("not used")

        def review(self, request):
            raise AssertionError("not used")

        def health_check(self):
            raise RuntimeError("boom")

    health = router_with({"broken": Broken()}).health()
    assert health["broken"].status == HEALTH_PROVIDER_ERROR


# --------------------------------------------------------------------------- #
# 7  Provider metadata (no fallback in M4)
# --------------------------------------------------------------------------- #


def test_provider_metadata_returned():
    adapter = legacy_adapter()
    router = router_with({PROVIDER_CODEX_LEGACY: adapter})

    result = router.plan(plan_request(step_no=4, attempt=2))

    assert result.provider == PROVIDER_CODEX_LEGACY
    assert result.model == "codex-model"
    # M4 has no fallback, so the fallback metadata is explicitly empty.
    assert result.fallback_from is None
    assert result.fallback_reason is None

    record = router.calls[-1]
    assert record["operation"] == PLAN_OPERATION
    assert record["provider"] == PROVIDER_CODEX_LEGACY
    assert record["step_no"] == 4
    assert record["attempt"] == 2
    assert record["fallback_from"] is None


def test_summary_is_read_only():
    router = router_with({PROVIDER_CODEX_LEGACY: legacy_adapter()})
    summary = router.summary()
    assert summary["mode"] == POLICY_MODE_MANUAL
    assert summary["plan_provider"] == PROVIDER_CODEX_LEGACY
    assert summary["review_provider"] == PROVIDER_CODEX_LEGACY
    assert summary["registered"] == [PROVIDER_CODEX_LEGACY]


# --------------------------------------------------------------------------- #
# 8  No automatic fallback in M4
# --------------------------------------------------------------------------- #


def failing_adapter(provider_id: str, category: str = "provider_unavailable"):
    return FakeSupervisorAdapter(
        provider_id,
        "m",
        plan_outcomes=[
            FakeOutcome(
                error=SupervisorRunError(
                    "provider unavailable",
                    provider=provider_id,
                    operation=PLAN_OPERATION,
                    category=category,
                )
            )
        ],
        config=ProviderConfig(provider_id, provider_id, enabled=True),
    )


def test_no_fallback_in_m4():
    primary = failing_adapter("primary")
    backup = legacy_adapter("backup")
    router = router_with(
        {"primary": primary, "backup": backup},
        plan_provider="primary",
        review_provider="primary",
        fallback_chain=("backup",),
        max_provider_retries=0,
    )

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request())

    assert info.value.retryable is True
    assert backup.calls == []  # M4 never chains providers
    assert [c["provider"] for c in router.calls] == ["primary"]


# --------------------------------------------------------------------------- #
# 9  No live API calls
# --------------------------------------------------------------------------- #


def test_no_live_api_calls(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("live call attempted")

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)

    router = router_with({PROVIDER_CODEX_LEGACY: legacy_adapter()})
    assert router.plan(plan_request()).payload["decision"] == "PLAN_READY"
    assert router.review(review_request()).payload["verdict"] == "APPROVE"
    assert router.health()[PROVIDER_CODEX_LEGACY].status == HEALTH_READY


# --------------------------------------------------------------------------- #
# Policy model (minimal V1, serializable)
# --------------------------------------------------------------------------- #


def test_policy_defaults_and_round_trip():
    policy = SupervisorPolicy()
    assert policy.mode == POLICY_MODE_MANUAL
    assert policy.plan_provider == DEFAULT_LEGACY_PROVIDER
    assert policy.fallback_chain == ()
    assert policy.max_provider_retries == 0  # no fallback in M4

    data = policy.to_dict()
    assert data["fallback_chain"] == []
    assert SupervisorPolicy.from_dict(data) == policy
    assert SupervisorPolicy.from_dict(None) == SupervisorPolicy()


def test_default_legacy_provider_matches_adapter_constant():
    assert DEFAULT_LEGACY_PROVIDER == PROVIDER_CODEX_LEGACY


# --------------------------------------------------------------------------- #
# Audit hooks
# --------------------------------------------------------------------------- #


def test_audit_selected_and_result_events():
    events: list[tuple[str, dict]] = []
    adapter = legacy_adapter()
    router = SupervisorRouter(
        {PROVIDER_CODEX_LEGACY: adapter},
        SupervisorPolicy(),
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )

    router.plan(plan_request(step_no=3, attempt=1))

    assert [k for k, _ in events] == [
        SUPERVISOR_SELECTED_EVENT,
        SUPERVISOR_RESULT_EVENT,
    ]
    selected = events[0][1]
    assert selected["operation"] == PLAN_OPERATION
    assert selected["provider"] == PROVIDER_CODEX_LEGACY
    assert selected["step_no"] == 3
    assert selected["attempt"] == 1
    assert selected["mode"] == POLICY_MODE_MANUAL
    assert selected["fallback_from"] is None

    result = events[1][1]
    assert result["result_hash"]
    assert result["fallback_from"] is None
    assert result["fallback_reason"] is None


def test_audit_error_event_is_sanitized():
    events: list[tuple[str, dict]] = []
    router = SupervisorRouter(
        {"primary": failing_adapter("primary", category="auth")},
        SupervisorPolicy(plan_provider="primary", review_provider="primary"),
        emit=lambda event_type, payload: events.append((event_type, payload)),
    )

    with pytest.raises(SupervisorRunError):
        router.plan(plan_request(step_no=2))

    assert [k for k, _ in events] == [
        SUPERVISOR_SELECTED_EVENT,
        SUPERVISOR_ERROR_EVENT,
    ]
    error = events[-1][1]
    assert error["retryable"] is False
    assert error["category"] == "auth"
    assert error["step_no"] == 2
    assert error["result_hash"] is None
    assert error["sanitized_error"]


def test_audit_failure_never_aborts_the_call():
    def broken_emit(event_type, payload):
        raise RuntimeError("audit down")

    router = SupervisorRouter(
        {PROVIDER_CODEX_LEGACY: legacy_adapter()}, emit=broken_emit
    )
    assert router.plan(plan_request()).payload["decision"] == "PLAN_READY"


def test_router_without_emit_hook_works():
    router = SupervisorRouter({PROVIDER_CODEX_LEGACY: legacy_adapter()})
    assert router.review(review_request()).payload["verdict"] == "APPROVE"



