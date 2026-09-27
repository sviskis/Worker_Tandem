"""Supervisor AUTO routing + provider fallback tests (milestone M8).

Pins the M8 contract:

* MANUAL stays the M4 behaviour exactly (one provider, no retry, no fallback, no
  health skipping) unless a policy explicitly opts into AUTO/fallback;
* AUTO selects PLAN and REVIEW per risk from ONE ``SupervisorPolicy`` object,
  walks a de-duplicated provider chain and never loops;
* a retryable provider outage gets EXACTLY one same-provider retry and then a
  fallback (quota_with_reset, rate_limit, timeout, transport, network,
  provider_unavailable, temporary_5xx); malformed/schema-invalid output may also
  retry once and then fall back (policy-gated);
* auth / invalid_api_key / billing_disabled / payment_required / invalid_model /
  unsupported_feature / local invalid_request / unknown FAIL CLOSED - no
  automatic fallback;
* provider attempts are NOT project attempts: no Cline attempt is consumed, no
  Cline run, no PLAN re-run during REVIEW, accepted report preserved;
* AUTO may skip KEY_MISSING / AUTH_ERROR / disabled providers using a cached,
  bounded-lifetime health snapshot - never persisted, never per request;
* audit events record the routing decision (provider, model, operation, step,
  attempt, risk, category, retryable, fallback_from/to, latency, usage,
  result_hash) and never a secret.

Offline only: providers are the M2 deterministic ``FakeSupervisorAdapter`` (plus
the M4 controller with its Codex runner stubbed). No network, no subprocess.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import socket
import subprocess
import sys

import pytest

from supervisor import (
    BUDGET_CHEAP,
    BUDGET_MODES,
    BUDGET_PREMIUM,
    BUDGET_STANDARD,
    DEFAULT_LEGACY_PROVIDER,
    ID_ANTHROPIC,
    ID_DEEPSEEK,
    ID_OPENAI,
    POLICY_MODE_AUTO,
    POLICY_MODE_MANUAL,
    PROVIDER_CODEX_LEGACY,
    PROVIDER_DEEPSEEK,
    PROVIDER_OPENAI,
    RETRYABLE_FALLBACK_CATEGORIES,
    SKIP_HEALTH_STATUSES,
    SUPERVISOR_ALL_FAILED_EVENT,
    SUPERVISOR_ERROR_EVENT,
    SUPERVISOR_FALLBACK_EVENT,
    SUPERVISOR_RESULT_EVENT,
    SUPERVISOR_RETRY_EVENT,
    SUPERVISOR_SELECTED_EVENT,
    FakeOutcome,
    FakeSupervisorAdapter,
    ProviderConfig,
    ProviderHealth,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorReviewRequest,
    SupervisorRouter,
    SupervisorRunError,
    default_auto_policy,
    default_fallback_chain,
    default_risk_routes,
    fake_plan_payload,
    fake_review_payload,
    supervisor_plan_schema,
    supervisor_review_schema,
)
from supervisor.errors import (
    CATEGORY_AUTH,
    CATEGORY_BILLING_DISABLED,
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_API_KEY,
    CATEGORY_INVALID_MODEL,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_SCHEMA,
    CATEGORY_TEMPORARY_5XX,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    CATEGORY_UNKNOWN,
    CATEGORY_UNSUPPORTED_FEATURE,
)
from supervisor.models import (
    HEALTH_AUTH_ERROR,
    HEALTH_KEY_MISSING,
    HEALTH_READY,
    PLAN_OPERATION,
    REVIEW_OPERATION,
)

#: Categories that MUST trigger a fallback (M8 §4).
RETRYABLE_WITH_FALLBACK = (
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    CATEGORY_NETWORK,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_TEMPORARY_5XX,
)

#: Categories that MUST fail closed, whatever the chain looks like (M8 §5).
NON_RETRYABLE_NO_FALLBACK = (
    CATEGORY_AUTH,
    CATEGORY_INVALID_API_KEY,
    CATEGORY_BILLING_DISABLED,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_INVALID_MODEL,
    CATEGORY_UNSUPPORTED_FEATURE,
    CATEGORY_INVALID_REQUEST,
)

#: A realistic-looking credential that must never reach an audit payload.
SECRET_KEY = "sk-" + "D" * 32


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


def adapter(
    provider_id: str,
    *,
    plan=None,
    review=None,
    health=None,
    enabled: bool = True,
    model=None,
):
    """A deterministic fake provider (defaults: one successful outcome)."""
    resolved_model = model or f"{provider_id}-model"
    return FakeSupervisorAdapter(
        provider_id,
        resolved_model,
        plan_outcomes=plan,
        review_outcomes=review,
        health=health,
        config=ProviderConfig(
            provider_id=provider_id,
            display_name=provider_id,
            model=resolved_model,
            enabled=enabled,
        ),
    )


def outcome_error(
    provider_id: str,
    category: str,
    *,
    operation: str = PLAN_OPERATION,
    retryable=None,
    message=None,
) -> FakeOutcome:
    """A scripted failing outcome carrying a canonical classification."""
    return FakeOutcome(
        error=SupervisorRunError(
            message or f"{provider_id} failed: {category}",
            provider=provider_id,
            model=f"{provider_id}-model",
            operation=operation,
            category=category,
            retryable=retryable,
        )
    )


def outcome_ok(*, operation: str = PLAN_OPERATION) -> FakeOutcome:
    if operation == REVIEW_OPERATION:
        return FakeOutcome(payload=fake_review_payload())
    return FakeOutcome(payload=fake_plan_payload())


def failing(
    provider_id: str,
    category: str,
    *,
    operation: str = PLAN_OPERATION,
    retryable=None,
    message=None,
    **kwargs,
):
    """A fake that always fails with ``category`` for one operation."""
    outcome = outcome_error(
        provider_id, category, operation=operation, retryable=retryable, message=message
    )
    if operation == REVIEW_OPERATION:
        return adapter(provider_id, review=[outcome], **kwargs)
    return adapter(provider_id, plan=[outcome], **kwargs)


def healthy(provider_id: str, **kwargs):
    """A fake that always succeeds for both operations."""
    return adapter(provider_id, **kwargs)


def health(provider_id: str, status: str):
    return ProviderHealth(provider_id=provider_id, status=status, model="probe")


class Recorder:
    """Records every audit event the router emits."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, dict(payload)))

    def types(self) -> list[str]:
        return [event_type for event_type, _ in self.events]

    def of(self, event_type: str) -> list[dict]:
        return [payload for kind, payload in self.events if kind == event_type]


def chain_router(adapters: dict, recorder: Recorder | None = None, **policy_kwargs):
    """An AUTO router whose chain is exactly deepseek -> openai (test default)."""
    policy_kwargs.setdefault("fallback_chain", (ID_DEEPSEEK, ID_OPENAI))
    policy_kwargs.setdefault("allow_codex_legacy", False)
    return SupervisorRouter(
        adapters, default_auto_policy(**policy_kwargs), recorder or Recorder()
    )


def manual_router(adapters: dict, recorder: Recorder | None = None, **policy_kwargs):
    return SupervisorRouter(
        adapters, SupervisorPolicy(**policy_kwargs), recorder or Recorder()
    )


# --------------------------------------------------------------------------- #
# 1-6  One policy object owns every route
# --------------------------------------------------------------------------- #


def test_default_policy_is_still_manual_codex_legacy():
    policy = SupervisorPolicy()
    assert policy.mode == POLICY_MODE_MANUAL
    assert policy.is_auto is False
    assert policy.plan_provider == PROVIDER_CODEX_LEGACY
    assert policy.review_provider == PROVIDER_CODEX_LEGACY
    assert policy.fallback_chain == ()
    assert policy.max_provider_retries == 0
    # MANUAL: exactly one candidate; chain/risk routes are ignored.
    assert policy.candidates_for(PLAN_OPERATION, "HIGH") == (PROVIDER_CODEX_LEGACY,)


def test_router_provider_ids_match_the_adapters():
    assert ID_DEEPSEEK == PROVIDER_DEEPSEEK
    assert ID_OPENAI == PROVIDER_OPENAI
    assert DEFAULT_LEGACY_PROVIDER == PROVIDER_CODEX_LEGACY
    # Anthropic (M7) is the declared HIGH/REVIEW preference and - since M7.4 -
    # a registered provider; this test uses a fake to stay at the router level.
    assert ID_ANTHROPIC == "anthropic"


def test_default_auto_policy_matches_the_spec():
    policy = default_auto_policy()
    assert policy.mode == POLICY_MODE_AUTO
    assert policy.is_auto is True
    assert policy.budget_mode == BUDGET_STANDARD
    assert policy.max_provider_retries == 1
    assert policy.allow_codex_legacy is True
    assert policy.allow_manual_fallback is False
    assert policy.fallback_on_contract_failure is True
    assert policy.allow_unknown_fallback is False
    assert policy.skip_unhealthy_providers is True
    routes = policy.risk_routes
    assert routes["LOW"] == {
        PLAN_OPERATION: (ID_DEEPSEEK,),
        REVIEW_OPERATION: (ID_DEEPSEEK,),
    }
    assert routes["MEDIUM"] == {
        PLAN_OPERATION: (ID_DEEPSEEK,),
        REVIEW_OPERATION: (ID_DEEPSEEK,),
    }
    assert routes["HIGH"] == {
        PLAN_OPERATION: (ID_OPENAI,),
        REVIEW_OPERATION: (ID_ANTHROPIC,),
    }
    assert policy.fallback_chain == default_fallback_chain()


def test_default_risk_routes_helper_is_fresh_and_complete():
    first, second = default_risk_routes(), default_risk_routes()
    assert first == second
    assert first is not second
    for risk in ("LOW", "MEDIUM", "HIGH"):
        assert set(first[risk]) == {PLAN_OPERATION, REVIEW_OPERATION}


def test_auto_policy_round_trips_through_dict():
    policy = default_auto_policy()
    data = policy.to_dict()
    assert data["mode"] == POLICY_MODE_AUTO
    assert data["fallback_chain"] == list(default_fallback_chain())
    assert data["risk_routes"]["HIGH"][PLAN_OPERATION] == [ID_OPENAI]
    assert data["budget_mode"] == BUDGET_STANDARD
    restored = SupervisorPolicy.from_dict(data)
    assert restored == policy
    # Unknown/blank risk levels use the MEDIUM route but keep the whole chain.
    for unknown in ("", "UNKNOWN", None):
        assert restored.candidates_for(PLAN_OPERATION, unknown) == (
            ID_DEEPSEEK,
            ID_OPENAI,
            ID_ANTHROPIC,
            DEFAULT_LEGACY_PROVIDER,
        )


def test_auto_is_opt_in_through_the_policy_only():
    deepseek = healthy(ID_DEEPSEEK)
    manual = SupervisorPolicy(
        plan_provider=ID_DEEPSEEK, review_provider=ID_DEEPSEEK, fallback_chain=()
    )
    assert manual.mode == POLICY_MODE_MANUAL
    assert manual.candidates_for(PLAN_OPERATION, "LOW") == (ID_DEEPSEEK,)
    router = SupervisorRouter({ID_DEEPSEEK: deepseek}, manual)
    assert router.plan(plan_request(risk="LOW")).provider == ID_DEEPSEEK


# --------------------------------------------------------------------------- #
# 7-11  AUTO route selection (LOW / MEDIUM / HIGH, PLAN vs REVIEW) and budget
# --------------------------------------------------------------------------- #


def test_low_risk_uses_deepseek_for_plan_and_review():
    deepseek, openai = healthy(ID_DEEPSEEK), healthy(ID_OPENAI)
    router = SupervisorRouter(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, default_auto_policy()
    )

    assert router.plan(plan_request(risk="LOW")).provider == ID_DEEPSEEK
    assert router.review(review_request(risk="LOW")).provider == ID_DEEPSEEK
    assert [call[0] for call in openai.calls] == []


def test_medium_risk_uses_deepseek():
    deepseek, openai = healthy(ID_DEEPSEEK), healthy(ID_OPENAI)
    router = SupervisorRouter(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, default_auto_policy()
    )

    assert router.plan(plan_request(risk="MEDIUM")).provider == ID_DEEPSEEK
    assert router.review(review_request(risk="MEDIUM")).provider == ID_DEEPSEEK
    assert [call[0] for call in openai.calls] == []


def test_high_risk_plan_uses_openai():
    deepseek, openai = healthy(ID_DEEPSEEK), healthy(ID_OPENAI)
    router = SupervisorRouter(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, default_auto_policy()
    )

    result = router.plan(plan_request(risk="HIGH"))

    assert result.provider == ID_OPENAI
    assert [call[0] for call in deepseek.calls] == []


def test_high_risk_review_uses_anthropic():
    deepseek, anthropic = healthy(ID_DEEPSEEK), healthy(ID_ANTHROPIC)
    router = SupervisorRouter(
        {ID_DEEPSEEK: deepseek, ID_ANTHROPIC: anthropic}, default_auto_policy()
    )

    result = router.review(review_request(risk="HIGH"))

    assert result.provider == ID_ANTHROPIC
    assert result.payload["verdict"] == "APPROVE"
    assert [call[0] for call in deepseek.calls] == []


def test_plan_and_review_may_use_different_providers():
    deepseek, anthropic = healthy(ID_DEEPSEEK), healthy(ID_ANTHROPIC)
    router = SupervisorRouter(
        {ID_DEEPSEEK: deepseek, ID_ANTHROPIC: anthropic}, default_auto_policy()
    )

    plan = router.plan(plan_request(risk="HIGH"))  # openai route, not registered
    review = router.review(review_request(risk="HIGH"))

    assert plan.provider == ID_DEEPSEEK
    assert review.provider == ID_ANTHROPIC
    # REVIEW must not re-run PLAN (M8 §6).
    assert [call[0] for call in deepseek.calls] == [PLAN_OPERATION]
    assert [call[0] for call in anthropic.calls] == [REVIEW_OPERATION]


def test_budget_mode_orders_the_fallback_tail():
    cheap = default_auto_policy(budget_mode=BUDGET_CHEAP)
    standard = default_auto_policy(budget_mode=BUDGET_STANDARD)
    premium = default_auto_policy(budget_mode=BUDGET_PREMIUM)

    assert cheap.candidates_for(PLAN_OPERATION, "HIGH") == (
        ID_OPENAI,
        ID_DEEPSEEK,
        ID_ANTHROPIC,
        DEFAULT_LEGACY_PROVIDER,
    )
    assert standard.candidates_for(PLAN_OPERATION, "HIGH") == cheap.candidates_for(
        PLAN_OPERATION, "HIGH"
    )
    assert premium.candidates_for(PLAN_OPERATION, "HIGH") == (
        ID_OPENAI,
        ID_ANTHROPIC,
        ID_DEEPSEEK,
        DEFAULT_LEGACY_PROVIDER,
    )
    # The route primary always keeps slot 0.
    assert cheap.candidates_for(REVIEW_OPERATION, "HIGH") == (
        ID_ANTHROPIC,
        ID_DEEPSEEK,
        ID_OPENAI,
        DEFAULT_LEGACY_PROVIDER,
    )
    assert premium.candidates_for(REVIEW_OPERATION, "HIGH") == (
        ID_ANTHROPIC,
        ID_OPENAI,
        ID_DEEPSEEK,
        DEFAULT_LEGACY_PROVIDER,
    )
    assert set(BUDGET_MODES) == {BUDGET_CHEAP, BUDGET_STANDARD, BUDGET_PREMIUM}




def test_codex_legacy_is_reached_only_after_every_modern_provider_failed():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = failing(ID_OPENAI, CATEGORY_TIMEOUT)
    anthropic = failing(ID_ANTHROPIC, CATEGORY_NETWORK)
    legacy = healthy(DEFAULT_LEGACY_PROVIDER)
    router = SupervisorRouter(
        {
            ID_DEEPSEEK: deepseek,
            ID_OPENAI: openai,
            ID_ANTHROPIC: anthropic,
            DEFAULT_LEGACY_PROVIDER: legacy,
        },
        default_auto_policy(),
    )

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == DEFAULT_LEGACY_PROVIDER
    assert result.fallback_from == ID_ANTHROPIC
    assert result.fallback_reason == CATEGORY_NETWORK
    assert result.route_chain == (
        ID_DEEPSEEK,
        ID_OPENAI,
        ID_ANTHROPIC,
        DEFAULT_LEGACY_PROVIDER,
    )
    # Bounded: one retry each, then the next provider. Never a loop.
    assert len(deepseek.calls) == 2
    assert len(openai.calls) == 2
    assert len(anthropic.calls) == 2
    assert len(legacy.calls) == 1


def test_codex_legacy_can_be_disabled():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = failing(ID_OPENAI, CATEGORY_TIMEOUT)
    legacy = healthy(DEFAULT_LEGACY_PROVIDER)
    router = SupervisorRouter(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai, DEFAULT_LEGACY_PROVIDER: legacy},
        default_auto_policy(allow_codex_legacy=False),
    )

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    assert [call[0] for call in legacy.calls] == []
    assert info.value.retryable is True
    assert ID_DEEPSEEK in str(info.value)
    assert ID_OPENAI in str(info.value)
    assert DEFAULT_LEGACY_PROVIDER not in str(info.value)


# --------------------------------------------------------------------------- #
# 12-24  Fallback rules: what falls back, what fails closed, how often
# --------------------------------------------------------------------------- #


def test_fallback_category_set_tracks_the_error_taxonomy():
    from supervisor.errors import RETRYABLE_CATEGORIES

    assert RETRYABLE_FALLBACK_CATEGORIES == RETRYABLE_CATEGORIES


def test_same_provider_is_retried_once_before_falling_back():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.plan(plan_request(risk="LOW"))

    assert [call[0] for call in deepseek.calls] == [PLAN_OPERATION, PLAN_OPERATION]
    assert len(openai.calls) == 1
    assert result.provider == ID_OPENAI
    assert result.provider_attempt == 1
    assert result.fallback_from == ID_DEEPSEEK
    assert result.fallback_reason == CATEGORY_RATE_LIMIT
    assert result.route_chain == (ID_DEEPSEEK, ID_OPENAI)


def test_retry_succeeds_before_any_fallback_is_needed():
    deepseek = adapter(
        ID_DEEPSEEK,
        plan=[outcome_error(ID_DEEPSEEK, CATEGORY_TIMEOUT), outcome_ok()],
    )
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_DEEPSEEK
    assert result.provider_attempt == 2
    assert result.fallback_from is None
    assert result.fallback_reason is None
    assert [call[0] for call in openai.calls] == []


@pytest.mark.parametrize("category", RETRYABLE_WITH_FALLBACK)
def test_retryable_categories_fall_back(category):
    deepseek = failing(ID_DEEPSEEK, category)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.fallback_reason == category
    assert len(deepseek.calls) == 2  # one retry, never more


@pytest.mark.parametrize("category", NON_RETRYABLE_NO_FALLBACK)
def test_non_retryable_categories_fail_closed(category):
    deepseek = failing(ID_DEEPSEEK, category, retryable=False)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    assert info.value.category == category
    assert info.value.retryable is False
    assert [call[0] for call in openai.calls] == []  # no fallback
    assert len(deepseek.calls) == 1  # and no retry either




@pytest.mark.parametrize("category", [CATEGORY_CONTRACT, CATEGORY_SCHEMA])
def test_malformed_or_schema_invalid_output_may_fall_back_once(category):
    deepseek = failing(ID_DEEPSEEK, category)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.fallback_reason == category
    assert len(deepseek.calls) == 2  # one retry, then the fallback


def test_contract_fallback_can_be_disabled():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_CONTRACT)
    openai = healthy(ID_OPENAI)
    router = chain_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        fallback_on_contract_failure=False,
    )

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    assert info.value.category == CATEGORY_CONTRACT
    assert [call[0] for call in openai.calls] == []
    assert len(deepseek.calls) == 1


def test_unknown_fails_closed_by_default():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_UNKNOWN, retryable=False)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    assert info.value.category == CATEGORY_UNKNOWN
    assert info.value.retryable is False
    assert [call[0] for call in openai.calls] == []


def test_unknown_falls_back_when_the_policy_allows_it():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_UNKNOWN, retryable=False)
    openai = healthy(ID_OPENAI)
    router = chain_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, allow_unknown_fallback=True
    )

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.fallback_reason == CATEGORY_UNKNOWN
    assert len(deepseek.calls) == 1  # explicit opt-in: fall back, no retry


def test_a_duplicate_provider_is_never_tried_twice():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = failing(ID_OPENAI, CATEGORY_TIMEOUT)
    router = chain_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        fallback_chain=(ID_DEEPSEEK, ID_OPENAI, ID_DEEPSEEK, ID_OPENAI, ID_DEEPSEEK),
    )

    assert router.policy.candidates_for(PLAN_OPERATION, "LOW") == (
        ID_DEEPSEEK,
        ID_OPENAI,
    )
    with pytest.raises(SupervisorRunError):
        router.plan(plan_request(risk="LOW"))

    # Bounded: 2 providers x (1 attempt + 1 retry), then it stops - no loop.
    assert len(deepseek.calls) == 2
    assert len(openai.calls) == 2


def test_review_fallback_never_re_runs_plan():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT, operation=REVIEW_OPERATION)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.review(review_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.payload["verdict"] == "APPROVE"
    assert [call[0] for call in deepseek.calls] == [REVIEW_OPERATION, REVIEW_OPERATION]
    assert [call[0] for call in openai.calls] == [REVIEW_OPERATION]


# --------------------------------------------------------------------------- #
# 25-30  MANUAL semantics: the user-selected provider is the only one called
# --------------------------------------------------------------------------- #


def test_manual_uses_exactly_the_selected_provider():
    deepseek, openai = healthy(ID_DEEPSEEK), healthy(ID_OPENAI)
    router = manual_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        plan_provider=ID_DEEPSEEK,
        review_provider=ID_DEEPSEEK,
        fallback_chain=(ID_OPENAI,),
    )

    result = router.plan(plan_request(risk="HIGH"))

    assert result.provider == ID_DEEPSEEK
    assert [call[0] for call in openai.calls] == []
    assert router.policy.candidates_for(PLAN_OPERATION, "HIGH") == (ID_DEEPSEEK,)


def test_manual_ignores_the_auto_risk_routes():
    deepseek, openai = healthy(ID_DEEPSEEK), healthy(ID_OPENAI)
    router = manual_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        plan_provider=ID_DEEPSEEK,
        review_provider=ID_DEEPSEEK,
        risk_routes={
            "HIGH": {PLAN_OPERATION: (ID_OPENAI,), REVIEW_OPERATION: (ID_OPENAI,)}
        },
    )

    assert router.plan(plan_request(risk="HIGH")).provider == ID_DEEPSEEK


def test_manual_does_not_fall_back_on_a_retryable_failure():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = healthy(ID_OPENAI)
    router = manual_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        plan_provider=ID_DEEPSEEK,
        review_provider=ID_DEEPSEEK,
        fallback_chain=(ID_OPENAI,),
    )

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    assert info.value.category == CATEGORY_RATE_LIMIT
    assert info.value.retryable is True  # still deferrable for REVIEW
    assert len(deepseek.calls) == 1  # no retry in MANUAL either
    assert [call[0] for call in openai.calls] == []


def test_manual_fallback_requires_an_explicit_opt_in():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = healthy(ID_OPENAI)
    router = manual_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        plan_provider=ID_DEEPSEEK,
        review_provider=ID_DEEPSEEK,
        fallback_chain=(ID_OPENAI,),
        allow_manual_fallback=True,
    )

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.fallback_from == ID_DEEPSEEK
    assert len(deepseek.calls) == 1  # max_provider_retries defaults to 0
    assert len(openai.calls) == 1


def test_manual_never_skips_an_unhealthy_provider():
    deepseek = healthy(ID_DEEPSEEK, health=health(ID_DEEPSEEK, HEALTH_KEY_MISSING))
    openai = healthy(ID_OPENAI, health=health(ID_OPENAI, HEALTH_READY))
    router = manual_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        plan_provider=ID_DEEPSEEK,
        review_provider=ID_DEEPSEEK,
        fallback_chain=(ID_OPENAI,),
    )

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_DEEPSEEK
    assert len(deepseek.calls) == 1
    assert [call[0] for call in openai.calls] == []




# --------------------------------------------------------------------------- #
# 31-37  Health-aware AUTO routing (cached, bounded, never persisted)
# --------------------------------------------------------------------------- #


def test_skip_health_statuses_are_the_documented_ones():
    assert SKIP_HEALTH_STATUSES == {HEALTH_KEY_MISSING, HEALTH_AUTH_ERROR}


def test_auto_skips_a_key_missing_provider():
    deepseek = healthy(ID_DEEPSEEK, health=health(ID_DEEPSEEK, HEALTH_KEY_MISSING))
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.fallback_from == ID_DEEPSEEK
    assert result.fallback_reason == f"skipped:{HEALTH_KEY_MISSING}"
    assert [call[0] for call in deepseek.calls] == []


def test_auto_skips_an_auth_error_provider():
    deepseek = healthy(ID_DEEPSEEK, health=health(ID_DEEPSEEK, HEALTH_AUTH_ERROR))
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.fallback_reason == f"skipped:{HEALTH_AUTH_ERROR}"
    assert [call[0] for call in deepseek.calls] == []


def test_auto_skips_a_disabled_provider():
    deepseek = healthy(ID_DEEPSEEK, enabled=False)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_OPENAI
    assert result.fallback_reason == "skipped:disabled"
    assert [call[0] for call in deepseek.calls] == []


def test_auto_skips_an_unregistered_provider():
    deepseek = healthy(ID_DEEPSEEK)
    router = chain_router({ID_DEEPSEEK: deepseek})

    assert router.policy.candidates_for(PLAN_OPERATION, "LOW") == (
        ID_DEEPSEEK,
        ID_OPENAI,
    )
    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_DEEPSEEK
    assert result.route_chain == (ID_DEEPSEEK, ID_OPENAI)
    assert result.fallback_from is None  # the first candidate succeeded


def test_health_skipping_can_be_disabled_by_policy():
    deepseek = healthy(ID_DEEPSEEK, health=health(ID_DEEPSEEK, HEALTH_KEY_MISSING))
    openai = healthy(ID_OPENAI)
    router = chain_router(
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, skip_unhealthy_providers=False
    )

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_DEEPSEEK
    assert len(deepseek.calls) == 1


class CountingAdapter(FakeSupervisorAdapter):
    """A fake that counts health probes (M8 §10: no per-request probing)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.health_calls = 0

    def health_check(self):
        self.health_calls += 1
        return super().health_check()


def test_health_is_probed_once_per_cache_window():
    deepseek = CountingAdapter(ID_DEEPSEEK, "deepseek-model")
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    for _ in range(3):
        assert router.plan(plan_request(risk="LOW")).provider == ID_DEEPSEEK

    assert deepseek.health_calls == 1  # cached across requests


def test_health_cache_can_be_invalidated():
    deepseek = CountingAdapter(ID_DEEPSEEK, "deepseek-model")
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    router.plan(plan_request(risk="LOW"))
    router.forget_health(ID_DEEPSEEK)
    router.plan(plan_request(risk="LOW"))
    assert deepseek.health_calls == 2

    router.forget_health()
    router.plan(plan_request(risk="LOW"))
    assert deepseek.health_calls == 3


def test_the_last_remaining_provider_is_always_attempted():
    """Nothing is left to fall back to, so the final candidate is tried."""
    deepseek = healthy(ID_DEEPSEEK, health=health(ID_DEEPSEEK, HEALTH_KEY_MISSING))
    router = chain_router({ID_DEEPSEEK: deepseek}, fallback_chain=(ID_DEEPSEEK,))

    result = router.plan(plan_request(risk="LOW"))

    assert result.provider == ID_DEEPSEEK
    assert len(deepseek.calls) == 1


# --------------------------------------------------------------------------- #
# 38-44  Audit trail, all-providers-failed, routing metadata
# --------------------------------------------------------------------------- #


def test_audit_records_a_same_provider_retry_then_a_fallback():
    recorder = Recorder()
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, recorder)

    router.plan(plan_request(step_no=2, attempt=3, risk="LOW"))

    assert recorder.types() == [
        SUPERVISOR_SELECTED_EVENT,
        SUPERVISOR_ERROR_EVENT,
        SUPERVISOR_SELECTED_EVENT,
        SUPERVISOR_RETRY_EVENT,
        SUPERVISOR_ERROR_EVENT,
        SUPERVISOR_FALLBACK_EVENT,
        SUPERVISOR_SELECTED_EVENT,
        SUPERVISOR_RESULT_EVENT,
    ]
    selected = recorder.of(SUPERVISOR_SELECTED_EVENT)
    assert [event["provider"] for event in selected] == [
        ID_DEEPSEEK,
        ID_DEEPSEEK,
        ID_OPENAI,
    ]
    assert [event["provider_attempt"] for event in selected] == [1, 2, 1]
    assert selected[2]["step_no"] == 2
    assert selected[2]["attempt"] == 3
    assert selected[2]["risk"] == "LOW"

    retry = recorder.of(SUPERVISOR_RETRY_EVENT)[0]
    assert retry["provider"] == ID_DEEPSEEK
    assert retry["provider_attempt"] == 2
    assert retry["category"] == CATEGORY_RATE_LIMIT
    assert retry["retryable"] is True

    fallback = recorder.of(SUPERVISOR_FALLBACK_EVENT)[0]
    assert fallback["fallback_from"] == ID_DEEPSEEK
    assert fallback["fallback_to"] == ID_OPENAI
    assert fallback["category"] == CATEGORY_RATE_LIMIT
    assert fallback["retryable"] is True
    assert fallback["sanitized_error"]
    assert fallback["step_no"] == 2
    assert fallback["attempt"] == 3

    errors = recorder.of(SUPERVISOR_ERROR_EVENT)
    assert [event["category"] for event in errors] == [
        CATEGORY_RATE_LIMIT,
        CATEGORY_RATE_LIMIT,
    ]
    assert all(event["result_hash"] is None for event in errors)

    result_event = recorder.of(SUPERVISOR_RESULT_EVENT)[0]
    assert result_event["provider"] == ID_OPENAI
    assert result_event["model"] == "openai-model"
    assert result_event["provider_attempt"] == 1
    assert result_event["fallback_from"] == ID_DEEPSEEK
    assert result_event["fallback_reason"] == CATEGORY_RATE_LIMIT
    assert result_event["route_chain"] == [ID_DEEPSEEK, ID_OPENAI]
    assert result_event["result_hash"]
    assert isinstance(result_event["latency_ms"], int)
    assert result_event["usage"] is None  # the fake reports no usage


def test_audit_records_an_all_failed_route():
    recorder = Recorder()
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = failing(ID_OPENAI, CATEGORY_TIMEOUT)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, recorder)

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    failure = recorder.of(SUPERVISOR_ALL_FAILED_EVENT)[0]
    assert failure["providers"] == [ID_DEEPSEEK, ID_DEEPSEEK, ID_OPENAI, ID_OPENAI]
    assert failure["categories"] == [
        CATEGORY_RATE_LIMIT,
        CATEGORY_RATE_LIMIT,
        CATEGORY_TIMEOUT,
        CATEGORY_TIMEOUT,
    ]
    assert failure["route_chain"] == [ID_DEEPSEEK, ID_OPENAI]
    assert failure["retryable"] is True
    assert failure["step_no"] == 1
    assert "rate_limit" in failure["sanitized_error"]

    # The aggregate error keeps the M4 controller contract: one neutral error.
    assert info.value.retryable is True
    assert info.value.category == CATEGORY_TIMEOUT
    assert info.value.provider == ID_OPENAI
    assert info.value.operation == PLAN_OPERATION
    assert recorder.of(SUPERVISOR_RESULT_EVENT) == []
    assert router.route_history[-1]["success"] is False


def test_all_failed_for_provider_side_output_failures_is_retryable():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_CONTRACT)
    openai = failing(ID_OPENAI, CATEGORY_SCHEMA)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    # Malformed provider output is still the provider's fault, not the project's.
    assert info.value.retryable is True
    assert info.value.category == CATEGORY_SCHEMA


def test_all_providers_unusable_is_not_retryable():
    deepseek = healthy(ID_DEEPSEEK, health=health(ID_DEEPSEEK, HEALTH_KEY_MISSING))
    openai = healthy(ID_OPENAI, enabled=False)
    recorder = Recorder()
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, recorder)

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request(risk="LOW"))

    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False
    assert HEALTH_KEY_MISSING in info.value.sanitized_message
    assert "disabled" in info.value.sanitized_message
    assert [call[0] for call in deepseek.calls] == []
    assert [call[0] for call in openai.calls] == []
    assert recorder.of(SUPERVISOR_ALL_FAILED_EVENT)[0]["retryable"] is False


# --------------------------------------------------------------------------- #
# 45-48  Routing metadata, identity and credential hygiene
# --------------------------------------------------------------------------- #


def test_routing_metadata_is_returned_and_authority_is_untouched():
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})
    request = plan_request(step_no=7, attempt=4, risk="LOW")

    result = router.plan(request)

    assert result.provider == ID_OPENAI
    assert result.model == "openai-model"
    assert result.provider_attempt == 1
    assert result.route_chain == (ID_DEEPSEEK, ID_OPENAI)
    assert result.fallback_from == ID_DEEPSEEK
    assert result.fallback_reason == CATEGORY_RATE_LIMIT
    assert result.usage is None  # honest: the fake reports no usage
    # Controller-owned identity is echoed, never invented or mutated.
    assert "step_no" not in result.payload
    assert "attempt" not in result.payload
    assert request.step_no == 7
    assert request.attempt == 4

    history = router.route_history[-1]
    assert history["step_no"] == 7
    assert history["attempt"] == 4
    assert history["provider_attempt"] == 1
    assert history["route_chain"] == [ID_DEEPSEEK, ID_OPENAI]
    assert history["fallback_from"] == ID_DEEPSEEK
    assert history["success"] is True
    # The M4 in-memory trace keeps its exact, backward-compatible shape.
    assert [call["provider"] for call in router.calls] == [ID_DEEPSEEK, ID_OPENAI]
    assert set(router.calls[0]) == {
        "operation",
        "provider",
        "model",
        "step_no",
        "attempt",
        "risk",
        "fallback_from",
    }


def test_provider_identity_is_rejected_not_adopted():
    from supervisor import validate_plan_payload
    from supervisor.schemas import controller_owned_identity_problems

    polluted = dict(fake_plan_payload(), step_no=9, attempt=5)
    assert controller_owned_identity_problems(polluted)
    assert validate_plan_payload(polluted)

    # A clean provider payload stays clean: the router adds routing metadata to
    # the RESULT only, never identity to the payload.
    deepseek = healthy(ID_DEEPSEEK)
    router = chain_router({ID_DEEPSEEK: deepseek})
    result = router.plan(plan_request(step_no=9, attempt=5, risk="LOW"))

    assert result.payload == fake_plan_payload()
    assert validate_plan_payload(result.payload) == []


def test_audit_never_contains_a_secret():
    recorder = Recorder()
    deepseek = failing(
        ID_DEEPSEEK,
        CATEGORY_RATE_LIMIT,
        message=f"upstream 429 for Authorization: Bearer {SECRET_KEY}",
    )
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai}, recorder)

    result = router.plan(plan_request(risk="LOW"))

    blob = json.dumps(recorder.events, default=str)
    assert SECRET_KEY not in blob
    assert "***" in blob  # redacted, not simply dropped
    assert SECRET_KEY not in json.dumps(router.route_history, default=str)
    assert SECRET_KEY not in json.dumps(router.calls, default=str)
    assert SECRET_KEY not in str(result)
    assert result.provider == ID_OPENAI


def test_router_module_holds_no_state_and_no_credentials():
    import supervisor.router as router_module

    source = pathlib.Path(router_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("import sqlite3", "API_KEY", "os.environ", "getenv", "open("):
        assert forbidden not in source


# --------------------------------------------------------------------------- #
# 49-53  Controller integration: provider attempts are not project attempts
# --------------------------------------------------------------------------- #

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    """Load the app under a private module name (no cross-test interference)."""
    spec = importlib.util.spec_from_file_location("worker_tandem_m8", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_m8"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()

CONTROLLER_PLAN = {
    "plan_version": "m8-test-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "STEP ONE",
            "description": "m8 integration step one",
            "risk": "LOW",
        },
    ],
}


def report_ok():
    return {
        "step_no": 1,
        "status": "DONE",
        "files_created": [],
        "files_changed": ["a.py"],
        "files_deleted": [],
        "tests": {"passed": 1, "failed": 0, "command": "pytest -q"},
        "dependencies_added": [],
        "architecture_questions": [],
        "issues": [],
        "summary": "done",
    }


@pytest.fixture
def ctrl(tmp_path):
    controller = wt.TandemController(tmp_path)
    controller.db.import_plan(CONTROLLER_PLAN)
    yield controller
    controller.db.close()


def stub_runner(controller, payload):
    """Replace the Codex call with a fixed, already-validated payload."""

    def fake(
        prompt, schema_path, output_path, trace_path, timeout_sec=None, reasoning_effort=None
    ):
        return dict(payload)

    controller.codex.run_structured = fake


def install_auto_router(controller, adapters, **policy_kwargs):
    """Swap the controller's router for an AUTO, multi-provider one.

    Uses the controller's own audit hook, so every routing event is stored in
    the existing ``events`` table (no schema change).
    """
    controller.supervisor_policy = default_auto_policy(**policy_kwargs)
    controller.supervisor_router = SupervisorRouter(
        {DEFAULT_LEGACY_PROVIDER: controller.supervisor_adapter, **adapters},
        controller.supervisor_policy,
        emit=controller._supervisor_emit,
    )
    return controller.supervisor_router


def controller_events(controller, event_type, step_no=1):
    rows = controller.db.db.execute(
        "SELECT details FROM events WHERE step_no=? AND event_type=? ORDER BY id",
        (step_no, event_type),
    ).fetchall()
    return [json.loads(row["details"]) for row in rows]


def to_cline_dispatched(controller, *, attempt=None):
    controller.db.transition(1, "CODEX_PLAN_RUNNING")
    if attempt is not None:
        while controller.db.get_step(1).attempt < attempt:
            controller.db.increment_attempt(1)
    controller.db.transition(1, "PLAN_READY")
    controller.db.transition(1, "WAITING_HUMAN_APPROVAL")
    controller.db.transition(1, "CLINE_DISPATCHED")


def accept_report(controller):
    wt.write_json_atomic(
        controller.expected_cline_report_path(1), report_ok()
    )
    return controller.process_cline_report()


def test_controller_default_policy_is_still_manual_codex_legacy(ctrl):
    policy = ctrl.supervisor_policy
    assert policy.mode == POLICY_MODE_MANUAL
    assert policy.is_auto is False
    assert policy.plan_provider == PROVIDER_CODEX_LEGACY
    assert policy.review_provider == PROVIDER_CODEX_LEGACY
    assert policy.fallback_chain == ()
    assert policy.max_provider_retries == 0
    assert ctrl.supervisor_router.registered_providers() == (PROVIDER_CODEX_LEGACY,)
    assert ctrl.supervisor_router.calls == []
    # AUTO exists as a capability, but nothing selects it by default.
    assert default_auto_policy().mode == POLICY_MODE_AUTO
    assert ctrl.db.get_step(1) is not None  # existing DB still opens


def test_controller_honours_an_auto_policy_for_plan(ctrl):
    stub_runner(ctrl, fake_plan_payload())
    deepseek, openai = healthy(ID_DEEPSEEK), healthy(ID_OPENAI)
    router = install_auto_router(ctrl, {ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    plan = ctrl.run_codex_plan()

    # The controller still owns identity; the provider only supplies the payload.
    assert plan == {**fake_plan_payload(), "step_no": 1}
    assert router.calls == [
        {
            "operation": PLAN_OPERATION,
            "provider": ID_DEEPSEEK,
            "model": "deepseek-model",
            "step_no": 1,
            "attempt": 1,
            "risk": "LOW",
            "fallback_from": None,
        }
    ]
    assert [call[0] for call in openai.calls] == []
    step = ctrl.db.get_step(1)
    assert step.state == "PLAN_READY"
    assert step.attempt == 1  # the pre-existing PLAN increment, done once


def test_provider_retry_and_fallback_do_not_consume_a_cline_attempt(ctrl):
    stub_runner(ctrl, fake_plan_payload())
    deepseek = failing(
        ID_DEEPSEEK, CATEGORY_RATE_LIMIT, message=f"429 rate limited: {SECRET_KEY}"
    )
    openai = healthy(ID_OPENAI)
    router = install_auto_router(
        ctrl,
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        fallback_chain=(ID_DEEPSEEK, ID_OPENAI),
        allow_codex_legacy=False,
    )

    plan = ctrl.run_codex_plan()

    assert plan == {**fake_plan_payload(), "step_no": 1}
    assert [call["provider"] for call in router.calls] == [ID_DEEPSEEK, ID_OPENAI]
    step = ctrl.db.get_step(1)
    assert step.state == "PLAN_READY"
    # Three provider attempts, but exactly one project attempt.
    assert step.attempt == 1
    assert len(deepseek.calls) == 2  # one provider retry
    assert len(openai.calls) == 1  # one fallback

    # The routing audit is stored in the existing events table, sanitized.
    fallbacks = controller_events(ctrl, SUPERVISOR_FALLBACK_EVENT)
    assert len(fallbacks) == 1
    assert fallbacks[0]["fallback_from"] == ID_DEEPSEEK
    assert fallbacks[0]["fallback_to"] == ID_OPENAI
    assert fallbacks[0]["category"] == CATEGORY_RATE_LIMIT
    assert fallbacks[0]["retryable"] is True
    error_blob = json.dumps(controller_events(ctrl, SUPERVISOR_ERROR_EVENT))
    assert SECRET_KEY not in error_blob
    assert "***" in error_blob


def test_all_providers_failing_defers_review_and_keeps_the_report(ctrl):
    stub_runner(ctrl, fake_review_payload())
    to_cline_dispatched(ctrl, attempt=1)
    accept_report(ctrl)
    assert ctrl.db.get_step(1).state == "CLINE_REPORT_RECEIVED"
    accepted = ctrl.db.cline_report_for_attempt(1, 1)
    assert accepted is not None

    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT, operation=REVIEW_OPERATION)
    openai = failing(ID_OPENAI, CATEGORY_TIMEOUT, operation=REVIEW_OPERATION)
    router = install_auto_router(
        ctrl,
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        fallback_chain=(ID_DEEPSEEK, ID_OPENAI),
        allow_codex_legacy=False,
    )

    result = ctrl.run_supervisor_review()

    # M8 §9: every provider failed for retryable external reasons, so the
    # existing deferred REVIEW state is reused (no FSM rename, no schema change).
    assert result["deferred"] is True
    assert result["category"] == CATEGORY_TIMEOUT
    step = ctrl.db.get_step(1)
    assert step.state == "CODEX_REVIEW_RETRYABLE"
    assert step.attempt == 1  # the project attempt is untouched
    assert ctrl.db.has_codex_review(1, 1) is False  # nothing was persisted
    assert ctrl.db.cline_report_for_attempt(1, 1) == accepted  # report preserved
    assert len(deepseek.calls) == 2
    assert len(openai.calls) == 2
    assert [call["provider"] for call in router.calls] == [
        ID_DEEPSEEK,
        ID_OPENAI,
    ]
    all_failed = controller_events(ctrl, SUPERVISOR_ALL_FAILED_EVENT)
    assert all_failed and all_failed[0]["retryable"] is True
    assert controller_events(ctrl, "CODEX_REVIEW_RETRYABLE")


def test_all_providers_failing_during_plan_keeps_safe_failure_semantics(ctrl):
    stub_runner(ctrl, fake_plan_payload())
    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = failing(ID_OPENAI, CATEGORY_TIMEOUT)
    router = install_auto_router(
        ctrl,
        {ID_DEEPSEEK: deepseek, ID_OPENAI: openai},
        fallback_chain=(ID_DEEPSEEK, ID_OPENAI),
        allow_codex_legacy=False,
    )

    with pytest.raises(wt.SupervisorRunError) as info:
        ctrl.run_codex_plan()

    # M8 §9: PLAN keeps its safe failure semantics; provider retries never
    # fabricate extra project attempts or extra Cline runs.
    assert info.value.retryable is True
    assert info.value.category == CATEGORY_TIMEOUT
    step = ctrl.db.get_step(1)
    assert step.state == "FAILED"
    assert step.attempt == 1  # exactly the one pre-existing PLAN increment
    assert ctrl.db.latest_codex_plan(1) is None  # nothing was published
    assert [call["provider"] for call in router.calls] == [ID_DEEPSEEK, ID_OPENAI]
    assert len(deepseek.calls) == 2
    assert len(openai.calls) == 2
    assert controller_events(ctrl, SUPERVISOR_ALL_FAILED_EVENT)[0]["retryable"] is True


def test_auto_routing_needs_no_network_and_no_process(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("AUTO routing must stay offline in tests")

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)

    deepseek = failing(ID_DEEPSEEK, CATEGORY_RATE_LIMIT)
    openai = healthy(ID_OPENAI)
    router = chain_router({ID_DEEPSEEK: deepseek, ID_OPENAI: openai})

    assert router.plan(plan_request(risk="LOW")).provider == ID_OPENAI
    # The PLAN failure was deepseek-only; REVIEW uses deepseek's success path.
    assert router.review(review_request(risk="LOW")).provider == ID_DEEPSEEK
    assert router.health()[ID_OPENAI].status == HEALTH_READY




