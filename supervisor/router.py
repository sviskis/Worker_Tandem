"""Provider-neutral supervisor router (Milestones M4 + M8).

Selects the provider(s) for an operation and executes PLAN/REVIEW through them.
The router is advisory only: it never touches the FSM, SQLite or project files,
and it never invents provider identity.

* ``MANUAL`` (the default, unchanged M4 behaviour): exactly the configured
  provider - no retry, no fallback, no health skipping.
* ``AUTO`` (opt-in through :func:`default_auto_policy`): the risk route picks the
  PLAN provider and the REVIEW provider from ONE :class:`SupervisorPolicy`, and
  a de-duplicated fallback chain is walked with at most one same-provider retry
  per provider. Retryable provider outages fall back; auth/billing/model/request
  failures fail closed; ``unknown`` fails closed unless the policy allows it.

Provider attempts are NOT project attempts: the router never increments the
Cline attempt, never re-runs PLAN during REVIEW and never mutates an accepted
report. Health-aware skipping in AUTO uses a bounded-lifetime in-memory snapshot
that is never persisted.

Audit is emitted through an injected ``emit(event_type, payload)`` callback, so
the router stays free of ``sqlite3``/``tkinter`` and remains unit-testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from .errors import (
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_SCHEMA,
    CATEGORY_TEMPORARY_5XX,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    CATEGORY_UNKNOWN,
    SupervisorRunError,
    sanitize_diagnostic,
)
from .models import (
    HEALTH_AUTH_ERROR,
    HEALTH_KEY_MISSING,
    HEALTH_PROVIDER_ERROR,
    PLAN_OPERATION,
    REVIEW_OPERATION,
    ProviderHealth,
    SupervisorPlanRequest,
    SupervisorPlanResult,
    SupervisorReviewRequest,
    SupervisorReviewResult,
)
from .schemas import payload_hash

# --------------------------------------------------------------------------- #
# Audit event types (written through the existing ``events`` table)
# --------------------------------------------------------------------------- #

SUPERVISOR_SELECTED_EVENT = "SUPERVISOR_SELECTED"
SUPERVISOR_RETRY_EVENT = "SUPERVISOR_RETRY"
SUPERVISOR_FALLBACK_EVENT = "SUPERVISOR_FALLBACK"
SUPERVISOR_RESULT_EVENT = "SUPERVISOR_RESULT"
SUPERVISOR_ERROR_EVENT = "SUPERVISOR_ERROR"
SUPERVISOR_ALL_FAILED_EVENT = "SUPERVISOR_ALL_FAILED"

# --------------------------------------------------------------------------- #
# Modes, provider ids, budgets
# --------------------------------------------------------------------------- #

POLICY_MODE_MANUAL = "MANUAL"
POLICY_MODE_AUTO = "AUTO"
POLICY_MODES = (POLICY_MODE_MANUAL, POLICY_MODE_AUTO)

#: Provider ids referenced by the DEFAULT policy. Duplicated as literals so the
#: router never imports an adapter; a test asserts they match the adapters.
ID_DEEPSEEK = "deepseek"
ID_OPENAI = "openai"
ID_ANTHROPIC = "anthropic"

#: M4 default provider. Kept in sync with
#: ``supervisor.codex_legacy_adapter.PROVIDER_CODEX_LEGACY`` (asserted by test)
#: without importing an adapter from the router.
DEFAULT_LEGACY_PROVIDER = "codex_legacy"

BUDGET_CHEAP = "CHEAP"
BUDGET_STANDARD = "STANDARD"
BUDGET_PREMIUM = "PREMIUM"
BUDGET_MODES = (BUDGET_CHEAP, BUDGET_STANDARD, BUDGET_PREMIUM)

#: Preferred ordering per budget mode (M8 §12 - no currency accounting yet).
#: Applied to the FALLBACK TAIL only: the risk route always keeps slot 0.
BUDGET_ORDERS: Mapping[str, tuple[str, ...]] = {
    BUDGET_CHEAP: (ID_DEEPSEEK, ID_OPENAI, ID_ANTHROPIC, DEFAULT_LEGACY_PROVIDER),
    BUDGET_STANDARD: (ID_DEEPSEEK, ID_OPENAI, ID_ANTHROPIC, DEFAULT_LEGACY_PROVIDER),
    BUDGET_PREMIUM: (ID_ANTHROPIC, ID_OPENAI, ID_DEEPSEEK, DEFAULT_LEGACY_PROVIDER),
}

#: Failures that justify a same-provider retry and then a provider fallback.
RETRYABLE_FALLBACK_CATEGORIES = frozenset(
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

#: Malformed / schema-invalid provider output: retry once, then fall back.
CONTRACT_FAILURE_CATEGORIES = frozenset({CATEGORY_CONTRACT, CATEGORY_SCHEMA})

#: "The provider (not the project) let us down" - used to decide whether an
#: exhausted chain is deferrable instead of a project failure.
PROVIDER_SIDE_CATEGORIES = RETRYABLE_FALLBACK_CATEGORIES | CONTRACT_FAILURE_CATEGORIES

#: Health statuses that make AUTO skip a provider (never in MANUAL).
SKIP_HEALTH_STATUSES = frozenset({HEALTH_KEY_MISSING, HEALTH_AUTH_ERROR})

#: How long a health snapshot stays valid (M8 §10: never probe per request).
DEFAULT_HEALTH_TTL_SEC = 60.0


def default_risk_routes() -> dict[str, dict[str, tuple[str, ...]]]:
    """M8 §2 default AUTO routes: LOW/MEDIUM -> deepseek, HIGH -> openai/anthropic."""
    return {
        "LOW": {PLAN_OPERATION: (ID_DEEPSEEK,), REVIEW_OPERATION: (ID_DEEPSEEK,)},
        "MEDIUM": {PLAN_OPERATION: (ID_DEEPSEEK,), REVIEW_OPERATION: (ID_DEEPSEEK,)},
        "HIGH": {PLAN_OPERATION: (ID_OPENAI,), REVIEW_OPERATION: (ID_ANTHROPIC,)},
    }


def default_fallback_chain() -> tuple[str, ...]:
    """M8 §2 example chain: deepseek -> openai -> anthropic -> codex_legacy."""
    return (ID_DEEPSEEK, ID_OPENAI, ID_ANTHROPIC, DEFAULT_LEGACY_PROVIDER)


def _dedupe_keep_order(items) -> tuple[str, ...]:
    """De-duplicate while preserving first-seen order (no provider loops)."""
    seen: list[str] = []
    for item in items:
        value = str(item or "")
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _order_by_budget(providers, budget_mode: str) -> tuple[str, ...]:
    """Order the fallback tail by the budget preference (stable for unknowns)."""
    order = BUDGET_ORDERS.get(str(budget_mode).upper(), ())
    rank = {provider: index for index, provider in enumerate(order)}
    return tuple(
        sorted(
            (p for p in providers),
            key=lambda p: rank.get(p, len(rank)),
        )
    )


def _normalise_risk_routes(routes) -> dict[str, dict[str, tuple[str, ...]]]:
    """Accept ``{risk: {op: [providers]}}`` from JSON and store tuples."""
    if not isinstance(routes, Mapping):
        return default_risk_routes()
    normalised: dict[str, dict[str, tuple[str, ...]]] = {}
    for risk, per_op in routes.items():
        if not isinstance(per_op, Mapping):
            continue
        normalised[str(risk).upper()] = {
            str(operation).upper(): tuple(str(p) for p in providers or ())
            for operation, providers in per_op.items()
        }
    return normalised or default_risk_routes()



@dataclass(frozen=True)
class SupervisorPolicy:
    """Serializable provider-selection policy (MANUAL default, AUTO opt-in).

    ``SupervisorPolicy()`` keeps the M4 behaviour exactly: MANUAL selection of
    ``codex_legacy`` with an empty fallback chain and no retries, so no existing
    project changes behaviour until a policy explicitly selects AUTO (M8 §14).
    :func:`default_auto_policy` builds the M8 §2 AUTO policy.
    """

    mode: str = POLICY_MODE_MANUAL
    plan_provider: str = DEFAULT_LEGACY_PROVIDER
    review_provider: str = DEFAULT_LEGACY_PROVIDER
    fallback_chain: tuple[str, ...] = ()
    max_provider_retries: int = 0
    # --- M8 routing policy ------------------------------------------------ #
    risk_routes: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=default_risk_routes
    )
    budget_mode: str = BUDGET_STANDARD
    #: Legacy Codex is only ever a FALLBACK candidate, and only when allowed.
    allow_codex_legacy: bool = False
    #: MANUAL must never fall back unless the operator says so (M8 §11).
    allow_manual_fallback: bool = False
    #: One retry (+ fallback) after malformed/schema-invalid output (M8 §4).
    fallback_on_contract_failure: bool = True
    #: ``unknown`` fails closed unless explicitly allowed (M8 §5).
    allow_unknown_fallback: bool = False
    #: AUTO may skip providers known to be KEY_MISSING/AUTH_ERROR (M8 §10).
    skip_unhealthy_providers: bool = True

    # -- selection ---------------------------------------------------------- #

    @property
    def is_auto(self) -> bool:
        return str(self.mode).upper() == POLICY_MODE_AUTO

    def provider_for(self, operation: str) -> str:
        """The configured provider for ``PLAN`` or ``REVIEW``.

        PLAN and REVIEW may use different providers (Phase 15); an empty
        REVIEW provider falls back to the PLAN provider.
        """
        if str(operation).upper() == REVIEW_OPERATION:
            return self.review_provider or self.plan_provider
        return self.plan_provider

    def route_for(self, operation: str, risk: str = "") -> tuple[str, ...]:
        """Primary candidates for an operation at a risk level (AUTO only)."""
        if not self.is_auto:
            primary = self.provider_for(operation)
            return (primary,) if primary else ()
        routes = self.risk_routes or {}
        per_op = routes.get(str(risk or "").strip().upper()) or routes.get("MEDIUM")
        chosen = (per_op or {}).get(str(operation).upper()) or ()
        if not chosen:
            chain = self.fallback_chain
            chosen = (chain[0],) if chain else ()
        return tuple(chosen)

    def candidates_for(self, operation: str, risk: str = "") -> tuple[str, ...]:
        """The ordered provider chain the router may try for this call.

        * MANUAL: exactly the selected provider - plus the fallback chain only
          when ``allow_manual_fallback`` is set (M8 §11).
        * AUTO: risk route first, then the fallback chain ordered by budget,
          de-duplicated (no provider loops), with codex_legacy gated by
          ``allow_codex_legacy``.
        """
        if not self.is_auto:
            primary = self.provider_for(operation)
            if not primary:
                return ()
            if not self.allow_manual_fallback:
                return (primary,)
            base, tail = (primary,), self.fallback_chain
        else:
            base = self.route_for(operation, risk)
            if not base:
                return ()
            tail = self.fallback_chain

        ordered = _dedupe_keep_order(tuple(base) + tuple(tail))
        # §12: the route primary keeps slot 0; the budget orders the tail.
        ordered = ordered[:1] + _order_by_budget(ordered[1:], self.budget_mode)
        if self.is_auto and not self.allow_codex_legacy:
            ordered = tuple(p for p in ordered if p != DEFAULT_LEGACY_PROVIDER)
        return ordered

    # -- serialization ------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "plan_provider": self.plan_provider,
            "review_provider": self.review_provider,
            "fallback_chain": list(self.fallback_chain),
            "max_provider_retries": self.max_provider_retries,
            "risk_routes": {
                risk: {op: list(providers) for op, providers in per_op.items()}
                for risk, per_op in (self.risk_routes or {}).items()
            },
            "budget_mode": self.budget_mode,
            "allow_codex_legacy": self.allow_codex_legacy,
            "allow_manual_fallback": self.allow_manual_fallback,
            "fallback_on_contract_failure": self.fallback_on_contract_failure,
            "allow_unknown_fallback": self.allow_unknown_fallback,
            "skip_unhealthy_providers": self.skip_unhealthy_providers,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SupervisorPolicy":
        data = data or {}
        return cls(
            mode=str(data.get("mode") or POLICY_MODE_MANUAL),
            plan_provider=str(data.get("plan_provider") or DEFAULT_LEGACY_PROVIDER),
            review_provider=str(
                data.get("review_provider") or DEFAULT_LEGACY_PROVIDER
            ),
            fallback_chain=tuple(data.get("fallback_chain") or ()),
            max_provider_retries=int(data.get("max_provider_retries") or 0),
            risk_routes=_normalise_risk_routes(data.get("risk_routes")),
            budget_mode=str(data.get("budget_mode") or BUDGET_STANDARD),
            allow_codex_legacy=bool(data.get("allow_codex_legacy", False)),
            allow_manual_fallback=bool(data.get("allow_manual_fallback", False)),
            fallback_on_contract_failure=bool(
                data.get("fallback_on_contract_failure", True)
            ),
            allow_unknown_fallback=bool(data.get("allow_unknown_fallback", False)),
            skip_unhealthy_providers=bool(
                data.get("skip_unhealthy_providers", True)
            ),
        )


def default_auto_policy(**overrides: Any) -> SupervisorPolicy:
    """The M8 §2 default AUTO policy (opt-in; MANUAL stays the default)."""
    values: dict[str, Any] = {
        "mode": POLICY_MODE_AUTO,
        "plan_provider": ID_DEEPSEEK,
        "review_provider": ID_DEEPSEEK,
        "fallback_chain": default_fallback_chain(),
        "max_provider_retries": 1,
        "risk_routes": default_risk_routes(),
        "budget_mode": BUDGET_STANDARD,
        "allow_codex_legacy": True,
        "allow_manual_fallback": False,
        "fallback_on_contract_failure": True,
        "allow_unknown_fallback": False,
        "skip_unhealthy_providers": True,
    }
    values.update(overrides)
    return SupervisorPolicy(**values)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

#: ``emit(event_type, payload)`` - the audit hook supplied by the controller.
EmitHook = Callable[[str, dict], None]


class SupervisorRouter:
    """Chooses a provider and executes PLAN/REVIEW through it.

    The router is provider-neutral: it knows nothing about HTTP, Codex or
    prompts. It never mutates the FSM and never opens SQLite.
    """

    def __init__(
        self,
        adapters: dict[str, Any],
        policy: Optional[SupervisorPolicy] = None,
        emit: Optional[EmitHook] = None,
        *,
        health_ttl_sec: float = DEFAULT_HEALTH_TTL_SEC,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._adapters = dict(adapters or {})
        self.policy = policy or SupervisorPolicy()
        self._emit = emit
        #: In-memory trace of every provider attempt (M4-compatible shape).
        self.calls: list[dict] = []
        #: Richer per-attempt routing metadata (M8 §7) - never persisted.
        self.route_history: list[dict] = []
        self._health_ttl = max(0.0, float(health_ttl_sec))
        self._monotonic = clock or time.monotonic
        #: provider_id -> (monotonic_timestamp, health status); never persisted.
        self._health_cache: dict[str, tuple[float, str]] = {}

    # -- introspection ----------------------------------------------------- #

    def registered_providers(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def adapter_for(self, provider_id: str):
        return self._adapters.get(provider_id)

    def selected_provider(self, operation: str) -> str:
        return self.policy.provider_for(operation)

    def summary(self) -> dict:
        """Small, read-only status dict for the GUI/CLI."""
        return {
            "mode": self.policy.mode,
            "plan_provider": self.policy.plan_provider,
            "review_provider": self.policy.review_provider,
            "registered": list(self._adapters),
            "budget_mode": self.policy.budget_mode,
            "allow_codex_legacy": self.policy.allow_codex_legacy,
            "skip_unhealthy_providers": self.policy.skip_unhealthy_providers,
        }

    def health(self) -> dict[str, ProviderHealth]:
        """Cheap health of every REGISTERED provider (never a big request)."""
        result: dict[str, ProviderHealth] = {}
        for provider_id, adapter in self._adapters.items():
            try:
                result[provider_id] = adapter.health_check()
            except Exception as exc:
                result[provider_id] = ProviderHealth(
                    provider_id=provider_id,
                    status=HEALTH_PROVIDER_ERROR,
                    detail=sanitize_diagnostic(exc),
                )
        return result

    # -- execution --------------------------------------------------------- #

    def plan(
        self,
        request: SupervisorPlanRequest,
        *,
        risk: Optional[str] = None,
        requires_human: Optional[bool] = None,
    ) -> SupervisorPlanResult:
        return self._run(
            PLAN_OPERATION, request, risk=risk, requires_human=requires_human
        )

    def review(
        self,
        request: SupervisorReviewRequest,
        *,
        risk: Optional[str] = None,
        requires_human: Optional[bool] = None,
    ) -> SupervisorReviewResult:
        return self._run(
            REVIEW_OPERATION, request, risk=risk, requires_human=requires_human
        )

    # -- selection --------------------------------------------------------- #

    @staticmethod
    def _fail_closed(operation: str, message: str) -> SupervisorRunError:
        """A configuration/selection problem: always non-retryable."""
        return SupervisorRunError(
            message,
            operation=operation,
            category=CATEGORY_INVALID_REQUEST,
            retryable=False,
            sanitized_message=sanitize_diagnostic(message),
        )

    # -- candidate selection + health (M8) --------------------------------- #

    def _health_state(self, provider_id: str) -> Optional[str]:
        """Cached health status for one provider.

        M8 §10: the router must not probe health endpoints before every request,
        so a bounded-lifetime in-memory snapshot is used. Nothing is persisted
        and no secret ever enters the cache.
        """
        now = self._monotonic()
        entry = self._health_cache.get(provider_id)
        if entry is not None and (now - entry[0]) <= self._health_ttl:
            return entry[1]

        adapter = self._adapters.get(provider_id)
        if adapter is None:
            return None
        try:
            status = str(getattr(adapter.health_check(), "status", "") or "")
        except Exception:
            status = HEALTH_PROVIDER_ERROR
        self._health_cache[provider_id] = (now, status)
        return status

    def forget_health(self, provider_id: Optional[str] = None) -> None:
        """Drop the cached health snapshot (all providers, or just one)."""
        if provider_id is None:
            self._health_cache.clear()
        else:
            self._health_cache.pop(provider_id, None)

    def _usable(self, provider_id: str, adapter, *, allow_health_skip: bool):
        """``(usable, reason)`` - never raises; the caller decides fail-closed."""
        if adapter is None:
            return False, "not-registered"
        config = getattr(adapter, "config", None)
        if config is not None and not bool(getattr(config, "enabled", True)):
            return False, "disabled"
        if allow_health_skip:
            status = self._health_state(provider_id)
            if status in SKIP_HEALTH_STATUSES:
                return False, str(status)
        return True, ""

    # -- audit ------------------------------------------------------------- #

    def _emit_event(self, event_type: str, payload: dict) -> None:
        if self._emit is None:
            return
        try:
            self._emit(event_type, payload)
        except Exception:
            # Audit is best-effort: it must never abort a supervisor call.
            pass

    # -- the execution path (M8: same-provider retry + provider fallback) --- #

    def _action_for(self, exc: SupervisorRunError) -> str:
        """``retry`` | ``fallback`` | ``stop`` after a provider failure.

        M8 §4/§5: retryable provider outages get one same-provider retry and then
        a fallback; malformed/schema-invalid output does the same when the policy
        allows it; auth/billing/model/request failures and ``unknown`` stop the
        chain and fail closed.
        """
        if exc.category in RETRYABLE_FALLBACK_CATEGORIES:
            return "retry"
        if (
            exc.category in CONTRACT_FAILURE_CATEGORIES
            and self.policy.fallback_on_contract_failure
        ):
            return "retry"
        if exc.category == CATEGORY_UNKNOWN and self.policy.allow_unknown_fallback:
            return "fallback"
        return "stop"

    def _run(
        self,
        operation: str,
        request: Any,
        *,
        risk: Optional[str] = None,
        requires_human: Optional[bool] = None,
    ):
        risk_value = risk if risk is not None else getattr(request, "risk", "")
        human_value = (
            bool(requires_human)
            if requires_human is not None
            else bool(getattr(request, "requires_human", False))
        )
        step_no = getattr(request, "step_no", None)
        attempt = getattr(request, "attempt", None)

        candidates = self.policy.candidates_for(operation, risk_value)
        if not candidates:
            raise self._fail_closed(
                operation, f"no supervisor provider configured for {operation}"
            )

        max_retries = max(0, int(self.policy.max_provider_retries))
        skip_health = bool(self.policy.skip_unhealthy_providers) and self.policy.is_auto
        last_index = len(candidates) - 1
        failures: list[tuple[str, str]] = []
        last_error: Optional[SupervisorRunError] = None
        fallback_from: Optional[str] = None
        fallback_reason: Optional[str] = None

        for index, provider_id in enumerate(candidates):
            record = {
                "operation": operation,
                "provider": provider_id,
                "model": None,
                "step_no": step_no,
                "attempt": attempt,
                "risk": risk_value,
                "fallback_from": fallback_from,
            }
            self.calls.append(record)

            adapter = self._adapters.get(provider_id)
            # M8 §10: skip a KNOWN-unhealthy provider only while a later
            # candidate remains, so the last provider always gets a chance.
            usable, reason = self._usable(
                provider_id,
                adapter,
                allow_health_skip=skip_health and index < last_index,
            )
            if not usable:
                failures.append((provider_id, f"skipped:{reason}"))
                fallback_from, fallback_reason = provider_id, f"skipped:{reason}"
                continue

            model = getattr(adapter, "model", None)
            record["model"] = model

            for provider_attempt in range(1, max_retries + 2):
                self._emit_event(
                    SUPERVISOR_SELECTED_EVENT,
                    {
                        "operation": operation,
                        "provider": provider_id,
                        "model": model,
                        "step_no": step_no,
                        "attempt": attempt,
                        "risk": risk_value,
                        "requires_human": human_value,
                        "mode": self.policy.mode,
                        "provider_attempt": provider_attempt,
                        "route_chain": list(candidates),
                        "fallback_from": fallback_from,
                        "fallback_reason": fallback_reason,
                    },
                )
                if provider_attempt > 1:
                    self._emit_event(
                        SUPERVISOR_RETRY_EVENT,
                        {
                            "operation": operation,
                            "provider": provider_id,
                            "model": model,
                            "step_no": step_no,
                            "attempt": attempt,
                            "risk": risk_value,
                            "provider_attempt": provider_attempt,
                            "category": getattr(last_error, "category", None),
                            "retryable": bool(getattr(last_error, "retryable", False)),
                            "sanitized_error": getattr(
                                last_error, "sanitized_message", ""
                            ),
                        },
                    )

                started = self._monotonic()
                try:
                    if operation == PLAN_OPERATION:
                        result = adapter.plan(request)
                    else:
                        result = adapter.review(request)
                except SupervisorRunError as exc:
                    last_error = exc
                    failures.append((provider_id, exc.category))
                    self._emit_error(
                        operation,
                        provider_id,
                        model,
                        step_no,
                        attempt,
                        risk_value,
                        exc,
                        provider_attempt=provider_attempt,
                    )
                    action = self._action_for(exc)
                    if action == "retry" and provider_attempt <= max_retries:
                        continue  # same provider once more
                    if action == "stop":
                        raise  # M8 §5: fail closed, never fall back
                    if index < last_index:
                        fallback_from, fallback_reason = provider_id, exc.category
                        self._emit_event(
                            SUPERVISOR_FALLBACK_EVENT,
                            {
                                "operation": operation,
                                "provider": provider_id,
                                "model": model,
                                "step_no": step_no,
                                "attempt": attempt,
                                "risk": risk_value,
                                "provider_attempt": provider_attempt,
                                "category": exc.category,
                                "retryable": bool(exc.retryable),
                                "fallback_from": provider_id,
                                "fallback_to": candidates[index + 1],
                                "sanitized_error": exc.sanitized_message,
                            },
                        )
                        break  # next provider
                    if len(candidates) > 1:
                        # The whole chain failed: report the route as one neutral
                        # error and audit SUPERVISOR_ALL_FAILED.
                        break
                    # Single-provider route (MANUAL): keep the provider's own
                    # error so the M4 controller contract is unchanged.
                    raise
                except Exception as exc:
                    # Defensive: a broken adapter must still surface a neutral
                    # error, and an unclassifiable failure fails closed.
                    wrapped = SupervisorRunError(
                        "supervisor provider raised an unexpected error",
                        provider=provider_id,
                        model=model,
                        operation=operation,
                        category=CATEGORY_UNKNOWN,
                        retryable=False,
                        sanitized_message=sanitize_diagnostic(exc),
                    )
                    last_error = wrapped
                    failures.append((provider_id, CATEGORY_UNKNOWN))
                    self._emit_error(
                        operation,
                        provider_id,
                        model,
                        step_no,
                        attempt,
                        risk_value,
                        wrapped,
                        provider_attempt=provider_attempt,
                    )
                    raise wrapped from exc

                latency_ms = max(0, int(round((self._monotonic() - started) * 1000)))
                payload = getattr(result, "payload", None) or {}
                usage = getattr(result, "usage", None)
                resolved_provider = getattr(result, "provider", None) or provider_id
                resolved_model = getattr(result, "model", None) or model
                result_hash = payload_hash(payload)
                usage_payload = self._usage_dict(usage)
                self._emit_event(
                    SUPERVISOR_RESULT_EVENT,
                    {
                        "operation": operation,
                        "provider": resolved_provider,
                        "model": resolved_model,
                        "step_no": step_no,
                        "attempt": attempt,
                        "risk": risk_value,
                        "provider_attempt": provider_attempt,
                        "result_hash": result_hash,
                        "latency_ms": latency_ms,
                        "usage": usage_payload,
                        "fallback_from": fallback_from,
                        "fallback_reason": fallback_reason,
                        "route_chain": list(candidates),
                    },
                )
                self.route_history.append(
                    {
                        "operation": operation,
                        "provider": resolved_provider,
                        "model": resolved_model,
                        "step_no": step_no,
                        "attempt": attempt,
                        "risk": risk_value,
                        "provider_attempt": provider_attempt,
                        "route_chain": list(candidates),
                        "fallback_from": fallback_from,
                        "fallback_reason": fallback_reason,
                        "latency_ms": latency_ms,
                        "usage": usage_payload,
                        "result_hash": result_hash,
                        "success": True,
                    }
                )
                return self._decorate(
                    result,
                    provider_attempt=provider_attempt,
                    route_chain=candidates,
                    fallback_from=fallback_from,
                    fallback_reason=fallback_reason,
                )

        # Every candidate was skipped or failed.
        real = [cat for _, cat in failures if not cat.startswith("skipped:")]
        retryable = bool(real) and all(cat in PROVIDER_SIDE_CATEGORIES for cat in real)
        self.route_history.append(
            {
                "operation": operation,
                "provider": "",
                "model": None,
                "step_no": step_no,
                "attempt": attempt,
                "risk": risk_value,
                "provider_attempt": 0,
                "route_chain": list(candidates),
                "fallback_from": fallback_from,
                "fallback_reason": fallback_reason,
                "latency_ms": None,
                "usage": None,
                "result_hash": None,
                "success": False,
                "failures": [f"{p}={c}" for p, c in failures],
                "retryable": retryable,
            }
        )
        self._emit_event(
            SUPERVISOR_ALL_FAILED_EVENT,
            {
                "operation": operation,
                "step_no": step_no,
                "attempt": attempt,
                "risk": risk_value,
                "route_chain": list(candidates),
                "providers": [p for p, _ in failures],
                "categories": [c for _, c in failures],
                "retryable": retryable,
                "fallback_from": fallback_from,
                "sanitized_error": "; ".join(f"{p}={c}" for p, c in failures),
            },
        )
        raise self._aggregate_failure(
            operation, candidates, failures, retryable, last_error
        )

    def _aggregate_failure(
        self,
        operation: str,
        candidates,
        failures,
        retryable: bool,
        last_error: Optional[SupervisorRunError],
    ) -> SupervisorRunError:
        """One neutral error describing why the whole route was exhausted."""
        if not any(not cat.startswith("skipped:") for _, cat in failures):
            reasons = ", ".join(f"{p}={c}" for p, c in failures) or "none"
            return self._fail_closed(
                operation,
                f"no usable supervisor provider for {operation} ({reasons})",
            )
        summary = "; ".join(f"{p}={c}" for p, c in failures)
        return SupervisorRunError(
            f"all supervisor providers failed for {operation}",
            provider=getattr(last_error, "provider", "") or "",
            model=getattr(last_error, "model", None),
            operation=operation,
            category=getattr(last_error, "category", CATEGORY_UNKNOWN),
            retryable=retryable,
            http_status=getattr(last_error, "http_status", None),
            provider_error_code=getattr(last_error, "provider_error_code", None),
            sanitized_message=sanitize_diagnostic(
                f"all {len(candidates)} provider(s) failed for {operation}: {summary}"
            ),
            usage=getattr(last_error, "usage", None),
        )

    @staticmethod
    def _decorate(
        result,
        *,
        provider_attempt: int,
        route_chain,
        fallback_from: Optional[str],
        fallback_reason: Optional[str],
    ):
        """Attach M8 §7 routing metadata to the provider-neutral result."""
        common = {
            "payload": result.payload,
            "provider": result.provider,
            "model": result.model,
            "usage": result.usage,
            "fallback_from": fallback_from,
            "fallback_reason": fallback_reason,
            "provider_attempt": provider_attempt,
            "route_chain": tuple(route_chain),
        }
        if isinstance(result, SupervisorPlanResult):
            return SupervisorPlanResult(**common)
        return SupervisorReviewResult(**common)

    @staticmethod
    def _usage_dict(usage) -> Optional[dict]:
        """Audit-safe usage snapshot (never a secret)."""
        if usage is None:
            return None
        return {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "estimated_cost": getattr(usage, "estimated_cost", None),
            "latency_ms": getattr(usage, "latency_ms", None),
        }






    def _emit_error(
        self,
        operation: str,
        provider_id: str,
        model: Optional[str],
        step_no: Any,
        attempt: Any,
        risk: Any,
        exc: SupervisorRunError,
        *,
        provider_attempt: int = 1,
    ) -> None:
        self._emit_event(
            SUPERVISOR_ERROR_EVENT,
            {
                "operation": operation,
                "provider": getattr(exc, "provider", "") or provider_id,
                "model": getattr(exc, "model", None) or model,
                "step_no": step_no,
                "attempt": attempt,
                "risk": risk,
                "provider_attempt": provider_attempt,
                "retryable": bool(getattr(exc, "retryable", False)),
                "category": getattr(exc, "category", CATEGORY_UNKNOWN),
                "sanitized_error": sanitize_diagnostic(
                    getattr(exc, "sanitized_message", "")
                ),
                "result_hash": None,
            },
        )


