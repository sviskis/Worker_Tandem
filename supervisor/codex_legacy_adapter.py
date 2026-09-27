"""Legacy Codex supervisor adapter (Milestone M3).

Wraps the *existing* ``CodexRunner`` behind the provider-neutral
``SupervisorPort`` so the old Codex implementation can satisfy the new
contract.

Hard rules honoured here
------------------------
* The supervisor package NEVER imports ``worker_tandem_v0.1.2.py``. The runner
  is injected and only duck-typed (see :class:`StructuredRunnerPort`), so there
  is no import cycle and the package stays independent.
* ``CodexRunner`` itself is NOT modified. Only a tiny, additive ``available()``
  probe is optionally used by ``health_check()``.
* The adapter is not wired into the controller in M3: Worker Tandem keeps using
  its direct Codex path unchanged.
* No FSM/SQLite/network access. The only file the adapter writes is the
  canonical schema file that ``run_structured`` requires to exist.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

from .base import normalize_plan_payload, normalize_review_payload
from .errors import (
    CATEGORY_AUTH,
    CATEGORY_BILLING_DISABLED,
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_API_KEY,
    CATEGORY_INVALID_MODEL,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_NOT_FOUND,
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
    SupervisorRunError,
    is_retryable_category,
    sanitize_diagnostic,
)
from .models import (
    HEALTH_PROVIDER_ERROR,
    HEALTH_READY,
    PLAN_OPERATION,
    REVIEW_OPERATION,
    ProviderConfig,
    ProviderHealth,
    SupervisorPlanRequest,
    SupervisorPlanResult,
    SupervisorReviewRequest,
    SupervisorReviewResult,
)
from .schemas import supervisor_plan_schema, supervisor_review_schema

#: Stable, provider-neutral identity of the legacy Codex path.
PROVIDER_CODEX_LEGACY = "codex_legacy"

#: Reasoning-effort values understood by the Codex CLI.
CODEX_REASONING_LOW = "low"
CODEX_REASONING_MEDIUM = "medium"
CODEX_REASONING_HIGH = "high"

#: Default schema filenames (identical to the controller's convention).
CODEX_PLAN_SCHEMA_NAME = "codex_plan.schema.json"
CODEX_REVIEW_SCHEMA_NAME = "codex_review.schema.json"

#: How a model-supplied ``step_no`` is treated.
#:
#: ``reject``        - a provider must never claim controller identity (M3 default).
#: ``strip_step_no`` - mirror ``CodexRunner``'s legacy behaviour exactly: the
#:                     model value is removed BEFORE validation (the runner
#:                     already does ``obj.pop("step_no", None)``) and the
#:                     controller injects the authoritative value. Required for
#:                     byte-identical M4 parity; step identity is still never
#:                     taken from the provider.
IDENTITY_REJECT = "reject"
IDENTITY_STRIP_STEP_NO = "strip_step_no"
IDENTITY_POLICIES = (IDENTITY_REJECT, IDENTITY_STRIP_STEP_NO)

#: Where a provider payload is validated against the canonical schema.
#:
#: ``adapter`` - the adapter validates and fails closed (M1-M3 default).
#: ``runner``  - the INJECTED runner already validated against the very same
#:               canonical schema (``CodexRunner.run_structured`` does exactly
#:               that), so the adapter passes the payload through unchanged.
#:               This keeps the legacy controller path byte-identical to its
#:               pre-M4 behaviour instead of double-validating.
VALIDATION_ADAPTER = "adapter"
VALIDATION_RUNNER = "runner"
VALIDATION_MODES = (VALIDATION_ADAPTER, VALIDATION_RUNNER)

# --------------------------------------------------------------------------- #
# Injected, duck-typed runner seam (no import of the Worker Tandem module)
# --------------------------------------------------------------------------- #


@runtime_checkable
class StructuredRunnerPort(Protocol):
    """The minimal surface of ``CodexRunner`` the adapter depends on.

    ``CodexRunner.run_structured`` matches this signature exactly. Any test
    double implementing it can be injected instead.
    """

    def run_structured(
        self,
        prompt: str,
        schema_path: Path,
        output_path: Path,
        trace_path: Path,
        timeout_sec: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> dict:
        ...


#: ``(operation, step_no, attempt) -> CodexArtifactPaths``
PathProvider = Callable[[str, int, int], "CodexArtifactPaths"]

#: ``(risk, requires_human, attempt) -> "low" | "medium" | "high"``
EffortPolicy = Callable[[str, bool, int], str]


@dataclass(frozen=True)
class CodexArtifactPaths:
    """The three paths ``CodexRunner.run_structured`` needs."""

    schema_path: Path
    output_path: Path
    trace_path: Path


def default_codex_paths(workdir: Path) -> PathProvider:
    """The legacy artifact layout already used by the controller.

    ``workdir`` is the ``<project>/.worker_tandem`` directory, matching
    ``CodexRunner.workdir``.
    """
    base = Path(workdir)

    def _paths(operation: str, step_no: int, attempt: int) -> CodexArtifactPaths:
        if str(operation).upper() == REVIEW_OPERATION:
            return CodexArtifactPaths(
                schema_path=base / "context" / CODEX_REVIEW_SCHEMA_NAME,
                output_path=base / "from_codex" / f"step_{step_no:03d}_review.json",
                trace_path=(
                    base / "logs" / f"step_{step_no:03d}_codex_review_trace.json"
                ),
            )
        return CodexArtifactPaths(
            schema_path=base / "context" / CODEX_PLAN_SCHEMA_NAME,
            output_path=base / "from_codex" / f"step_{step_no:03d}_plan.json",
            trace_path=base / "logs" / f"step_{step_no:03d}_codex_plan_trace.json",
        )

    return _paths


def default_effort_policy(risk: str, requires_human: bool, attempt: int) -> str:
    """Behavioural mirror of ``TandemController._codex_reasoning_effort``.

    Default ``low``; ``medium`` for a repeated attempt; ``high`` for a
    HIGH-risk step or one that requires a human. It never relaxes a gate.
    """
    if str(risk or "").upper() == "HIGH" or requires_human:
        return CODEX_REASONING_HIGH
    if int(attempt or 0) > 1:
        return CODEX_REASONING_MEDIUM
    return CODEX_REASONING_LOW


# --------------------------------------------------------------------------- #
# CodexRunError -> SupervisorRunError translation (duck-typed, no import)
# --------------------------------------------------------------------------- #

#: Legacy Codex category -> canonical supervisor category.
CODEX_CATEGORY_MAP = {
    # retryable provider outages
    "quota": CATEGORY_QUOTA_WITH_RESET,
    "quota_with_reset": CATEGORY_QUOTA_WITH_RESET,
    "rate_limit": CATEGORY_RATE_LIMIT,
    "timeout": CATEGORY_TIMEOUT,
    "transport": CATEGORY_TRANSPORT,
    "network": CATEGORY_NETWORK,
    "provider_unavailable": CATEGORY_PROVIDER_UNAVAILABLE,
    "temporary_5xx": CATEGORY_TEMPORARY_5XX,
    # non-retryable (fail closed)
    "auth": CATEGORY_AUTH,
    "invalid_api_key": CATEGORY_INVALID_API_KEY,
    "billing": CATEGORY_BILLING_DISABLED,
    "billing_disabled": CATEGORY_BILLING_DISABLED,
    "payment_required": CATEGORY_PAYMENT_REQUIRED,
    "invalid_model": CATEGORY_INVALID_MODEL,
    "invalid_request": CATEGORY_INVALID_REQUEST,
    "config": CATEGORY_INVALID_REQUEST,
    "not_found": CATEGORY_NOT_FOUND,
    "contract": CATEGORY_CONTRACT,
    "schema": CATEGORY_SCHEMA,
    "unsupported_feature": CATEGORY_UNSUPPORTED_FEATURE,
    "quota_unconfirmed": CATEGORY_UNKNOWN,
    "unknown": CATEGORY_UNKNOWN,
}


@dataclass(frozen=True)
class _CodexErrorView:
    """The parts of a legacy ``CodexRunError`` we translate."""

    category: str
    retryable: bool
    returncode: Optional[int]
    sanitized_error: str
    source: str
    provider: str


def _codex_error_view(exc: BaseException) -> Optional[_CodexErrorView]:
    """Recognize a legacy ``CodexRunError`` by SHAPE, not by import.

    ``worker_tandem_v0.1.2.py`` is not importable as a package module, so the
    adapter must never depend on its class identity.
    """
    category = getattr(exc, "category", None)
    retryable = getattr(exc, "retryable", None)
    if not isinstance(category, str) or not isinstance(retryable, bool):
        return None
    returncode = getattr(exc, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        returncode = None
    return _CodexErrorView(
        category=category,
        retryable=retryable,
        returncode=returncode,
        sanitized_error=str(getattr(exc, "sanitized_error", "") or ""),
        source=str(getattr(exc, "source", "") or ""),
        provider=str(getattr(exc, "provider", "") or ""),
    )


def translate_codex_exception(
    exc: BaseException,
    *,
    provider: str = PROVIDER_CODEX_LEGACY,
    model: Optional[str] = None,
    operation: str = "",
) -> SupervisorRunError:
    """Translate a legacy Codex failure into a canonical supervisor failure.

    Preserves category (mapped), model, operation, return code and sanitized
    text. Never leaks credentials, headers or raw provider payloads:
    everything passes through :func:`supervisor.errors.sanitize_diagnostic`.
    """
    if isinstance(exc, SupervisorRunError):
        return exc

    view = _codex_error_view(exc)
    if view is None:
        # An unclassified runner failure must fail closed.
        detail = sanitize_diagnostic(str(exc) or "Codex structured run failed.")
        return SupervisorRunError(
            detail,
            provider=provider,
            model=model,
            operation=operation,
            category=CATEGORY_UNKNOWN,
            retryable=False,
            sanitized_message=detail,
        )

    mapped = CODEX_CATEGORY_MAP.get(view.category, CATEGORY_UNKNOWN)
    # Retryable ONLY when the mapped category is retryable AND the legacy
    # classifier agreed. Either side saying "no" fails closed.
    retryable = is_retryable_category(mapped) and bool(view.retryable)

    detail = sanitize_diagnostic(view.sanitized_error or str(exc) or view.category)
    if view.source:
        detail = f"{detail} [source={view.source}]"

    return SupervisorRunError(
        detail,
        provider=provider,
        model=model,
        operation=operation,
        category=mapped,
        retryable=retryable,
        http_status=None,  # Codex is a subprocess: there is no HTTP status.
        provider_error_code=(
            str(view.returncode) if view.returncode is not None else None
        ),
        sanitized_message=detail,
        # Audit-parity only: the provider's own raw classification.
        origin_provider=view.provider or None,
        origin_category=view.category or None,
        origin_code=(
            str(view.returncode) if view.returncode is not None else None
        ),
        origin_source=view.source or None,
    )


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class LegacyCodexSupervisorAdapter:
    """``SupervisorPort`` implementation backed by the existing ``CodexRunner``.

    The runner is injected (see :class:`StructuredRunnerPort`); the adapter
    never imports the Worker Tandem module, never touches the FSM or SQLite,
    and never runs a live CLI under test.
    """

    provider_id = PROVIDER_CODEX_LEGACY

    def __init__(
        self,
        runner: StructuredRunnerPort,
        *,
        config: Optional[ProviderConfig] = None,
        provider_id: str = PROVIDER_CODEX_LEGACY,
        model: Optional[str] = None,
        paths: Optional[PathProvider] = None,
        workdir: Optional[Path] = None,
        effort_policy: Optional[EffortPolicy] = None,
        reasoning_effort: Optional[str] = None,
        identity_policy: str = IDENTITY_REJECT,
        validation_mode: str = VALIDATION_ADAPTER,
    ) -> None:
        if identity_policy not in IDENTITY_POLICIES:
            raise ValueError(
                f"identity_policy must be one of {IDENTITY_POLICIES}, "
                f"got {identity_policy!r}"
            )
        if validation_mode not in VALIDATION_MODES:
            raise ValueError(
                f"validation_mode must be one of {VALIDATION_MODES}, "
                f"got {validation_mode!r}"
            )
        self.identity_policy = identity_policy
        self.validation_mode = validation_mode
        self._runner = runner
        self.provider_id = provider_id
        self.model = model if model is not None else getattr(runner, "model", None)
        self.config = config or ProviderConfig(
            provider_id=provider_id,
            display_name="Codex Legacy",
            model=self.model,
            # Legacy is opt-in only: never mandatory, never the default path.
            enabled=False,
            supports_structured_output=True,
            cost_policy="free",
            priority=99,
        )
        if paths is None:
            base = workdir if workdir is not None else getattr(runner, "workdir", None)
            if base is None:
                raise ValueError(
                    "LegacyCodexSupervisorAdapter needs paths= or workdir= "
                    "(or a runner exposing .workdir)"
                )
            paths = default_codex_paths(base)
        self._paths = paths
        self._effort_policy = effort_policy or default_effort_policy
        self._fixed_effort = reasoning_effort

    # -- SupervisorPort ----------------------------------------------------- #

    def plan(self, request: SupervisorPlanRequest) -> SupervisorPlanResult:
        payload = self._run(PLAN_OPERATION, request, supervisor_plan_schema())
        return SupervisorPlanResult(
            payload=payload, provider=self.provider_id, model=self.model
        )

    def review(self, request: SupervisorReviewRequest) -> SupervisorReviewResult:
        payload = self._run(REVIEW_OPERATION, request, supervisor_review_schema())
        return SupervisorReviewResult(
            payload=payload, provider=self.provider_id, model=self.model
        )

    def health_check(self) -> ProviderHealth:
        """Cheap probe only: ``available()``. Never runs a Codex completion."""
        probe = getattr(self._runner, "available", None)
        if not callable(probe):
            return ProviderHealth(
                provider_id=self.provider_id,
                status=HEALTH_PROVIDER_ERROR,
                detail="runner does not expose available()",
                model=self.model,
            )
        try:
            ok, info = probe()
        except Exception as exc:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=HEALTH_PROVIDER_ERROR,
                detail=sanitize_diagnostic(exc),
                model=self.model,
            )
        return ProviderHealth(
            provider_id=self.provider_id,
            status=HEALTH_READY if ok else HEALTH_PROVIDER_ERROR,
            detail=sanitize_diagnostic(info),
            model=self.model,
        )

    # -- internals ---------------------------------------------------------- #

    def _effort(self, request) -> str:
        if self._fixed_effort:
            return self._fixed_effort
        return self._effort_policy(
            request.risk, bool(request.requires_human), int(request.attempt or 0)
        )

    def _run(self, operation: str, request, schema: dict) -> dict:
        paths = self._paths(operation, request.step_no, request.attempt)
        self._ensure_schema(paths.schema_path, schema)
        try:
            raw = self._runner.run_structured(
                request.prompt,
                paths.schema_path,
                paths.output_path,
                paths.trace_path,
                reasoning_effort=self._effort(request),
            )
        except SupervisorRunError:
            raise
        except Exception as exc:
            raise translate_codex_exception(
                exc,
                provider=self.provider_id,
                model=self.model,
                operation=operation,
            ) from exc

        payload = self._identity_hygiene(raw)

        if self.validation_mode == VALIDATION_RUNNER:
            # The injected runner already validated this payload against the
            # same canonical schema, so re-validating here would change pre-M4
            # behaviour. The adapter still guarantees a JSON object.
            if not isinstance(payload, dict):
                raise SupervisorRunError(
                    "provider did not return a JSON object",
                    provider=self.provider_id,
                    model=self.model,
                    operation=operation,
                    category=CATEGORY_CONTRACT,
                    retryable=False,
                    sanitized_message="provider payload is not a JSON object",
                )
            return dict(payload)

        if operation == PLAN_OPERATION:
            return normalize_plan_payload(
                payload, provider=self.provider_id, model=self.model, operation=operation
            )
        return normalize_review_payload(
            payload, provider=self.provider_id, model=self.model, operation=operation
        )

    def _identity_hygiene(self, raw: Any) -> Any:
        """Apply the configured identity policy to a raw provider payload.

        ``IDENTITY_STRIP_STEP_NO`` mirrors ``CodexRunner`` exactly (it pops
        ``step_no`` before schema validation); the authoritative value is
        always injected later by the controller.
        """
        if self.identity_policy != IDENTITY_STRIP_STEP_NO:
            return raw
        if not isinstance(raw, dict):
            return raw
        return {key: value for key, value in raw.items() if key != "step_no"}

    @staticmethod
    def _ensure_schema(schema_path: Path, schema: dict) -> None:
        """Write the canonical schema atomically (Codex needs the file to exist).

        Idempotent: identical content is left untouched. This is the only file
        the adapter ever writes.
        """
        schema_path = Path(schema_path)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(schema, ensure_ascii=False, indent=2)
        try:
            if schema_path.read_text(encoding="utf-8") == payload:
                return
        except OSError:
            pass
        temp = schema_path.with_name(schema_path.name + ".tmp")
        temp.write_text(payload, encoding="utf-8")
        os.replace(temp, schema_path)



