"""Legacy Codex supervisor adapter tests (milestone M3).

Proves that the EXISTING Codex implementation can satisfy the new
provider-neutral ``SupervisorPort`` contract - without running the Codex CLI,
without a live API, without touching the FSM or SQLite.

The runner is always injected as a deterministic fake. One test additionally
uses the REAL ``CodexRunError`` class loaded by path from
``worker_tandem_v0.1.2.py`` to prove the duck-typed translation works against
the genuine legacy exception (still without executing Codex).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

from supervisor import (
    CODEX_CATEGORY_MAP,
    CODEX_PLAN_SCHEMA_NAME,
    CODEX_REVIEW_SCHEMA_NAME,
    PROVIDER_CODEX_LEGACY,
    CodexArtifactPaths,
    LegacyCodexSupervisorAdapter,
    StructuredRunnerPort,
    SupervisorPort,
    SupervisorPlanRequest,
    SupervisorReviewRequest,
    SupervisorRunError,
    default_codex_paths,
    default_effort_policy,
    fake_plan_payload,
    fake_review_payload,
    supervisor_plan_schema,
    supervisor_review_schema,
    translate_codex_exception,
)
from supervisor.errors import (
    CATEGORY_AUTH,
    CATEGORY_BILLING_DISABLED,
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_SCHEMA,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    CATEGORY_UNKNOWN,
    is_retryable_category,
)
from supervisor.models import (
    HEALTH_PROVIDER_ERROR,
    HEALTH_READY,
    PLAN_OPERATION,
    REVIEW_OPERATION,
)

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem_legacy", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_legacy"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()

PLAN_DEF = {
    "plan_version": "legacy-test-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "ONE",
            "description": "legacy adapter test step",
            "risk": "LOW",
        },
    ],
}

# --------------------------------------------------------------------------- #
# Deterministic fakes
# --------------------------------------------------------------------------- #


class FakeCodexRunError(RuntimeError):
    """Shape-identical stand-in for the legacy ``CodexRunError``."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        returncode=None,
        provider: str = "gpt-5-codex",
        sanitized_error: str = "",
        source: str = "jsonl",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.returncode = returncode
        self.provider = provider
        self.sanitized_error = sanitized_error
        self.source = source


class FakeStructuredRunner:
    """Records calls; never spawns a process."""

    def __init__(
        self,
        *,
        plan_payload=None,
        review_payload=None,
        plan_error=None,
        review_error=None,
        model: str = "gpt-5-codex",
        available=(True, "codex-cli 0.157.1"),
        workdir=None,
    ) -> None:
        self.model = model
        self.workdir = workdir
        self._plan_payload = plan_payload if plan_payload is not None else fake_plan_payload()
        self._review_payload = (
            review_payload if review_payload is not None else fake_review_payload()
        )
        self._plan_error = plan_error
        self._review_error = review_error
        self._available = available
        self.calls: list[dict] = []

    def run_structured(
        self,
        prompt,
        schema_path,
        output_path,
        trace_path,
        timeout_sec=None,
        reasoning_effort=None,
    ):
        name = pathlib.Path(schema_path).name
        self.calls.append(
            {
                "prompt": prompt,
                "schema_path": pathlib.Path(schema_path),
                "schema_name": name,
                "output_path": pathlib.Path(output_path),
                "trace_path": pathlib.Path(trace_path),
                "timeout_sec": timeout_sec,
                "reasoning_effort": reasoning_effort,
            }
        )
        if name == CODEX_PLAN_SCHEMA_NAME:
            if self._plan_error is not None:
                raise self._plan_error
            return dict(self._plan_payload)
        if self._review_error is not None:
            raise self._review_error
        return dict(self._review_payload)

    def available(self):
        return self._available


def plan_request(step_no: int = 1, attempt: int = 1, risk: str = "LOW", requires_human: bool = False):
    return SupervisorPlanRequest(
        step_no=step_no,
        attempt=attempt,
        risk=risk,
        requires_human=requires_human,
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


def make_adapter(runner, **kwargs) -> LegacyCodexSupervisorAdapter:
    return LegacyCodexSupervisorAdapter(runner, **kwargs)


# --------------------------------------------------------------------------- #
# 1-5  PLAN / REVIEW success paths
# --------------------------------------------------------------------------- #


def test_plan_success_maps_to_supervisor_plan_result(tmp_path):
    runner = FakeStructuredRunner(workdir=tmp_path / ".worker_tandem")
    adapter = make_adapter(runner)

    result = adapter.plan(plan_request())

    assert result.provider == PROVIDER_CODEX_LEGACY
    assert result.model == "gpt-5-codex"
    assert result.payload["decision"] == "PLAN_READY"
    assert "step_no" not in result.payload
    assert "attempt" not in result.payload
    # Codex needs the schema file to exist on disk: the adapter guarantees it.
    schema_path = tmp_path / ".worker_tandem" / "context" / CODEX_PLAN_SCHEMA_NAME
    assert json.loads(schema_path.read_text(encoding="utf-8")) == supervisor_plan_schema()


@pytest.mark.parametrize(
    "verdict", ["APPROVE", "REVISE", "BLOCKED", "ARCHITECTURE_DECISION_REQUIRED"]
)
def test_review_verdicts_mapped(tmp_path, verdict):
    runner = FakeStructuredRunner(
        review_payload=fake_review_payload(verdict=verdict),
        workdir=tmp_path / ".worker_tandem",
    )
    adapter = make_adapter(runner)

    result = adapter.review(review_request())

    assert result.payload["verdict"] == verdict
    assert result.provider == PROVIDER_CODEX_LEGACY
    assert result.model == "gpt-5-codex"


# --------------------------------------------------------------------------- #
# 6-9  Malformed / identity-claiming payloads fail closed
# --------------------------------------------------------------------------- #


def test_malformed_plan_rejected(tmp_path):
    runner = FakeStructuredRunner(
        plan_payload={"decision": "PLAN_READY"}, workdir=tmp_path / ".worker_tandem"
    )
    adapter = make_adapter(runner)
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False


def test_malformed_review_rejected(tmp_path):
    runner = FakeStructuredRunner(
        review_payload={"verdict": "APPROVE"}, workdir=tmp_path / ".worker_tandem"
    )
    adapter = make_adapter(runner)
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False


def test_provider_supplied_step_no_rejected(tmp_path):
    payload = fake_plan_payload()
    payload["step_no"] = 42
    runner = FakeStructuredRunner(
        plan_payload=payload, workdir=tmp_path / ".worker_tandem"
    )
    adapter = make_adapter(runner)
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert "step_no" in info.value.sanitized_message


def test_provider_supplied_attempt_rejected(tmp_path):
    payload = fake_review_payload()
    payload["attempt"] = 3
    runner = FakeStructuredRunner(
        review_payload=payload, workdir=tmp_path / ".worker_tandem"
    )
    adapter = make_adapter(runner)
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_CONTRACT


# --------------------------------------------------------------------------- #
# 10-16  CodexRunError -> SupervisorRunError translation
# --------------------------------------------------------------------------- #

TRANSLATION_CASES = [
    ("quota", True, CATEGORY_QUOTA_WITH_RESET, True),
    ("rate_limit", True, CATEGORY_RATE_LIMIT, True),
    ("timeout", True, CATEGORY_TIMEOUT, True),
    ("transport", True, CATEGORY_TRANSPORT, True),
    ("network", True, CATEGORY_NETWORK, True),
    ("provider_unavailable", True, CATEGORY_PROVIDER_UNAVAILABLE, True),
    ("auth", False, CATEGORY_AUTH, False),
    ("billing", False, CATEGORY_BILLING_DISABLED, False),
    ("config", False, CATEGORY_INVALID_REQUEST, False),
    ("contract", False, CATEGORY_CONTRACT, False),
    ("schema", False, CATEGORY_SCHEMA, False),
    ("quota_unconfirmed", False, CATEGORY_UNKNOWN, False),
    ("unknown", False, CATEGORY_UNKNOWN, False),
    ("never_seen_before", False, CATEGORY_UNKNOWN, False),
]


@pytest.mark.parametrize(
    "codex_category,codex_retryable,mapped,expected_retryable", TRANSLATION_CASES
)
def test_plan_error_translated_through_adapter(
    tmp_path, codex_category, codex_retryable, mapped, expected_retryable
):
    error = FakeCodexRunError(
        "codex failed",
        category=codex_category,
        retryable=codex_retryable,
        returncode=7,
    )
    runner = FakeStructuredRunner(plan_error=error, workdir=tmp_path / ".worker_tandem")
    adapter = make_adapter(runner)

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    err = info.value
    assert err.category == mapped
    assert err.retryable is expected_retryable
    assert err.provider == PROVIDER_CODEX_LEGACY
    assert err.operation == PLAN_OPERATION
    assert err.provider_error_code == "7"
    assert err.http_status is None  # Codex is a subprocess, not HTTP
    assert json.dumps(err.to_dict())  # audit-safe serialization


def test_review_error_translated_through_adapter(tmp_path):
    error = FakeCodexRunError(
        "codex review failed", category="rate_limit", retryable=True
    )
    runner = FakeStructuredRunner(
        review_error=error, workdir=tmp_path / ".worker_tandem"
    )
    adapter = make_adapter(runner)
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_RATE_LIMIT
    assert info.value.retryable is True
    assert info.value.operation == REVIEW_OPERATION


def test_fail_closed_when_legacy_retryable_flag_disagrees():
    # The category maps to a retryable supervisor category, but the legacy
    # classifier said non-retryable: the result MUST fail closed.
    exc = FakeCodexRunError("odd", category="quota", retryable=False)
    err = translate_codex_exception(exc, model="gpt-5-codex", operation=PLAN_OPERATION)
    assert err.category == CATEGORY_QUOTA_WITH_RESET
    assert err.retryable is False


def test_unclassified_exception_fails_closed(tmp_path):
    runner = FakeStructuredRunner(
        plan_error=RuntimeError("something unexpected"),
        workdir=tmp_path / ".worker_tandem",
    )
    adapter = make_adapter(runner)
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_UNKNOWN
    assert info.value.retryable is False


def test_legacy_retryable_categories_all_map_to_retryable():
    assert wt.RETRYABLE_CODEX_CATEGORIES
    for category in wt.RETRYABLE_CODEX_CATEGORIES:
        mapped = CODEX_CATEGORY_MAP.get(category)
        assert mapped is not None, category
        assert is_retryable_category(mapped), category


def test_real_codex_run_error_is_translated():
    """The genuine legacy exception class must translate (duck-typed)."""
    real = wt.CodexRunError(
        "Codex PLAN failed",
        category="quota",
        retryable=True,
        returncode=1,
        provider="gpt-5-codex",
        sanitized_error="You've hit your usage limit. Try again at Sep 27th.",
        source="jsonl",
    )
    err = translate_codex_exception(real, model="gpt-5-codex", operation=PLAN_OPERATION)
    assert isinstance(err, SupervisorRunError)
    assert err.category == CATEGORY_QUOTA_WITH_RESET
    assert err.retryable is True
    assert err.provider_error_code == "1"
    assert "usage limit" in err.sanitized_message
    assert "source=jsonl" in err.sanitized_message


# --------------------------------------------------------------------------- #
# 17  Sanitized diagnostics
# --------------------------------------------------------------------------- #


def test_sanitized_error_text_contains_no_secret():
    secret = "sk-live-LEGACYSECRET123456789"

    exc = FakeCodexRunError(
        f"Codex failed with {secret}",
        category="auth",
        retryable=False,
        sanitized_error=f"Invalid API key {secret}",
    )
    err = translate_codex_exception(exc, model="gpt-5-codex", operation=REVIEW_OPERATION)

    assert secret not in str(err)
    assert secret not in err.sanitized_message
    assert secret not in json.dumps(err.to_dict())
    assert "***" in err.sanitized_message


# --------------------------------------------------------------------------- #
# 18  Provider / model metadata
# --------------------------------------------------------------------------- #


def test_model_and_provider_metadata_retained(tmp_path):
    runner = FakeStructuredRunner(
        model="gpt-5.1-codex", workdir=tmp_path / ".worker_tandem"
    )
    adapter = make_adapter(runner)

    assert adapter.provider_id == PROVIDER_CODEX_LEGACY
    assert adapter.model == "gpt-5.1-codex"

    result = adapter.plan(plan_request())
    assert result.provider == PROVIDER_CODEX_LEGACY
    assert result.model == "gpt-5.1-codex"

    # The legacy provider is opt-in only: never the default mandatory path.
    assert adapter.config.enabled is False
    assert adapter.config.priority == 99
    assert adapter.config.supports_structured_output is True


def test_explicit_model_overrides_runner_model(tmp_path):
    runner = FakeStructuredRunner(workdir=tmp_path / ".worker_tandem")
    adapter = make_adapter(runner, model="pinned-model")
    assert adapter.model == "pinned-model"
    assert adapter.plan(plan_request()).model == "pinned-model"


# --------------------------------------------------------------------------- #
# 19-20  Port conformance and the injected runner contract
# --------------------------------------------------------------------------- #


def test_adapter_satisfies_supervisor_port(tmp_path):
    runner = FakeStructuredRunner(workdir=tmp_path / ".worker_tandem")
    adapter = make_adapter(runner)
    assert isinstance(adapter, SupervisorPort)
    assert isinstance(runner, StructuredRunnerPort)


def test_runner_receives_correct_operation_schema_and_context(tmp_path):
    workdir = tmp_path / ".worker_tandem"
    runner = FakeStructuredRunner(workdir=workdir)
    adapter = make_adapter(runner)

    adapter.plan(plan_request(step_no=1, attempt=1))
    plan_call = runner.calls[-1]
    assert plan_call["prompt"] == "PLAN prompt"
    assert plan_call["schema_name"] == CODEX_PLAN_SCHEMA_NAME
    assert plan_call["schema_path"] == workdir / "context" / CODEX_PLAN_SCHEMA_NAME
    assert plan_call["output_path"] == workdir / "from_codex" / "step_001_plan.json"
    assert plan_call["trace_path"] == workdir / "logs" / "step_001_codex_plan_trace.json"
    assert plan_call["reasoning_effort"] == "low"
    assert plan_call["timeout_sec"] is None  # the runner keeps its own timeout

    adapter.review(review_request(step_no=2, attempt=3))
    review_call = runner.calls[-1]
    assert review_call["prompt"] == "REVIEW prompt"
    assert review_call["schema_name"] == CODEX_REVIEW_SCHEMA_NAME
    assert review_call["output_path"] == workdir / "from_codex" / "step_002_review.json"
    assert (
        review_call["trace_path"]
        == workdir / "logs" / "step_002_codex_review_trace.json"
    )
    assert review_call["reasoning_effort"] == "medium"  # attempt > 1


def test_effort_policy_is_injected_and_mirrors_legacy(tmp_path):
    runner = FakeStructuredRunner(workdir=tmp_path / ".worker_tandem")
    adapter = make_adapter(runner)
    adapter.plan(plan_request(attempt=1, risk="LOW"))
    assert runner.calls[-1]["reasoning_effort"] == "low"
    adapter.plan(plan_request(attempt=2, risk="LOW"))
    assert runner.calls[-1]["reasoning_effort"] == "medium"
    adapter.plan(plan_request(attempt=1, risk="HIGH"))
    assert runner.calls[-1]["reasoning_effort"] == "high"
    adapter.plan(plan_request(attempt=1, risk="LOW", requires_human=True))
    assert runner.calls[-1]["reasoning_effort"] == "high"


def test_fixed_reasoning_effort_override(tmp_path):
    runner = FakeStructuredRunner(workdir=tmp_path / ".worker_tandem")
    adapter = make_adapter(runner, reasoning_effort="high")
    adapter.plan(plan_request(attempt=1, risk="LOW"))
    assert runner.calls[-1]["reasoning_effort"] == "high"


def test_default_codex_paths_layout(tmp_path):
    paths = default_codex_paths(tmp_path / ".worker_tandem")
    plan = paths(PLAN_OPERATION, 3, 1)
    assert isinstance(plan, CodexArtifactPaths)
    assert plan.schema_path.name == CODEX_PLAN_SCHEMA_NAME
    assert plan.output_path.name == "step_003_plan.json"
    assert plan.trace_path.name == "step_003_codex_plan_trace.json"

    review = paths(REVIEW_OPERATION, 3, 1)
    assert review.schema_path.name == CODEX_REVIEW_SCHEMA_NAME
    assert review.output_path.name == "step_003_review.json"
    assert review.trace_path.name == "step_003_codex_review_trace.json"


def test_adapter_derives_paths_from_runner_workdir(tmp_path):
    workdir = tmp_path / ".worker_tandem"
    runner = FakeStructuredRunner(workdir=workdir)
    adapter = make_adapter(runner)  # neither paths= nor workdir= given
    adapter.plan(plan_request())
    assert runner.calls[-1]["schema_path"] == workdir / "context" / CODEX_PLAN_SCHEMA_NAME


def test_adapter_requires_paths_or_workdir():
    class Bare:
        def run_structured(self, *args, **kwargs):
            return {}

    with pytest.raises(ValueError):
        LegacyCodexSupervisorAdapter(Bare())


# --------------------------------------------------------------------------- #
# Health checks are cheap (never a Codex completion)
# --------------------------------------------------------------------------- #


def test_health_check_ready(tmp_path):
    adapter = make_adapter(
        FakeStructuredRunner(
            workdir=tmp_path / "a", available=(True, "codex-cli 0.157.1")
        )
    )
    health = adapter.health_check()
    assert health.status == HEALTH_READY
    assert health.ready is True
    assert health.provider_id == PROVIDER_CODEX_LEGACY
    assert health.model == "gpt-5-codex"
    assert health.detail == "codex-cli 0.157.1"


def test_health_check_error_when_unavailable(tmp_path):
    adapter = make_adapter(
        FakeStructuredRunner(
            workdir=tmp_path / "b",
            available=(False, "Codex executable not found: codex"),
        )
    )
    health = adapter.health_check()
    assert health.status == HEALTH_PROVIDER_ERROR
    assert health.ready is False
    assert "not found" in health.detail


def test_health_check_without_available_probe(tmp_path):
    class Bare:
        model = None
        workdir = tmp_path / ".worker_tandem"

        def run_structured(self, *args, **kwargs):
            return {}

    adapter = make_adapter(Bare())
    assert adapter.health_check().status == HEALTH_PROVIDER_ERROR


def test_health_check_never_runs_a_completion(tmp_path, monkeypatch):
    runner = FakeStructuredRunner(workdir=tmp_path / ".worker_tandem")

    def boom(*args, **kwargs):
        raise AssertionError("health_check must not run a Codex completion")

    monkeypatch.setattr(runner, "run_structured", boom)
    assert make_adapter(runner).health_check().status == HEALTH_READY


# --------------------------------------------------------------------------- #
# 21  No live Codex process
# --------------------------------------------------------------------------- #


def test_no_live_codex_process_invoked(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a real subprocess was spawned")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "check_output", boom)

    runner = FakeStructuredRunner(workdir=tmp_path / ".worker_tandem")
    adapter = make_adapter(runner)

    assert adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"
    assert adapter.review(review_request()).payload["verdict"] == "APPROVE"
    assert adapter.health_check().status == HEALTH_READY
    assert len(runner.calls) == 2


# --------------------------------------------------------------------------- #
# Contract parity with the legacy Codex integration (drift guard)
# --------------------------------------------------------------------------- #


def test_canonical_schemas_match_legacy_codex_schemas():
    assert supervisor_plan_schema() == wt.codex_plan_schema()
    assert supervisor_review_schema() == wt.codex_review_schema()


def test_default_effort_policy_matches_controller_policy(tmp_path):
    controller = wt.TandemController(tmp_path)
    for risk, requires_human, attempt in (
        ("LOW", False, 1),
        ("LOW", False, 2),
        ("MEDIUM", False, 1),
        ("HIGH", False, 1),
        ("LOW", True, 1),
    ):
        step = wt.TandemStep(
            step_no=1,
            phase="FOUNDATION",
            title="t",
            description="d",
            state="READY_FOR_CODEX_PLAN",
            attempt=attempt,
            max_attempts=3,
            risk=risk,
            requires_human=requires_human,
        )
        assert default_effort_policy(
            risk, requires_human, attempt
        ) == controller._codex_reasoning_effort(step)
    controller.db.close()


# --------------------------------------------------------------------------- #
# 22-23  No FSM mutation, no SQLite mutation
# --------------------------------------------------------------------------- #


def test_adapter_does_not_mutate_fsm_or_sqlite(tmp_path):
    controller = wt.TandemController(tmp_path)
    controller.db.import_plan(PLAN_DEF)
    before = controller.db.get_step(1)

    def count(table: str) -> int:
        row = controller.db.db.execute(
            f"SELECT COUNT(*) AS n FROM {table}"
        ).fetchone()
        return int(row["n"])

    events_before = count("events")
    plans_before = count("codex_plans")
    reviews_before = count("codex_reviews")

    runner = FakeStructuredRunner(workdir=controller.db.workdir)
    adapter = make_adapter(runner)
    # M3: the adapter is exercised directly; the controller never calls it.
    adapter.plan(plan_request())
    adapter.review(review_request())

    after = controller.db.get_step(1)
    assert after.state == before.state
    assert after.attempt == before.attempt
    assert count("events") == events_before
    assert count("codex_plans") == plans_before
    assert count("codex_reviews") == reviews_before
    assert controller.db.latest_codex_plan(1) is None
    assert controller.db.latest_codex_review(1) is None
    controller.db.close()


def test_adapter_alone_creates_no_sqlite(tmp_path):
    workdir = tmp_path / ".worker_tandem"
    adapter = make_adapter(FakeStructuredRunner(workdir=workdir))
    adapter.plan(plan_request())

    assert not (workdir / "tandem_state.db").exists()
    # The canonical schema file is the ONLY file the adapter may write.
    files = {path.name for path in workdir.rglob("*") if path.is_file()}
    assert files == {CODEX_PLAN_SCHEMA_NAME}


# --------------------------------------------------------------------------- #
# M3 guarantees: the legacy direct Codex path is untouched
# --------------------------------------------------------------------------- #


def test_legacy_direct_codex_path_is_still_present(tmp_path):
    controller = wt.TandemController(tmp_path)
    # Worker Tandem still uses CodexRunner directly...
    assert isinstance(controller.codex, wt.CodexRunner)
    # ...and the new abstraction is NOT wired in yet.
    assert not hasattr(controller, "router")
    controller.db.close()


def test_supervisor_package_does_not_import_worker_tandem():
    import supervisor
    import supervisor.codex_legacy_adapter as adapter_module

    # The supervisor package exposes no Worker Tandem controller/GUI symbols.
    for name in ("TandemController", "CodexRunner", "TandemGUI", "wt"):
        assert not hasattr(supervisor, name)

    source = pathlib.Path(adapter_module.__file__).read_text(encoding="utf-8")
    assert "import worker_tandem" not in source
    assert "from worker_tandem" not in source







