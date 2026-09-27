"""Supervisor controller-integration tests (milestone M4).

Proves the M4 contract: Worker Tandem's PLAN and REVIEW now execute through
``SupervisorRouter`` while every observable behaviour stays identical to the
pre-M4 direct-Codex flow.

The key acceptance gate is the PARITY test at the bottom: the router path and a
faithful re-implementation of the old direct path must persist byte-identical
``codex_plans`` / ``codex_reviews`` payloads and artifacts.

No live Codex process, no network, no paid API.
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
    PROVIDER_CODEX_LEGACY,
    SUPERVISOR_ERROR_EVENT,
    SUPERVISOR_RESULT_EVENT,
    SUPERVISOR_SELECTED_EVENT,
)

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem_m4", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_m4"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()

PLAN_DEF = {
    "plan_version": "m4-test-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "STEP ONE",
            "description": "m4 integration step one",
            "risk": "LOW",
        },
    ],
}

PLAN_OK = {
    "decision": "PLAN_READY",
    "goal": "do the safe thing",
    "files": ["a.py"],
    "changes": ["edit a.py"],
    "checks": ["run tests"],
    "architecture_risk": "none",
    "open_questions": [],
    "rollback": ["revert a.py"],
    "summary": "ok",
}

REVIEW_OK = {
    "verdict": "APPROVE",
    "issues": [],
    "required_changes": [],
    "checks": ["tests green"],
    "architecture_risk": "none",
    "summary": "looks good",
}

QUOTA_MESSAGE = (
    "You've hit your usage limit. Upgrade to Pro or try again at "
    "Sep 27th, 2026 1:29 AM."
)

#: The seven tables the pre-M4 schema created (no M4 migration is allowed).
EXPECTED_TABLES = {
    "meta",
    "steps",
    "plan_versions",
    "codex_plans",
    "cline_reports",
    "codex_reviews",
    "approvals",
    "events",
}

EXPECTED_STEP_COLUMNS = {
    "step_no",
    "phase",
    "title",
    "description",
    "state",
    "attempt",
    "max_attempts",
    "risk",
    "requires_human",
    "created_at",
    "started_at",
    "finished_at",
    "verified_at",
    "last_update_at",
}


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def ctrl(tmp_path):
    controller = wt.TandemController(tmp_path)
    controller.db.import_plan(PLAN_DEF)
    yield controller
    controller.db.close()


def stub_runner(controller, payload, *, capture=None):
    """Replace the Codex call with a fixed, already-validated payload."""

    def fake(
        prompt, schema_path, output_path, trace_path, timeout_sec=None, reasoning_effort=None
    ):
        if capture is not None:
            capture.update(
                prompt=prompt,
                schema_path=pathlib.Path(schema_path),
                output_path=pathlib.Path(output_path),
                trace_path=pathlib.Path(trace_path),
                reasoning_effort=reasoning_effort,
            )
        return dict(payload)

    controller.codex.run_structured = fake


def fail_runner(
    controller, *, category="quota", retryable=True, returncode=1, sanitized=QUOTA_MESSAGE
):
    def boom(*args, **kwargs):
        raise wt.CodexRunError(
            f"Codex structured run failed: {category}",
            category=category,
            retryable=retryable,
            returncode=returncode,
            provider="test-provider",
            sanitized_error=sanitized,
            source="jsonl",
        )

    controller.codex.run_structured = boom


def report_ok(step_no=1, status="DONE"):
    return {
        "step_no": step_no,
        "status": status,
        "files_created": [],
        "files_changed": ["a.py"],
        "files_deleted": [],
        "tests": {"passed": 1, "failed": 0, "command": "pytest -q"},
        "dependencies_added": [],
        "architecture_questions": [],
        "issues": [],
        "summary": "done",
    }


def to_cline_dispatched(controller, *, attempt=None):
    controller.db.transition(1, "CODEX_PLAN_RUNNING")
    if attempt is not None:
        while controller.db.get_step(1).attempt < attempt:
            controller.db.increment_attempt(1)
    controller.db.transition(1, "PLAN_READY")
    controller.db.transition(1, "WAITING_HUMAN_APPROVAL")
    controller.db.transition(1, "CLINE_DISPATCHED")


def accept_report(controller, *, status="DONE"):
    wt.write_json_atomic(
        controller.expected_cline_report_path(1), report_ok(status=status)
    )
    return controller.process_cline_report()


def to_review_received(controller, *, attempt=None):
    to_cline_dispatched(controller, attempt=attempt)
    accept_report(controller)


def events(controller, event_type, step_no=1):
    rows = controller.db.db.execute(
        "SELECT details FROM events WHERE step_no=? AND event_type=? ORDER BY id",
        (step_no, event_type),
    ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# 10-12  PLAN / REVIEW / retry all route through the supervisor
# --------------------------------------------------------------------------- #


def test_run_codex_plan_routes_through_router(ctrl):
    stub_runner(ctrl, PLAN_OK)
    result = ctrl.run_codex_plan()

    assert result["decision"] == "PLAN_READY"
    assert ctrl.supervisor_router.calls == [
        {
            "operation": "PLAN",
            "provider": PROVIDER_CODEX_LEGACY,
            "model": None,
            "step_no": 1,
            "attempt": 1,
            "risk": "LOW",
            "fallback_from": None,
        }
    ]
    # Provider-neutral audit is emitted alongside the legacy state events.
    assert len(events(ctrl, SUPERVISOR_SELECTED_EVENT)) == 1
    assert len(events(ctrl, SUPERVISOR_RESULT_EVENT)) == 1


def test_run_codex_review_routes_through_router(ctrl):
    to_review_received(ctrl, attempt=2)
    stub_runner(ctrl, REVIEW_OK)
    result = ctrl.run_codex_review()

    assert result["verdict"] == "APPROVE"
    call = ctrl.supervisor_router.calls[-1]
    assert call["operation"] == "REVIEW"
    assert call["provider"] == PROVIDER_CODEX_LEGACY
    assert call["step_no"] == 1
    assert call["attempt"] == 2
    assert len(events(ctrl, SUPERVISOR_RESULT_EVENT)) == 1


def test_retry_codex_review_routes_through_router(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_runner(ctrl)
    ctrl.run_codex_review()  # deferred -> CODEX_REVIEW_RETRYABLE
    assert ctrl.db.get_step(1).state == "CODEX_REVIEW_RETRYABLE"

    before = len(ctrl.supervisor_router.calls)
    stub_runner(ctrl, REVIEW_OK)
    result = ctrl.retry_codex_review()

    assert result["verdict"] == "APPROVE"
    assert len(ctrl.supervisor_router.calls) == before + 1
    assert ctrl.supervisor_router.calls[-1]["operation"] == "REVIEW"
    assert ctrl.db.get_step(1).state == "REVIEW_APPROVED"


# --------------------------------------------------------------------------- #
# 13-15  Identical persisted payloads and transitions
# --------------------------------------------------------------------------- #


def test_plan_db_payload_is_byte_identical(ctrl):
    stub_runner(ctrl, PLAN_OK)
    result = ctrl.run_codex_plan()

    assert result == {**PLAN_OK, "step_no": 1}
    row = ctrl.db.db.execute(
        "SELECT plan_json FROM codex_plans WHERE step_no=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    expected = json.dumps(
        {**PLAN_OK, "step_no": 1}, ensure_ascii=False, sort_keys=True, indent=2
    )
    assert row["plan_json"] == expected


def test_review_db_payload_is_byte_identical(ctrl):
    to_review_received(ctrl)
    stub_runner(ctrl, REVIEW_OK)
    result = ctrl.run_codex_review()

    assert result == {**REVIEW_OK, "step_no": 1}
    row = ctrl.db.db.execute(
        "SELECT review_json FROM codex_reviews WHERE step_no=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    expected = json.dumps(
        {**REVIEW_OK, "step_no": 1}, ensure_ascii=False, sort_keys=True, indent=2
    )
    assert row["review_json"] == expected


def test_plan_state_transition_unchanged(ctrl):
    stub_runner(ctrl, PLAN_OK)
    ctrl.run_codex_plan()
    step = ctrl.db.get_step(1)
    assert step.state == "PLAN_READY"
    assert step.attempt == 1


# --------------------------------------------------------------------------- #
# 16-17  The controller owns identity
# --------------------------------------------------------------------------- #


def test_controller_injects_step_no(ctrl):
    stub_runner(ctrl, PLAN_OK)
    assert ctrl.run_codex_plan()["step_no"] == 1

    ctrl.approve_plan_and_dispatch_cline()
    accept_report(ctrl)
    stub_runner(ctrl, REVIEW_OK)
    assert ctrl.run_codex_review()["step_no"] == 1


def test_provider_cannot_override_plan_identity(ctrl):
    stub_runner(ctrl, {**PLAN_OK, "step_no": 999})
    result = ctrl.run_codex_plan()

    assert result["step_no"] == 1
    assert ctrl.db.latest_codex_plan(1)["step_no"] == 1
    artifact = json.loads(
        (ctrl.db.from_codex / "step_001_plan.json").read_text(encoding="utf-8")
    )
    assert artifact["step_no"] == 1


def test_provider_cannot_override_review_identity(ctrl):
    to_review_received(ctrl)
    stub_runner(ctrl, {**REVIEW_OK, "step_no": 999})
    assert ctrl.run_codex_review()["step_no"] == 1
    assert ctrl.db.latest_codex_review(1)["step_no"] == 1


# --------------------------------------------------------------------------- #
# 18-20  Retryable review outage: deferred, no attempt consumed, report kept
# --------------------------------------------------------------------------- #


def test_retryable_review_error_defers_to_review_retryable(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_runner(ctrl, category="quota", retryable=True, returncode=1)

    result = ctrl.run_codex_review()

    assert result["deferred"] is True
    assert result["category"] == "quota"  # legacy category preserved
    assert result["provider"] == "test-provider"  # legacy label preserved
    assert result["returncode"] == 1
    assert ctrl.db.get_step(1).state == "CODEX_REVIEW_RETRYABLE"
    assert ctrl.db.has_codex_review(1, 3) is False

    # The provider-neutral audit records the CANONICAL category.
    error = json.loads(events(ctrl, SUPERVISOR_ERROR_EVENT)[0]["details"])
    assert error["operation"] == "REVIEW"
    assert error["provider"] == PROVIDER_CODEX_LEGACY
    assert error["retryable"] is True
    assert error["category"] == "quota_with_reset"
    assert error["step_no"] == 1
    assert error["attempt"] == 3


def test_retryable_review_error_does_not_increment_attempt(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_runner(ctrl)
    ctrl.run_codex_review()
    assert ctrl.db.get_step(1).attempt == 3


def test_retryable_review_error_preserves_accepted_report(ctrl):
    to_review_received(ctrl, attempt=3)
    before = ctrl.db.cline_report_for_attempt(1, 3)
    fail_runner(ctrl)
    ctrl.run_codex_review()
    after = ctrl.db.cline_report_for_attempt(1, 3)
    assert before is not None and after is not None
    assert after[1] == before[1]  # stored report hash unchanged


def test_non_retryable_review_error_still_fails(ctrl):
    to_review_received(ctrl)
    fail_runner(ctrl, category="contract", retryable=False)

    with pytest.raises(wt.CodexRunError):
        ctrl.run_codex_review()

    assert ctrl.db.get_step(1).state == "FAILED"


# --------------------------------------------------------------------------- #
# 21-24  Verdict transitions (unchanged)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "verdict,expected_state",
    [
        ("APPROVE", "REVIEW_APPROVED"),
        ("REVISE", "REVISE"),
        ("BLOCKED", "BLOCKED"),
        ("ARCHITECTURE_DECISION_REQUIRED", "BLOCKED"),
    ],
)
def test_review_verdict_transitions_unchanged(ctrl, verdict, expected_state):
    to_review_received(ctrl)
    stub_runner(ctrl, {**REVIEW_OK, "verdict": verdict})

    result = ctrl.run_codex_review()

    assert result["verdict"] == verdict
    assert ctrl.db.get_step(1).state == expected_state


# --------------------------------------------------------------------------- #
# 25  No SQLite schema change
# --------------------------------------------------------------------------- #


def test_no_sqlite_schema_change(ctrl):
    tables = {
        row[0]
        for row in ctrl.db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if not row[0].startswith("sqlite_")
    }
    assert tables == EXPECTED_TABLES

    columns = {
        row[1] for row in ctrl.db.db.execute("PRAGMA table_info(steps)").fetchall()
    }
    assert columns == EXPECTED_STEP_COLUMNS


def test_existing_database_still_opens(tmp_path):
    """A database written by the pre-M4 code must open unchanged."""
    first = wt.TandemController(tmp_path)
    first.db.import_plan(PLAN_DEF)
    stub_runner(first, PLAN_OK)
    first.run_codex_plan()
    stored = first.db.latest_codex_plan(1)
    first.close()

    second = wt.TandemController(tmp_path)
    try:
        assert second.db.latest_codex_plan(1) == stored
        assert second.db.get_step(1).state == "PLAN_READY"
    finally:
        second.close()


# --------------------------------------------------------------------------- #
# 26-28  The legacy Codex path is untouched
# --------------------------------------------------------------------------- #


def test_check_codex_still_available(tmp_path, monkeypatch):
    controller = wt.TandemController(tmp_path)
    try:
        monkeypatch.setattr(
            controller.codex, "available", lambda: (True, "codex-cli 0.157.1")
        )
        ok, info = controller.codex.available()
        assert ok is True and "codex" in info
        # The GUI entry points are unchanged.
        for name in (
            "check_codex",
            "run_codex_plan",
            "run_codex_review",
            "retry_codex_review",
        ):
            assert hasattr(wt.TandemGUI, name), name
    finally:
        controller.db.close()


def test_codex_runner_still_directly_available(ctrl):
    assert isinstance(ctrl.codex, wt.CodexRunner)


def test_adapter_wraps_the_same_runner_instance(ctrl):
    adapter = ctrl.supervisor_router.adapter_for(PROVIDER_CODEX_LEGACY)
    assert adapter is ctrl.supervisor_adapter
    assert adapter._runner is ctrl.codex  # no second CodexRunner instance
    assert ctrl.supervisor_router.registered_providers() == (PROVIDER_CODEX_LEGACY,)


def test_read_only_supervisor_helpers(ctrl):
    assert ctrl.supervisor_label() == PROVIDER_CODEX_LEGACY
    assert set(ctrl.supervisor_health()) == {PROVIDER_CODEX_LEGACY}


# --------------------------------------------------------------------------- #
# 29  Artifact paths remain compatible
# --------------------------------------------------------------------------- #


def test_legacy_plan_artifact_paths_preserved(ctrl):
    captured = {}
    stub_runner(ctrl, PLAN_OK, capture=captured)
    ctrl.run_codex_plan()

    workdir = ctrl.db.workdir
    assert captured["schema_path"] == workdir / "context" / "codex_plan.schema.json"
    assert captured["output_path"] == workdir / "from_codex" / "step_001_plan.json"
    assert captured["trace_path"] == workdir / "logs" / "step_001_codex_plan_trace.json"
    assert captured["reasoning_effort"] == "low"

    for parts in (
        ("context", "codex_plan.schema.json"),
        ("context", "step_001_codex_plan_context.json"),
        ("to_codex", "step_001_plan_request.json"),
        ("from_codex", "step_001_plan.json"),
    ):
        assert workdir.joinpath(*parts).is_file(), parts


def test_legacy_review_artifact_paths_preserved(ctrl):
    to_review_received(ctrl)
    captured = {}
    stub_runner(ctrl, REVIEW_OK, capture=captured)
    ctrl.run_codex_review()

    workdir = ctrl.db.workdir
    assert captured["schema_path"] == workdir / "context" / "codex_review.schema.json"
    assert captured["output_path"] == workdir / "from_codex" / "step_001_review.json"
    assert captured["trace_path"] == workdir / "logs" / "step_001_codex_review_trace.json"

    for parts in (
        ("context", "codex_review.schema.json"),
        ("context", "step_001_codex_review_context.json"),
        ("to_codex", "step_001_review_request.json"),
        ("from_codex", "step_001_review.json"),
    ):
        assert workdir.joinpath(*parts).is_file(), parts


# --------------------------------------------------------------------------- #
# 30-31  Unrelated projects are untouched
# --------------------------------------------------------------------------- #


def test_no_architecture_assistant_or_mini_worker_files_added():
    root = pathlib.Path(__file__).resolve().parent
    names = [entry.name.lower() for entry in root.iterdir()]
    for forbidden in ("architecture_assistant", "mini_worker", "miniworker"):
        assert not any(forbidden in name for name in names), forbidden
    # The Mini Worker integration constants are unchanged.
    assert wt.MINI_DIR == ".mini_build"
    assert wt.MINI_DB_NAME == "build_state.db"


def test_no_live_codex_process_or_network(ctrl, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("live process/network access attempted")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(socket, "create_connection", boom)

    stub_runner(ctrl, PLAN_OK)
    assert ctrl.run_codex_plan()["decision"] == "PLAN_READY"


# --------------------------------------------------------------------------- #
# 16  PARITY GATE: the router path must equal the pre-M4 direct path
# --------------------------------------------------------------------------- #


def _controller(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    controller = wt.TandemController(root)
    controller.db.import_plan(PLAN_DEF)
    return controller


def _old_direct_plan(controller):
    """Faithful re-implementation of the PRE-M4 controller PLAN sequence."""
    step = controller.db.current_step()
    attempt = controller.db.increment_attempt(step.step_no)
    controller.db.transition(step.step_no, "CODEX_PLAN_RUNNING", f"attempt={attempt}")
    step = controller.db.get_step(step.step_no)
    effort = controller._codex_reasoning_effort(step)
    output_path = controller.db.from_codex / f"step_{step.step_no:03d}_plan.json"
    trace_path = controller.db.logs / f"step_{step.step_no:03d}_codex_plan_trace.json"
    result = controller.codex.run_structured(
        "PLAN parity prompt",
        controller.db.context_dir / "codex_plan.schema.json",
        output_path,
        trace_path,
        reasoning_effort=effort,
    )
    result["step_no"] = step.step_no
    wt.write_json_atomic(output_path, result)
    controller.db.save_codex_plan(step.step_no, attempt, result)
    controller.db.transition(step.step_no, "PLAN_READY", "Codex plan ready.")
    return result


def test_parity_plan_payload_router_vs_direct_codex(tmp_path):
    fixture = {**PLAN_OK, "step_no": 999}  # the model echoes a wrong identity

    old = _controller(tmp_path, "old_plan")
    try:
        stub_runner(old, fixture)
        old_result = _old_direct_plan(old)
        old_stored = old.db.latest_codex_plan(1)
        old_artifact = (old.db.from_codex / "step_001_plan.json").read_text("utf-8")
        old_state = old.db.get_step(1).state
    finally:
        old.close()

    new = _controller(tmp_path, "new_plan")
    try:
        stub_runner(new, fixture)
        new_result = new.run_codex_plan()
        new_stored = new.db.latest_codex_plan(1)
        new_artifact = (new.db.from_codex / "step_001_plan.json").read_text("utf-8")
        new_state = new.db.get_step(1).state
    finally:
        new.close()

    assert new_result == old_result
    assert new_stored == old_stored
    assert new_artifact == old_artifact
    assert new_state == old_state == "PLAN_READY"
    assert new_stored["step_no"] == 1


def _old_direct_review(controller):
    """Faithful re-implementation of the PRE-M4 controller REVIEW sequence."""
    step = controller.db.current_step()
    controller.db.transition(
        step.step_no, "CODEX_REVIEW_RUNNING", "Codex review started."
    )
    step = controller.db.get_step(step.step_no)
    effort = controller._codex_reasoning_effort(step)
    output_path = controller.db.from_codex / f"step_{step.step_no:03d}_review.json"
    trace_path = controller.db.logs / f"step_{step.step_no:03d}_codex_review_trace.json"
    review = controller.codex.run_structured(
        "REVIEW parity prompt",
        controller.db.context_dir / "codex_review.schema.json",
        output_path,
        trace_path,
        reasoning_effort=effort,
    )
    review["step_no"] = step.step_no
    wt.write_json_atomic(output_path, review)
    controller.db.save_codex_review(step.step_no, step.attempt, review)
    controller.db.transition(step.step_no, "REVIEW_APPROVED", "Approved.")
    return review


def test_parity_review_payload_router_vs_direct_codex(tmp_path):
    fixture = {**REVIEW_OK, "step_no": 999}

    old = _controller(tmp_path, "old_review")
    try:
        to_review_received(old, attempt=2)
        stub_runner(old, fixture)
        old_result = _old_direct_review(old)
        old_stored = old.db.latest_codex_review(1)
        old_artifact = (old.db.from_codex / "step_001_review.json").read_text("utf-8")
        old_state = old.db.get_step(1).state
    finally:
        old.close()

    new = _controller(tmp_path, "new_review")
    try:
        to_review_received(new, attempt=2)
        stub_runner(new, fixture)
        new_result = new.run_codex_review()
        new_stored = new.db.latest_codex_review(1)
        new_artifact = (new.db.from_codex / "step_001_review.json").read_text("utf-8")
        new_state = new.db.get_step(1).state
    finally:
        new.close()

    assert new_result == old_result
    assert new_stored == old_stored
    assert new_artifact == old_artifact
    assert new_state == old_state == "REVIEW_APPROVED"
    assert new_stored["step_no"] == 1


def test_parity_deferred_review_payload_router_vs_direct_codex(ctrl):
    """A retryable outage must defer with the same audit payload as before."""
    to_review_received(ctrl, attempt=3)
    fail_runner(ctrl, category="quota", retryable=True, returncode=1)

    result = ctrl.run_codex_review()

    assert result["deferred"] is True
    row = ctrl.db.db.execute(
        "SELECT details FROM events WHERE step_no=1 AND event_type=? ORDER BY id",
        (wt.CODEX_REVIEW_RETRYABLE_EVENT,),
    ).fetchone()
    payload = json.loads(row["details"])
    assert payload["provider"] == "test-provider"
    assert payload["returncode"] == 1
    assert payload["category"] == "quota"
    assert payload["retryable"] is True
    assert payload["attempt"] == 3
    assert payload["sanitized_error"]
    assert payload["report_hash"]
    assert payload["timestamp"]





