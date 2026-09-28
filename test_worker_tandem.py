"""Worker_tandem tests: controller-owned step identity (PLAN/REVIEW).

These tests pin the architecture decision:
    Python/FSM is authoritative for step identity.
    Codex must never choose or echo step_no.

The driver is a single script whose filename is not a valid module name
(``worker_tandem_v0.1.2.py``), so it is loaded by path with importlib.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

PLAN_DEF = {
    "plan_version": "test-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "STEP ONE",
            "description": "test step one",
            "risk": "LOW",
        },
        {
            "step_no": 2,
            "phase": "FOUNDATION",
            "title": "STEP TWO",
            "description": "test step two",
            "risk": "LOW",
        },
    ],
}

# Valid model-owned PLAN payloads (NO step_no: the schema no longer requires it).
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


@pytest.fixture
def ctrl(tmp_path):
    controller = wt.TandemController(tmp_path)
    controller.db.import_plan(PLAN_DEF)
    yield controller
    controller.db.close()


def stub_run_structured(controller, payload):
    """Replace the real codex call with a fixed, already-'validated' payload."""

    def fake(prompt, schema_path, output_path, trace_path, reasoning_effort=None):
        return dict(payload)

    controller.codex.run_structured = fake


def advance_to(controller, target_state):
    """Move the current step forward using only legal FSM transitions."""
    path = {
        "CODEX_PLAN_RUNNING": ["CODEX_PLAN_RUNNING"],
        "PLAN_READY": ["CODEX_PLAN_RUNNING", "PLAN_READY"],
        "WAITING_HUMAN_APPROVAL": [
            "CODEX_PLAN_RUNNING",
            "PLAN_READY",
            "WAITING_HUMAN_APPROVAL",
        ],
        "CLINE_DISPATCHED": [
            "CODEX_PLAN_RUNNING",
            "PLAN_READY",
            "WAITING_HUMAN_APPROVAL",
            "CLINE_DISPATCHED",
        ],
        "CLINE_REPORT_RECEIVED": [
            "CODEX_PLAN_RUNNING",
            "PLAN_READY",
            "WAITING_HUMAN_APPROVAL",
            "CLINE_DISPATCHED",
            "CLINE_REPORT_RECEIVED",
        ],
    }[target_state]
    for state in path:
        controller.db.transition(1, state)


def plan_output_path(controller):
    return controller.db.from_codex / "step_001_plan.json"


def review_output_path(controller):
    return controller.db.from_codex / "step_001_review.json"


# --------------------------------------------------------------------------- #
# 1. Schemas no longer require step_no
# --------------------------------------------------------------------------- #


def test_plan_schema_does_not_require_step_no():
    schema = wt.codex_plan_schema()
    assert "step_no" not in schema["properties"]
    assert "step_no" not in schema["required"]


def test_review_schema_does_not_require_step_no():
    schema = wt.codex_review_schema()
    assert "step_no" not in schema["properties"]
    assert "step_no" not in schema["required"]


def test_controller_writes_step_no_free_schemas(ctrl):
    plan_schema = json.loads(
        (ctrl.db.context_dir / "codex_plan.schema.json").read_text(encoding="utf-8")
    )
    review_schema = json.loads(
        (ctrl.db.context_dir / "codex_review.schema.json").read_text(encoding="utf-8")
    )
    assert "step_no" not in plan_schema["properties"]
    assert "step_no" not in review_schema["properties"]


# --------------------------------------------------------------------------- #
# 2. run_structured strips a model-provided step_no before validation
# --------------------------------------------------------------------------- #


def test_run_structured_strips_model_step_no(monkeypatch, tmp_path):
    controller = wt.TandemController(tmp_path)
    try:
        schema_path = controller.db.context_dir / "codex_plan.schema.json"
        output_path = plan_output_path(controller)
        trace_path = controller.db.logs / "step_001_codex_plan_trace.json"

        # The model tries to choose its own identity; step_no must be dropped.
        model_payload = dict(PLAN_OK)
        model_payload["step_no"] = 999
        model_json = json.dumps(model_payload)
        stdout = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "t"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "i0",
                            "type": "agent_message",
                            "text": model_json,
                        },
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {}}),
            ]
        ) + "\n"

        fake_cp = wt.subprocess.CompletedProcess(
            args=["codex", "exec"], returncode=0, stdout=stdout, stderr=""
        )
        monkeypatch.setattr(wt.subprocess, "run", lambda *a, **k: fake_cp)
        monkeypatch.setattr(
            wt.CodexRunner, "build_command", lambda self, *a, **k: ["codex", "exec"]
        )

        result = controller.codex.run_structured(
            "prompt", schema_path, output_path, trace_path, reasoning_effort="low"
        )

        # Model-provided step_no never survives run_structured.
        assert "step_no" not in result
        published = json.loads(output_path.read_text(encoding="utf-8"))
        assert "step_no" not in published
    finally:
        controller.db.close()



# --------------------------------------------------------------------------- #
# 3. PLAN: controller injects the authoritative step_no
# --------------------------------------------------------------------------- #


def test_plan_missing_model_step_no_is_accepted(ctrl):
    stub_run_structured(ctrl, PLAN_OK)  # no step_no at all

    result = ctrl.run_codex_plan()

    assert result["step_no"] == 1
    assert ctrl.db.latest_codex_plan(1)["step_no"] == 1
    assert ctrl.db.current_step().state == "PLAN_READY"
    # Canonical on-disk artifact carries the controller-owned step_no.
    published = json.loads(plan_output_path(ctrl).read_text(encoding="utf-8"))
    assert published["step_no"] == 1


def test_plan_wrong_model_step_no_is_ignored(ctrl):
    stub_run_structured(ctrl, {**PLAN_OK, "step_no": 999})

    result = ctrl.run_codex_plan()

    assert result["step_no"] == 1
    assert ctrl.db.latest_codex_plan(1)["step_no"] == 1
    assert ctrl.db.current_step().state == "PLAN_READY"
    published = json.loads(plan_output_path(ctrl).read_text(encoding="utf-8"))
    assert published["step_no"] == 1


# --------------------------------------------------------------------------- #
# 4. REVIEW: same rule
# --------------------------------------------------------------------------- #


def test_review_missing_model_step_no_is_accepted(ctrl):
    advance_to(ctrl, "CLINE_REPORT_RECEIVED")
    # An accepted Cline report for the current attempt is a review precondition.
    ctrl.db.save_cline_report(1, 0, {"step_no": 1, "status": "DONE", "summary": "x"})
    ctrl.context_builder.review_context = lambda step: {"step_no": step.step_no}
    stub_run_structured(ctrl, REVIEW_OK)  # no step_no

    result = ctrl.run_codex_review()

    assert result["step_no"] == 1
    assert ctrl.db.latest_codex_review(1)["step_no"] == 1
    assert ctrl.db.current_step().state == "REVIEW_APPROVED"
    published = json.loads(review_output_path(ctrl).read_text(encoding="utf-8"))
    assert published["step_no"] == 1


def test_review_wrong_model_step_no_is_ignored(ctrl):
    advance_to(ctrl, "CLINE_REPORT_RECEIVED")
    # An accepted Cline report for the current attempt is a review precondition.
    ctrl.db.save_cline_report(1, 0, {"step_no": 1, "status": "DONE", "summary": "x"})
    ctrl.context_builder.review_context = lambda step: {"step_no": step.step_no}
    stub_run_structured(ctrl, {**REVIEW_OK, "step_no": 999})

    result = ctrl.run_codex_review()

    assert result["step_no"] == 1
    assert ctrl.db.latest_codex_review(1)["step_no"] == 1
    assert ctrl.db.current_step().state == "REVIEW_APPROVED"
    published = json.loads(review_output_path(ctrl).read_text(encoding="utf-8"))
    assert published["step_no"] == 1


# --------------------------------------------------------------------------- #
# 5. Cline report step_no validation is unchanged
# --------------------------------------------------------------------------- #


def test_cline_report_step_no_mismatch_still_rejected(ctrl):
    advance_to(ctrl, "CLINE_DISPATCHED")
    report_path = ctrl.db.from_cline / "step_001_report.json"
    wt.write_json_atomic(
        report_path, {"step_no": 999, "status": "DONE", "summary": "x"}
    )

    with pytest.raises(RuntimeError, match="Cline report step_no mismatch"):
        ctrl.process_cline_report()



# --------------------------------------------------------------------------- #
# 6. FSM gates unchanged
# --------------------------------------------------------------------------- #

EXPECTED_ALLOWED_TRANSITIONS = {
    "PENDING": {"READY_FOR_CODEX_PLAN", "SKIPPED", "ABORTED"},
    "READY_FOR_CODEX_PLAN": {"CODEX_PLAN_RUNNING", "BLOCKED", "ABORTED"},
    "CODEX_PLAN_RUNNING": {"PLAN_READY", "BLOCKED", "FAILED"},
    "PLAN_READY": {"WAITING_HUMAN_APPROVAL", "BLOCKED", "ABORTED"},
    "WAITING_HUMAN_APPROVAL": {"CLINE_DISPATCHED", "BLOCKED", "ABORTED"},
    "CLINE_DISPATCHED": {"CLINE_REPORT_RECEIVED", "BLOCKED", "FAILED"},
    "CLINE_REPORT_RECEIVED": {"CODEX_REVIEW_RUNNING", "BLOCKED", "FAILED"},
    "CODEX_REVIEW_RUNNING": {
        "REVIEW_APPROVED",
        "REVISE",
        "BLOCKED",
        "CODEX_REVIEW_RETRYABLE",
        # The one explicit recovery edge (interrupted REVIEW). It is reachable
        # only through TandemController.recover_review_running_to_received.
        "CLINE_REPORT_RECEIVED",
        "FAILED",
    },
    "CODEX_REVIEW_RETRYABLE": {"CODEX_REVIEW_RUNNING", "BLOCKED", "ABORTED"},
    "REVIEW_APPROVED": {"VERIFIED", "BLOCKED", "ABORTED"},
    "REVISE": {"CLINE_DISPATCHED", "BLOCKED", "ABORTED"},
    "BLOCKED": {"READY_FOR_CODEX_PLAN", "ABORTED"},
    "FAILED": {"READY_FOR_CODEX_PLAN", "ABORTED"},
    "VERIFIED": set(),
    "SKIPPED": set(),
    "ABORTED": set(),
}


def test_fsm_transition_table_is_unchanged():
    assert wt.ALLOWED_TRANSITIONS == EXPECTED_ALLOWED_TRANSITIONS


def test_fsm_valid_and_terminal_states_are_unchanged():
    assert wt.TERMINAL_STATES == {"VERIFIED", "SKIPPED", "ABORTED"}
    assert wt.VALID_STATES == set(EXPECTED_ALLOWED_TRANSITIONS)


def test_recovery_transition_failed_to_ready_is_legal():
    assert "READY_FOR_CODEX_PLAN" in wt.ALLOWED_TRANSITIONS["FAILED"]


def test_recovery_transition_review_running_to_received_is_legal():
    """The ONE interrupted-REVIEW recovery edge is an explicit FSM transition."""
    assert "CLINE_REPORT_RECEIVED" in wt.ALLOWED_TRANSITIONS["CODEX_REVIEW_RUNNING"]
    # ... and it is not a generic "go back" door: only the normal dispatch and
    # the interrupted-REVIEW recovery may ever enter CLINE_REPORT_RECEIVED.
    for state, targets in wt.ALLOWED_TRANSITIONS.items():
        if "CLINE_REPORT_RECEIVED" in targets:
            assert state in {"CLINE_DISPATCHED", "CODEX_REVIEW_RUNNING"}, state


def test_illegal_transition_is_rejected(ctrl):
    with pytest.raises(RuntimeError, match="Illegal transition"):
        ctrl.db.transition(1, "VERIFIED")  # READY_FOR_CODEX_PLAN -> VERIFIED


def test_codex_plan_state_gate_is_unchanged(ctrl):
    advance_to(ctrl, "CODEX_PLAN_RUNNING")  # no longer READY_FOR_CODEX_PLAN

    with pytest.raises(RuntimeError, match="READY_FOR_CODEX_PLAN"):
        ctrl.run_codex_plan()


def test_codex_review_state_gate_is_unchanged(ctrl):
    # step 1 is READY_FOR_CODEX_PLAN, not CLINE_REPORT_RECEIVED
    with pytest.raises(RuntimeError, match="CLINE_REPORT_RECEIVED"):
        ctrl.run_codex_review()

