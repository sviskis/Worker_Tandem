"""Worker_tandem PLAN-context and canonical-plan sync tests.

Pins the bugfix contract:
    - Codex PLAN always receives the COMPLETE current step (never title-only);
    - the locked controller rules and only the necessary previous VERIFIED
      steps are included, bounded, with reference file NAMES only;
    - the current step description can be re-synced from the canonical
      Architecture Assistant BUILD_PLAN.json without losing execution history.

The driver filename is not a valid module name, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem_ctx", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_ctx"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()


CATEGORY_SPEC_DESCRIPTION = (
    "Mandatory per-category SPEC.md at categories/<category_id>/SPEC.md with an "
    "immutable category id and display name, a version and a SHA-256 content "
    "hash. SPEC.md is required before a category can be ENABLED, is referenced "
    "by the STEP 2 catalog, persisted by STEP 5 and used by STEP 7."
)


def _controller(tmp_path, steps, policy=None):
    controller = wt.TandemController(tmp_path)
    plan = {
        "plan_version": "ctx-test-1",
        "controller_policy": dict(policy or {}),
        "steps": steps,
    }
    controller.db.import_plan(plan)
    return controller


def _spec_fixture(tmp_path):
    return _controller(
        tmp_path,
        [
            {
                "step_no": 1,
                "phase": "FOUNDATION",
                "title": "CATEGORY SPEC",
                "description": CATEGORY_SPEC_DESCRIPTION,
                "risk": "HIGH",
                "requires_human": True,
            }
        ],
        policy={
            "one_step_at_a_time": True,
            "python_is_authoritative_controller": True,
        },
    )


def _four_steps():
    return [
        {
            "step_no": n,
            "phase": "FOUNDATION",
            "title": f"STEP {n}",
            "description": f"Description for step {n}.",
            "risk": "LOW",
        }
        for n in (1, 2, 3, 4)
    ]


def _verify_step(controller, step_no):
    for state in (
        "CODEX_PLAN_RUNNING",
        "PLAN_READY",
        "WAITING_HUMAN_APPROVAL",
        "CLINE_DISPATCHED",
        "CLINE_REPORT_RECEIVED",
        "CODEX_REVIEW_RUNNING",
        "REVIEW_APPROVED",
    ):
        controller.db.transition(step_no, state)
    controller.db.mark_verified_and_advance(step_no)


# --------------------------------------------------------------------------- #
# 1. The complete current step reaches Codex (never title-only)
# --------------------------------------------------------------------------- #


def test_plan_context_includes_full_current_step(tmp_path):
    controller = _spec_fixture(tmp_path)
    try:
        step = controller.db.current_step()
        ctx = controller.context_builder.project_context(step)
        assert ctx["step"]["step_no"] == 1
        assert ctx["step"]["phase"] == "FOUNDATION"
        assert ctx["step"]["title"] == "CATEGORY SPEC"
        assert ctx["step"]["description"] == CATEGORY_SPEC_DESCRIPTION
        assert ctx["step"]["risk"] == "HIGH"
        assert ctx["step"]["requires_human"] is True
        assert ctx["step"]["attempt"] == step.attempt
    finally:
        controller.db.close()


def test_category_spec_fixture_receives_task_requirements(tmp_path):
    controller = _spec_fixture(tmp_path)
    try:
        ctx = controller.context_builder.project_context(controller.db.current_step())
        sent = ctx["step"]["description"]
        for needle in (
            "SPEC.md",
            "categories/<category_id>",
            "immutable",
            "version",
            "SHA-256",
            "ENABLED",
            "STEP 2",
            "STEP 5",
            "STEP 7",
        ):
            assert needle in sent
    finally:
        controller.db.close()


def test_title_only_description_is_not_sufficient(tmp_path):
    assert wt.is_sufficient_step_description("CATEGORY SPEC", "CATEGORY SPEC") is False
    assert wt.is_sufficient_step_description("", "CATEGORY SPEC") is False
    assert wt.is_sufficient_step_description("   ", "CATEGORY SPEC") is False
    assert (
        wt.is_sufficient_step_description(CATEGORY_SPEC_DESCRIPTION, "CATEGORY SPEC")
        is True
    )

    controller = _controller(
        tmp_path,
        [
            {
                "step_no": 1,
                "phase": "FOUNDATION",
                "title": "CATEGORY SPEC",
                "description": "CATEGORY SPEC",
                "risk": "HIGH",
            }
        ],
    )
    try:
        with pytest.raises(ValueError, match="title-only"):
            controller.context_builder.project_context(controller.db.current_step())
    finally:
        controller.db.close()


def test_run_codex_plan_refuses_title_only_without_state_change(tmp_path):
    controller = _controller(
        tmp_path,
        [
            {
                "step_no": 1,
                "phase": "FOUNDATION",
                "title": "CATEGORY SPEC",
                "description": "CATEGORY SPEC",
                "risk": "HIGH",
            }
        ],
    )
    try:
        before = controller.db.get_step(1)
        with pytest.raises(RuntimeError, match="title-only"):
            controller.run_codex_plan()
        after = controller.db.get_step(1)
        assert after.state == before.state == "READY_FOR_CODEX_PLAN"
        assert after.attempt == before.attempt == 0
    finally:
        controller.db.close()


# --------------------------------------------------------------------------- #
# 2. Locked controller rules + bounded, necessary previous steps
# --------------------------------------------------------------------------- #


def test_plan_context_includes_controller_policy(tmp_path):
    controller = _spec_fixture(tmp_path)
    try:
        ctx = controller.context_builder.project_context(controller.db.current_step())
        assert ctx["controller_policy"]["one_step_at_a_time"] is True
        assert ctx["controller_policy"]["python_is_authoritative_controller"] is True
    finally:
        controller.db.close()


def test_previous_steps_are_only_verified_and_ordered(tmp_path):
    controller = _controller(tmp_path, _four_steps())
    try:
        _verify_step(controller, 1)
        _verify_step(controller, 2)
        ctx = controller.context_builder.project_context(controller.db.current_step())
        prev_nos = [s["step_no"] for s in ctx["previous_steps"]]
        assert prev_nos == [1, 2]
        assert all(s["step_no"] < 3 for s in ctx["previous_steps"])
    finally:
        controller.db.close()


def test_plan_context_is_bounded(tmp_path):
    steps = [
        {
            "step_no": n,
            "phase": "FOUNDATION",
            "title": f"STEP {n}",
            "description": "X" * 5000,
            "risk": "LOW",
        }
        for n in range(1, 6)
    ]
    controller = _controller(tmp_path, steps)
    try:
        for n in (1, 2, 3, 4):
            _verify_step(controller, n)
        ctx = controller.context_builder.project_context(controller.db.current_step())
        assert len(ctx["previous_steps"]) <= wt.MAX_PREVIOUS_STEPS
        assert all(
            len(s["description"]) <= wt.MAX_PREVIOUS_STEP_DESCRIPTION_CHARS + 40
            for s in ctx["previous_steps"]
        )
        assert len(json.dumps(ctx, ensure_ascii=False)) <= wt.MAX_PLAN_CONTEXT_CHARS
    finally:
        controller.db.close()


def test_plan_context_sends_file_names_not_bodies(tmp_path):
    controller = _spec_fixture(tmp_path)
    try:
        sentinel = "SENTINEL-DOCUMENT-BODY-MUST-NOT-BE-SENT"
        (tmp_path / "ARCHITECTURE.md").write_text(sentinel, encoding="utf-8")
        ctx = controller.context_builder.project_context(controller.db.current_step())
        assert "ARCHITECTURE.md" in ctx["reference_files"]
        payload = json.dumps(ctx, ensure_ascii=False)
        assert sentinel not in payload
        assert "git_diff" not in ctx
        assert "approved_plan" not in ctx
        assert "cline_report" not in ctx
    finally:
        controller.db.close()


# --------------------------------------------------------------------------- #
# 3. Canonical-plan description sync preserves execution history
# --------------------------------------------------------------------------- #


def test_sync_step_description_preserves_history(tmp_path):
    controller = _controller(tmp_path, _four_steps())
    try:
        controller.db.increment_attempt(1)
        controller.db.save_codex_plan(
            1, 1, {"decision": "PLAN_READY", "summary": "keep me"}
        )
        before = controller.db.get_step(1)

        plan = {
            "plan_version": "canonical-9",
            "steps": [
                {
                    "step_no": n,
                    "phase": "FOUNDATION",
                    "title": f"STEP {n}",
                    "description": (
                        CATEGORY_SPEC_DESCRIPTION
                        if n == 1
                        else f"Description for step {n}."
                    ),
                    "risk": "LOW",
                }
                for n in (1, 2, 3, 4)
            ],
        }
        wt.write_json_atomic(tmp_path / wt.CANONICAL_PLAN_NAME, plan)

        result = controller.sync_current_step_description_from_plan(
            actor="tester", reason="unit-test"
        )

        after = controller.db.get_step(1)
        assert result["step_no"] == 1
        assert result["changed"] is True
        assert after.description == CATEGORY_SPEC_DESCRIPTION
        assert after.attempt == before.attempt
        assert after.state == before.state
        assert controller.db.latest_codex_plan(1)["summary"] == "keep me"

        rows = controller.db.db.execute(
            "SELECT details FROM events WHERE event_type='STEP_DESCRIPTION_SYNCED'"
        ).fetchall()
        assert len(rows) == 1
        details = json.loads(rows[-1]["details"])
        assert details["step_no"] == 1
        assert details["old_description_hash"] == wt.sha256_text(before.description)
        assert details["new_description_hash"] == wt.sha256_text(after.description)
        assert details["plan_hash"] == result["plan_hash"]
        assert details["plan_version"] == "canonical-9"
        assert details["actor"] == "tester"
        assert details["reason"] == "unit-test"
    finally:
        controller.db.close()


def test_sync_preserves_verified_and_pending_states(tmp_path):
    controller = _controller(tmp_path, _four_steps())
    try:
        _verify_step(controller, 1)
        _verify_step(controller, 2)
        before = {s.step_no: s.state for s in controller.db.list_steps()}
        assert before[1] == "VERIFIED"
        assert before[2] == "VERIFIED"
        assert before[3] == "READY_FOR_CODEX_PLAN"
        assert before[4] == "PENDING"

        plan = {
            "plan_version": "canonical-10",
            "steps": [
                {
                    "step_no": n,
                    "phase": "FOUNDATION",
                    "title": f"STEP {n}",
                    "description": (
                        "Full specification for step 3."
                        if n == 3
                        else f"Description for step {n}."
                    ),
                    "risk": "LOW",
                }
                for n in (1, 2, 3, 4)
            ],
        }
        wt.write_json_atomic(tmp_path / wt.CANONICAL_PLAN_NAME, plan)

        result = controller.sync_current_step_description_from_plan(reason="states")
        assert result["step_no"] == 3
        after = {s.step_no: s.state for s in controller.db.list_steps()}
        assert after == before
    finally:
        controller.db.close()


def test_sync_refuses_missing_canonical_plan(tmp_path):
    controller = _controller(tmp_path, _four_steps())
    try:
        with pytest.raises(FileNotFoundError):
            controller.sync_current_step_description_from_plan()
    finally:
        controller.db.close()


# --------------------------------------------------------------------------- #
# 4. FSM unchanged
# --------------------------------------------------------------------------- #


def test_fsm_unchanged_by_context_fix():
    assert wt.ALLOWED_TRANSITIONS["BLOCKED"] == {"READY_FOR_CODEX_PLAN", "ABORTED"}
    assert wt.TERMINAL_STATES == {"VERIFIED", "SKIPPED", "ABORTED"}
    assert wt.VALID_STATES == set(wt.ALLOWED_TRANSITIONS)


# --------------------------------------------------------------------------- #
# 5. The prompt must travel on stdin, never as an argv element
# --------------------------------------------------------------------------- #


def test_prompt_is_streamed_on_stdin_not_argv(monkeypatch, tmp_path):
    """Regression: a multi-line prompt must never be an argv element.

    On Windows the Codex CLI is usually an npm `.CMD` shim that re-parses argv
    through cmd.exe; a multi-line argument is truncated at the first newline,
    so the planner would only ever receive the first line ("no task provided").
    """
    controller = wt.TandemController(tmp_path)
    try:
        schema_path = controller.db.context_dir / "codex_plan.schema.json"
        output_path = controller.db.from_codex / "step_001_plan.json"
        trace_path = controller.db.logs / "trace.json"
        multiline = "line one\nline two\nline three\nReturn JSON conforming to the schema."

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["input"] = kwargs.get("input")
            model_json = json.dumps(
                {
                    "decision": "PLAN_READY",
                    "goal": "g",
                    "files": [],
                    "changes": [],
                    "checks": [],
                    "architecture_risk": "none",
                    "open_questions": [],
                    "rollback": [],
                    "summary": "s",
                }
            )
            stdout = (
                "\n".join(
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
                )
                + "\n"
            )
            return wt.subprocess.CompletedProcess(
                args=list(cmd), returncode=0, stdout=stdout, stderr=""
            )

        monkeypatch.setattr(wt.subprocess, "run", fake_run)
        monkeypatch.setattr(wt.CodexRunner, "resolved_executable", lambda self: "codex")
        monkeypatch.setattr(wt.CodexRunner, "_is_git_repo", lambda self: True)

        result = controller.codex.run_structured(
            multiline, schema_path, output_path, trace_path, reasoning_effort="low"
        )

        assert result["decision"] == "PLAN_READY"
        assert captured["input"] == multiline
        assert captured["cmd"][-1] == wt.CodexRunner.STDIN_PROMPT == "-"
        assert multiline not in captured["cmd"]
        assert all("\n" not in part for part in captured["cmd"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace["prompt_via"] == "stdin"
        assert trace["prompt"] == multiline
        assert trace["prompt_chars"] == len(multiline)
    finally:
        controller.db.close()



