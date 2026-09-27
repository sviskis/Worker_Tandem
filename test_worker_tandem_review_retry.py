"""Worker_tandem Codex REVIEW retry / recovery tests.

Pins the contract:
    - a retryable, external Codex failure during REVIEW is NOT a project
      failure: it parks the step in CODEX_REVIEW_RETRYABLE and consumes no
      attempt, while the accepted Cline report and all history survive;
    - a human can retry the review from the same attempt (no PLAN, no Cline);
    - the one-time FAILED -> CLINE_REPORT_RECEIVED recovery is narrowly scoped,
      audited, idempotence-guarded and cannot touch the attempt or the report;
    - classification fails closed: only certain provider outages are retryable.

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
    spec = importlib.util.spec_from_file_location("worker_tandem_retry", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_retry"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()

PLAN_DEF = {
    "plan_version": "retry-test-1",
    "steps": [
        {"step_no": 1, "phase": "FOUNDATION", "title": "ONE", "description": "d1", "risk": "LOW"},
    ],
}

DISPATCH_PATH = (
    "CODEX_PLAN_RUNNING",
    "PLAN_READY",
    "WAITING_HUMAN_APPROVAL",
    "CLINE_DISPATCHED",
)

#: The real provider message that burned STEP 3 (usage limit WITH reset time).
QUOTA_MESSAGE = (
    "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), "
    "visit https://chatgpt.com/codex/settings/usage to purchase more credits or try "
    "again at Sep 27th, 2026 1:29 AM."
)


def report_ok(step_no=1, status="DONE"):
    return {
        "step_no": step_no,
        "status": status,
        "files_created": ["a.py"],
        "files_changed": ["b.py"],
        "files_deleted": [],
        "tests": {"passed": 5, "failed": 0, "command": "pytest -q"},
        "dependencies_added": [],
        "architecture_questions": [],
        "issues": [],
        "summary": "done",
    }


@pytest.fixture
def ctrl(tmp_path):
    controller = wt.TandemController(tmp_path)
    controller.db.import_plan(PLAN_DEF)
    yield controller
    controller.db.close()


def to_cline_dispatched(ctrl, *, attempt=None):
    for state in DISPATCH_PATH:
        ctrl.db.transition(1, state)
    if attempt is not None:
        while ctrl.db.get_step(1).attempt < attempt:
            ctrl.db.increment_attempt(1)


def accept_report(ctrl, *, status="DONE"):
    """Write the accepted Cline report and process it (-> CLINE_REPORT_RECEIVED)."""
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok(status=status))
    return ctrl.process_cline_report()


def to_review_received(ctrl, *, attempt=None):
    to_cline_dispatched(ctrl, attempt=attempt)
    accept_report(ctrl)


def stub_review(ctrl, payload):
    ctrl.codex.run_structured = lambda *args, **kwargs: dict(payload)


def fail_review(
    ctrl,
    *,
    category="quota",
    retryable=True,
    returncode=1,
    sanitized=QUOTA_MESSAGE,
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

    ctrl.codex.run_structured = boom


def force_state(ctrl, state, step_no=1):
    """Test-only: fabricate a legacy state (e.g. the pre-fix FAILED)."""
    with ctrl.db.db:
        ctrl.db.db.execute(
            "UPDATE steps SET state=? WHERE step_no=?", (state, step_no)
        )


def events(ctrl, event_type, step_no=1):
    rows = ctrl.db.db.execute(
        "SELECT details, created_at FROM events WHERE step_no=? AND event_type=? "
        "ORDER BY id",
        (step_no, event_type),
    ).fetchall()
    return [dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Failure classification (fail closed; structured JSONL preferred)
# --------------------------------------------------------------------------- #


def test_usage_limit_with_reset_is_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": QUOTA_MESSAGE}], returncode=1
    )
    assert failure.category == "quota"
    assert failure.retryable is True
    assert failure.source == "jsonl"


def test_turn_failed_usage_limit_is_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "turn.failed", "error": {"message": QUOTA_MESSAGE}}],
        returncode=1,
    )
    assert failure.category == "quota" and failure.retryable is True


def test_quota_temporarily_exhausted_is_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "Quota temporarily exhausted; please retry."}]
    )
    assert failure.category == "quota" and failure.retryable is True


def test_rate_limit_is_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "Rate limit reached for requests"}]
    )
    assert failure.category == "rate_limit" and failure.retryable is True


def test_timeout_is_retryable():
    failure = wt.classify_codex_failure(timed_out=True, returncode=None)
    assert failure.category == "timeout" and failure.retryable is True


def test_timeout_text_is_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "request timed out"}]
    )
    assert failure.category == "timeout" and failure.retryable is True


def test_transport_only_in_stderr_is_retryable():
    failure = wt.classify_codex_failure(
        stderr=(
            "ERROR rmcp::transport::worker: worker quit with fatal: Transport "
            "channel closed ... 'error decoding response body'"
        ),
        returncode=1,
    )
    assert failure.category == "transport"
    assert failure.retryable is True
    assert failure.source == "stderr"


def test_temporary_network_failure_is_retryable():
    failure = wt.classify_codex_failure(
        stderr="curl: (6) Could not resolve host: api.openai.com", returncode=1
    )
    assert failure.category == "network" and failure.retryable is True


def test_provider_unavailable_is_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "503 Service Unavailable"}]
    )
    assert failure.category == "provider_unavailable" and failure.retryable is True


def test_billing_disabled_is_non_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "Billing hard limit has been reached"}]
    )
    assert failure.category == "billing" and failure.retryable is False


def test_payment_required_is_non_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "402 Payment Required"}]
    )
    assert failure.retryable is False


def test_insufficient_balance_without_reset_is_non_retryable():
    failure = wt.classify_codex_failure(
        events=[
            {
                "type": "error",
                "message": (
                    "insufficient_quota: you exceeded your current quota, please "
                    "check your plan and billing details"
                ),
            }
        ]
    )
    assert failure.retryable is False


def test_authentication_failure_is_non_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "401 Unauthorized: invalid api key"}]
    )
    assert failure.category == "auth" and failure.retryable is False


def test_contract_error_is_non_retryable():
    failure = wt.classify_codex_failure(
        events=[
            {"type": "error", "message": "invalid_request_error: output schema mismatch"}
        ]
    )
    assert failure.category == "contract" and failure.retryable is False


def test_unknown_failure_is_non_retryable():
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": "something inexplicable happened"}],
        returncode=1,
    )
    assert failure.category == "unknown" and failure.retryable is False


def test_jsonl_text_is_preferred_over_stderr_markers():
    """A usage-limit JSONL error must win over transport noise on stderr."""
    failure = wt.classify_codex_failure(
        events=[{"type": "error", "message": QUOTA_MESSAGE}],
        stderr="rmcp error decoding response body transport channel closed",
        returncode=1,
    )
    assert failure.category == "quota" and failure.retryable is True
    assert failure.source == "jsonl"


# --------------------------------------------------------------------------- #
# Retryable review failure -> CODEX_REVIEW_RETRYABLE (no attempt consumed)
# --------------------------------------------------------------------------- #


def test_done_report_at_max_attempts_is_accepted(ctrl):
    to_cline_dispatched(ctrl, attempt=3)
    assert ctrl.db.get_step(1).attempt == 3 == ctrl.db.get_step(1).max_attempts

    report = accept_report(ctrl)

    assert report["status"] == "DONE"
    step = ctrl.db.get_step(1)
    assert step.state == "CLINE_REPORT_RECEIVED"
    assert step.attempt == 3  # a report never charges an attempt
    assert ctrl.db.latest_cline_report(1)["status"] == "DONE"


def test_quota_failure_during_review_does_not_consume_attempt(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_review(ctrl, category="quota", retryable=True)

    result = ctrl.run_codex_review()

    step = ctrl.db.get_step(1)
    assert result["deferred"] is True
    assert result["category"] == "quota"
    assert step.state == "CODEX_REVIEW_RETRYABLE"
    assert step.attempt == 3
    assert ctrl.db.has_codex_review(1, 3) is False


def test_timeout_during_review_does_not_consume_attempt(ctrl):
    to_review_received(ctrl)
    fail_review(ctrl, category="timeout", retryable=True, returncode=None)

    result = ctrl.run_codex_review()

    assert result["deferred"] is True
    assert ctrl.db.get_step(1).state == "CODEX_REVIEW_RETRYABLE"
    assert ctrl.db.get_step(1).attempt == 0


def test_transport_failure_during_review_does_not_consume_attempt(ctrl):
    to_review_received(ctrl)
    fail_review(ctrl, category="transport", retryable=True)

    result = ctrl.run_codex_review()

    assert result["deferred"] is True
    assert ctrl.db.get_step(1).state == "CODEX_REVIEW_RETRYABLE"
    assert ctrl.db.get_step(1).attempt == 0


def test_non_retryable_review_failure_still_fails(ctrl):
    to_review_received(ctrl)
    fail_review(ctrl, category="contract", retryable=False)

    with pytest.raises(wt.CodexRunError):
        ctrl.run_codex_review()

    assert ctrl.db.get_step(1).state == "FAILED"


def test_accepted_cline_report_remains_stored(ctrl):
    to_review_received(ctrl, attempt=3)
    before = ctrl.db.cline_report_for_attempt(1, 3)
    fail_review(ctrl)
    ctrl.run_codex_review()

    after = ctrl.db.cline_report_for_attempt(1, 3)
    assert before is not None and after is not None
    assert after[1] == before[1]  # stored report hash unchanged


def test_retryable_review_failure_does_not_overwrite_report(ctrl):
    to_review_received(ctrl, attempt=3)
    count_before = ctrl.db.db.execute(
        "SELECT COUNT(*) FROM cline_reports WHERE step_no=1"
    ).fetchone()[0]
    fail_review(ctrl)
    ctrl.run_codex_review()

    count_after = ctrl.db.db.execute(
        "SELECT COUNT(*) FROM cline_reports WHERE step_no=1"
    ).fetchone()[0]
    assert count_after == count_before
    assert ctrl.db.cline_report_for_attempt(1, 3) is not None


def test_retry_never_calls_increment_attempt(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_review(ctrl)
    ctrl.run_codex_review()
    assert ctrl.db.get_step(1).state == "CODEX_REVIEW_RETRYABLE"

    calls = []
    real = ctrl.db.increment_attempt

    def spy(step_no):
        calls.append(step_no)
        return real(step_no)

    ctrl.db.increment_attempt = spy
    stub_review(ctrl, {"verdict": "APPROVE", "summary": "ok"})

    ctrl.retry_codex_review()

    assert calls == []
    assert ctrl.db.get_step(1).attempt == 3


def test_retry_review_works_from_same_attempt(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_review(ctrl)
    ctrl.run_codex_review()
    assert ctrl.db.get_step(1).state == "CODEX_REVIEW_RETRYABLE"

    stub_review(ctrl, {"verdict": "APPROVE", "summary": "retry ok"})
    review = ctrl.retry_codex_review()

    assert review["verdict"] == "APPROVE"
    assert ctrl.db.get_step(1).state == "REVIEW_APPROVED"
    assert ctrl.db.get_step(1).attempt == 3
    assert ctrl.db.has_codex_review(1, 3) is True


def test_successful_retry_produces_approve(ctrl):
    to_review_received(ctrl)
    fail_review(ctrl, category="rate_limit", retryable=True)
    ctrl.run_codex_review()

    stub_review(ctrl, {"verdict": "APPROVE", "summary": "ok"})
    result = ctrl.retry_codex_review()

    assert result["verdict"] == "APPROVE"
    assert ctrl.db.latest_codex_review(1)["verdict"] == "APPROVE"


def test_retry_can_produce_revise(ctrl):
    to_review_received(ctrl)
    fail_review(ctrl)
    ctrl.run_codex_review()

    stub_review(ctrl, {"verdict": "REVISE", "summary": "needs work"})
    result = ctrl.retry_codex_review()

    assert result["verdict"] == "REVISE"
    assert ctrl.db.get_step(1).state == "REVISE"


def test_retry_cannot_run_without_accepted_report(ctrl):
    to_cline_dispatched(ctrl)
    for state in ("CLINE_REPORT_RECEIVED", "CODEX_REVIEW_RUNNING", "CODEX_REVIEW_RETRYABLE"):
        ctrl.db.transition(1, state)
    assert ctrl.db.cline_report_for_attempt(1, ctrl.db.get_step(1).attempt) is None

    with pytest.raises(RuntimeError, match="no accepted Cline report"):
        ctrl.retry_codex_review()
    assert ctrl.db.get_step(1).state == "CODEX_REVIEW_RETRYABLE"


def test_retry_writes_its_own_audit_event(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_review(ctrl)
    ctrl.run_codex_review()
    stub_review(ctrl, {"verdict": "APPROVE", "summary": "ok"})
    ctrl.retry_codex_review()

    rows = events(ctrl, wt.CODEX_REVIEW_RETRY_EVENT)
    assert len(rows) == 1
    payload = json.loads(rows[0]["details"])
    assert payload["attempt"] == 3
    assert payload["report_hash"]
    assert payload["timestamp"]


def test_audit_stores_real_external_failure_cause(ctrl):
    to_review_received(ctrl, attempt=3)
    fail_review(ctrl, category="quota", retryable=True, returncode=1)
    ctrl.run_codex_review()

    rows = events(ctrl, wt.CODEX_REVIEW_RETRYABLE_EVENT)
    assert len(rows) == 1
    payload = json.loads(rows[0]["details"])
    assert payload["category"] == "quota"
    assert payload["retryable"] is True
    assert payload["returncode"] == 1
    assert payload["provider"] == "test-provider"
    assert payload["attempt"] == 3
    assert payload["report_hash"]
    assert payload["sanitized_error"]
    assert payload["timestamp"]


def test_quota_diagnostic_leads_with_usage_limit_not_mcp_hint(ctrl):
    """The MCP/model-config hint must never be the primary quota cause."""
    message = ctrl.codex._failure_report(
        1,
        [{"type": "error", "message": QUOTA_MESSAGE}],
        "",
        "rmcp error decoding response body transport channel closed",
        ctrl.db.logs / "trace.json",
        "codex exec exited with code 1.",
        category="quota",
        retryable=True,
        primary_error=QUOTA_MESSAGE,
    )

    assert QUOTA_MESSAGE in message
    assert "failure category: quota (retryable=true)" in message
    assert "MCP" not in message
    assert "config.toml" not in message


def test_transport_diagnostic_keeps_a_provider_side_hint(ctrl):
    message = ctrl.codex._failure_report(
        1,
        [],
        "",
        "rmcp error decoding response body",
        ctrl.db.logs / "trace.json",
        "codex exec exited with code 1.",
        category="transport",
        retryable=True,
    )
    assert "MCP transport failure" in message
    assert "provider-side condition" in message


# --------------------------------------------------------------------------- #
# One-time FAILED -> CLINE_REPORT_RECEIVED recovery (narrow, audited, guarded)
# --------------------------------------------------------------------------- #


def legacy_failed_step(ctrl, *, attempt=3):
    """Reproduce the exact legacy STEP 3 shape: FAILED + accepted report, no review."""
    to_review_received(ctrl, attempt=attempt)
    force_state(ctrl, "FAILED")
    return ctrl.db.get_step(1)


def test_legacy_failed_step_qualifies_for_recovery(ctrl):
    legacy_failed_step(ctrl, attempt=3)

    assert ctrl.review_recovery_qualified() is True


def test_recovery_restores_cline_report_received_without_touching_attempt(ctrl):
    legacy_failed_step(ctrl, attempt=3)
    before = ctrl.db.cline_report_for_attempt(1, 3)

    result = ctrl.recover_failed_review_to_received(
        actor="operator", reason="quota outage"
    )

    step = ctrl.db.get_step(1)
    assert step.state == "CLINE_REPORT_RECEIVED"
    assert step.attempt == 3
    assert result["attempt"] == 3
    assert result["report_hash"] == before[1]
    assert ctrl.db.cline_report_for_attempt(1, 3)[1] == before[1]


def test_recovery_appends_explicit_audit_event(ctrl):
    legacy_failed_step(ctrl, attempt=3)

    ctrl.recover_failed_review_to_received(actor="operator", reason="quota outage")

    rows = events(ctrl, wt.RECOVERY_REVIEW_RETRY_EVENT)
    assert len(rows) == 1
    payload = json.loads(rows[0]["details"])
    assert payload["actor"] == "operator"
    assert payload["reason"] == "quota outage"
    assert payload["from_state"] == "FAILED"
    assert payload["to_state"] == "CLINE_REPORT_RECEIVED"
    assert payload["attempt"] == 3
    assert payload["report_hash"]
    assert payload["timestamp"]
    assert events(ctrl, "STATE:FAILED->CLINE_REPORT_RECEIVED")


def test_recovery_helper_refuses_any_other_state(ctrl):
    to_review_received(ctrl)  # CLINE_REPORT_RECEIVED, not FAILED

    with pytest.raises(RuntimeError, match="requires FAILED"):
        ctrl.db.recover_failed_review_to_received(
            1, actor="a", reason="r", report_hash="x", attempt=0
        )
    # the generic FSM table exposes no FAILED -> CLINE_REPORT_RECEIVED edge
    assert "CLINE_REPORT_RECEIVED" not in wt.ALLOWED_TRANSITIONS["FAILED"]


def test_there_is_no_generic_recovery_transition_api(ctrl):
    assert not hasattr(ctrl.db, "recovery_transition")
    assert not hasattr(ctrl.db, "recovery_transition_to")
    assert not hasattr(ctrl.db, "force_transition")


def test_recovery_cannot_run_when_a_review_exists(ctrl):
    to_review_received(ctrl, attempt=3)
    ctrl.db.save_codex_review(1, 3, {"verdict": "REVISE", "summary": "x"})
    force_state(ctrl, "FAILED")

    assert ctrl.review_recovery_qualified() is False
    with pytest.raises(RuntimeError, match="already has a Codex review"):
        ctrl.recover_failed_review_to_received()


def test_recovery_cannot_run_without_an_accepted_report(ctrl):
    to_cline_dispatched(ctrl, attempt=3)
    ctrl.db.transition(1, "CLINE_REPORT_RECEIVED")
    ctrl.db.transition(1, "CODEX_REVIEW_RUNNING")
    ctrl.db.transition(1, "FAILED")
    assert ctrl.db.cline_report_for_attempt(1, 3) is None

    assert ctrl.review_recovery_qualified() is False
    with pytest.raises(RuntimeError, match="no accepted Cline report"):
        ctrl.recover_failed_review_to_received()


def test_recovery_cannot_change_attempt_count(ctrl):
    legacy_failed_step(ctrl, attempt=3)

    ctrl.recover_failed_review_to_received()

    assert ctrl.db.get_step(1).attempt == 3


def test_recovery_cannot_be_applied_twice(ctrl):
    legacy_failed_step(ctrl, attempt=3)
    ctrl.recover_failed_review_to_received()
    force_state(ctrl, "FAILED")  # a second FAILED on the same step

    assert ctrl.review_recovery_qualified() is False
    with pytest.raises(RuntimeError, match="already been used"):
        ctrl.recover_failed_review_to_received()


def test_recovery_rejects_a_tampered_report_hash(ctrl):
    legacy_failed_step(ctrl, attempt=3)
    with ctrl.db.db:
        ctrl.db.db.execute(
            "UPDATE cline_reports SET report_hash='deadbeef' WHERE step_no=1"
        )

    assert ctrl.review_recovery_qualified() is False
    with pytest.raises(RuntimeError, match="does not match its stored hash"):
        ctrl.recover_failed_review_to_received()


def test_recovery_preserves_all_prior_events(ctrl):
    legacy_failed_step(ctrl, attempt=3)
    before = [
        tuple(row)
        for row in ctrl.db.db.execute(
            "SELECT id, event_type, details, created_at FROM events "
            "WHERE step_no=1 ORDER BY id"
        ).fetchall()
    ]

    ctrl.recover_failed_review_to_received(actor="operator", reason="quota outage")

    after = [
        tuple(row)
        for row in ctrl.db.db.execute(
            "SELECT id, event_type, details, created_at FROM events "
            "WHERE step_no=1 ORDER BY id"
        ).fetchall()
    ]
    assert after[: len(before)] == before  # append-only: nothing rewritten
    assert len(after) > len(before)


def test_fsm_history_is_never_lost_across_retry(ctrl):
    to_review_received(ctrl, attempt=3)
    before_events = ctrl.db.db.execute(
        "SELECT COUNT(*) FROM events WHERE step_no=1"
    ).fetchone()[0]
    before_reports = ctrl.db.db.execute(
        "SELECT COUNT(*) FROM cline_reports WHERE step_no=1"
    ).fetchone()[0]

    fail_review(ctrl)
    ctrl.run_codex_review()
    stub_review(ctrl, {"verdict": "REVISE", "summary": "ok"})
    ctrl.retry_codex_review()

    after_events = ctrl.db.db.execute(
        "SELECT COUNT(*) FROM events WHERE step_no=1"
    ).fetchone()[0]
    after_reports = ctrl.db.db.execute(
        "SELECT COUNT(*) FROM cline_reports WHERE step_no=1"
    ).fetchone()[0]

    assert after_events > before_events
    assert after_reports == before_reports  # the accepted report is never regenerated
    assert ctrl.db.cline_report_for_attempt(1, 3) is not None
    assert ctrl.db.has_codex_review(1, 3) is True


# --------------------------------------------------------------------------- #
# GUI: Retry Codex REVIEW button policy + deferred messaging
# --------------------------------------------------------------------------- #


@pytest.fixture
def gui(tmp_path):
    try:
        app = wt.TandemGUI()
    except Exception as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk unavailable: {exc}")
    app.root.withdraw()
    controller = wt.TandemController(tmp_path)
    controller.db.import_plan(PLAN_DEF)
    app.controller = controller
    yield app, controller
    try:
        app.on_close()
    except Exception:
        pass
    controller.db.close()


def test_retry_button_disabled_in_unrelated_states(gui):
    app, controller = gui
    db = controller.db

    app._set_buttons()
    assert str(app.btn_retry_review["state"]) == "disabled"  # READY_FOR_CODEX_PLAN

    for state in (
        "CODEX_PLAN_RUNNING",
        "PLAN_READY",
        "WAITING_HUMAN_APPROVAL",
        "CLINE_DISPATCHED",
        "CLINE_REPORT_RECEIVED",
        "CODEX_REVIEW_RUNNING",
    ):
        db.transition(1, state)
        app._set_buttons()
        assert str(app.btn_retry_review["state"]) == "disabled", state

    db.transition(1, "CODEX_REVIEW_RETRYABLE")
    app._set_buttons()
    assert str(app.btn_retry_review["state"]) == "normal"


def test_retry_button_disabled_when_recovery_ineligible(gui):
    app, controller = gui
    db = controller.db
    for state in (
        "CODEX_PLAN_RUNNING",
        "PLAN_READY",
        "WAITING_HUMAN_APPROVAL",
        "CLINE_DISPATCHED",
        "CLINE_REPORT_RECEIVED",
        "CODEX_REVIEW_RUNNING",
        "FAILED",
    ):
        db.transition(1, state)

    app._set_buttons()

    # FAILED, but no accepted report for this attempt -> recovery not offered
    assert str(app.btn_retry_review["state"]) == "disabled"


def test_retry_button_enabled_for_legacy_failed_recovery(gui):
    app, controller = gui
    to_review_received(controller, attempt=3)
    force_state(controller, "FAILED")

    app._set_buttons()

    assert str(app.btn_retry_review["state"]) == "normal"


def test_deferred_review_does_not_show_generic_operation_failed(gui, monkeypatch):
    app, controller = gui
    shown = []

    def fake_showinfo(*args, **kwargs):
        shown.append(args)

    monkeypatch.setattr(wt.messagebox, "showinfo", fake_showinfo)
    monkeypatch.setattr(wt.messagebox, "showerror", fake_showinfo)

    app._on_review_success({"deferred": True, "category": "quota", "retryable": True})

    assert "deferred" in app.footer_var.get()
    assert "Usage limit reached." in app.footer_var.get()
    assert shown, "the user must be told why the review was deferred"


def test_review_deferred_message_categories():
    assert wt.review_deferred_message({"category": "quota"}) == "Usage limit reached."
    assert wt.review_deferred_message({"category": "rate_limit"}) == "Rate limit reached."
    assert wt.review_deferred_message({"category": "timeout"}) == "The provider timed out."
    assert (
        "temporarily unavailable"
        in wt.review_deferred_message({"category": "not-a-category"})
    )
