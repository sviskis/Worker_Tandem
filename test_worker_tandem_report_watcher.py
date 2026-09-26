"""Worker_tandem Cline report readiness watcher tests.

Pins the V1 contract:
    - the watcher detects (never auto-processes) the exact current-step report;
    - Process Cline Report is enabled only when the report is READY;
    - INVALID/absent reports never mutate the FSM;
    - an unchanged INVALID/READY file is not re-read every poll cycle;
    - the watcher never clobbers a foreground/Codex footer message.

The driver filename is not a valid module name, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time

import pytest

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem_watcher", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_watcher"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()


PLAN_DEF = {
    "plan_version": "watcher-test-1",
    "steps": [
        {"step_no": 1, "phase": "FOUNDATION", "title": "ONE", "description": "d1", "risk": "LOW"},
        {"step_no": 2, "phase": "FOUNDATION", "title": "TWO", "description": "d2", "risk": "LOW"},
    ],
}


def report_ok(step_no=1, status="DONE"):
    return {
        "step_no": step_no,
        "status": status,
        "files_created": [],
        "files_changed": ["a.py"],
        "files_deleted": [],
        "tests": {"passed": 3, "failed": 0, "command": "pytest -q"},
        "dependencies_added": [],
        "architecture_questions": [],
        "issues": [],
        "summary": "done",
    }


DISPATCH_PATH = (
    "CODEX_PLAN_RUNNING",
    "PLAN_READY",
    "WAITING_HUMAN_APPROVAL",
    "CLINE_DISPATCHED",
)


def dispatch(controller, step_no=1):
    """Move ``step_no`` to CLINE_DISPATCHED using only legal transitions."""
    for state in DISPATCH_PATH:
        controller.db.transition(step_no, state)


@pytest.fixture
def ctrl(tmp_path):
    controller = wt.TandemController(tmp_path)
    controller.db.import_plan(PLAN_DEF)
    yield controller
    controller.db.close()


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


# --------------------------------------------------------------------------- #
# 1-7. Readiness of the exact current-step report (controller level, Tk-free)
# --------------------------------------------------------------------------- #


def test_no_report_is_waiting(ctrl):
    dispatch(ctrl)
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "WAITING"
    assert readiness["step_no"] == 1
    assert readiness["report"] is None


def test_valid_report_is_ready(ctrl):
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "READY"
    assert readiness["report"]["status"] == "DONE"
    assert readiness["detail"] == "step_001_report.json"
    assert isinstance(readiness["size"], int)


def test_invalid_json_is_invalid_and_fsm_unchanged(ctrl):
    dispatch(ctrl)
    ctrl.expected_cline_report_path(1).write_text("{ not json", encoding="utf-8")
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "INVALID"
    assert "JSON parse error" in readiness["detail"]
    assert ctrl.db.current_step().state == "CLINE_DISPATCHED"


def test_wrong_step_no_is_invalid(ctrl):
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok(step_no=999))
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "INVALID"
    assert "step_no mismatch" in readiness["detail"]


def test_invalid_status_is_invalid(ctrl):
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok(status="PENDING"))
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "INVALID"
    assert "Invalid Cline report status" in readiness["detail"]


def test_missing_required_field_is_invalid(ctrl):
    dispatch(ctrl)
    payload = report_ok()
    del payload["summary"]
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), payload)
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "INVALID"
    assert "summary" in readiness["detail"]


def test_temp_partial_file_is_ignored(ctrl):
    dispatch(ctrl)
    final = ctrl.expected_cline_report_path(1)
    tmp = final.with_name(final.name + ".tmp")
    tmp.write_text(json.dumps(report_ok()), encoding="utf-8")
    assert not final.exists()
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "WAITING"


def test_expected_report_path_is_step_scoped(ctrl):
    assert ctrl.expected_cline_report_path(1).name == "step_001_report.json"
    assert ctrl.expected_cline_report_path(2).name == "step_002_report.json"
    assert ctrl.expected_cline_report_path(3).name == "step_003_report.json"


def test_stable_file_guard_waits_while_writing(ctrl, monkeypatch):
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())

    class _Stat:
        def __init__(self, size, mtime_ns):
            self.st_size = size
            self.st_mtime_ns = mtime_ns
            self.st_mtime = mtime_ns / 1e9

    counter = {"n": 0}

    def flaky_stat(self, *args, **kwargs):
        counter["n"] += 1
        # stat before the read != stat after the read -> still being written.
        return _Stat(10, 111) if counter["n"] == 1 else _Stat(20, 222)

    monkeypatch.setattr(pathlib.Path, "stat", flaky_stat)
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "WAITING"
    assert "still being written" in readiness["detail"]


def test_oversized_report_is_invalid(ctrl, monkeypatch):
    dispatch(ctrl)
    monkeypatch.setattr(wt, "CLINE_REPORT_MAX_BYTES", 8)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())
    readiness = ctrl.check_cline_report_readiness()
    assert readiness["status"] == "INVALID"
    assert "size limit" in readiness["detail"]


# --------------------------------------------------------------------------- #
# 10-12. The button and process_cline_report share one validator / no auto-run
# --------------------------------------------------------------------------- #


def test_process_uses_shared_validator(ctrl, monkeypatch):
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())

    calls = []
    real = wt.validate_cline_report

    def spy(report, step_no):
        calls.append(step_no)
        return real(report, step_no)

    monkeypatch.setattr(wt, "validate_cline_report", spy)
    assert ctrl.check_cline_report_readiness()["status"] == "READY"
    ctrl.process_cline_report()
    assert calls
    assert set(calls) == {1}


def test_process_valid_report_transitions_unchanged(ctrl):
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())
    report = ctrl.process_cline_report()
    assert report["status"] == "DONE"
    assert ctrl.db.current_step().state == "CLINE_REPORT_RECEIVED"
    assert ctrl.db.latest_cline_report(1)["status"] == "DONE"


def test_process_missing_report_raises_not_found(ctrl):
    dispatch(ctrl)
    with pytest.raises(FileNotFoundError, match="Cline report not found"):
        ctrl.process_cline_report()


def test_process_invalid_report_raises_and_fsm_unchanged(ctrl):
    dispatch(ctrl)
    ctrl.expected_cline_report_path(1).write_text("{ broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="JSON parse error"):
        ctrl.process_cline_report()
    assert ctrl.db.current_step().state == "CLINE_DISPATCHED"


def test_no_duplicate_processing(ctrl):
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())
    ctrl.process_cline_report()
    # V1 never auto-processes: a second manual attempt is rejected by the FSM.
    with pytest.raises(RuntimeError, match="Expected CLINE_DISPATCHED"):
        ctrl.process_cline_report()


# --------------------------------------------------------------------------- #
# GUI: WAITING / READY / INVALID status line + Process button policy
# --------------------------------------------------------------------------- #


def test_gui_no_report_waits_and_disables_button(gui):
    app, ctrl = gui
    dispatch(ctrl)
    app.refresh()
    assert "WAITING" in app.cline_report_var.get()
    assert str(app.btn_report["state"]) == "disabled"


def test_gui_valid_report_ready_enables_button(gui):
    app, ctrl = gui
    dispatch(ctrl)
    app.refresh()
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())
    app._refresh_cline_report_status()
    assert "READY" in app.cline_report_var.get()
    assert "step_001_report.json" in app.cline_report_var.get()
    assert str(app.btn_report["state"]) == "normal"


def test_gui_invalid_keeps_button_disabled(gui):
    app, ctrl = gui
    dispatch(ctrl)
    ctrl.expected_cline_report_path(1).write_text("{ broken", encoding="utf-8")
    app.refresh()
    assert "INVALID" in app.cline_report_var.get()
    assert str(app.btn_report["state"]) == "disabled"
    assert ctrl.db.current_step().state == "CLINE_DISPATCHED"


def test_gui_temp_file_kept_waiting(gui):
    app, ctrl = gui
    dispatch(ctrl)
    final = ctrl.expected_cline_report_path(1)
    final.with_name(final.name + ".tmp").write_text(
        json.dumps(report_ok()), encoding="utf-8"
    )
    app.refresh()
    assert "WAITING" in app.cline_report_var.get()
    assert str(app.btn_report["state"]) == "disabled"


def test_gui_retargets_when_step_changes(gui):
    app, ctrl = gui
    dispatch(ctrl, 1)
    app.refresh()
    assert app._watcher_target == (1, "CLINE_DISPATCHED")

    for state in ("CLINE_REPORT_RECEIVED", "CODEX_REVIEW_RUNNING", "REVIEW_APPROVED"):
        ctrl.db.transition(1, state)
    ctrl.db.mark_verified_and_advance(1)
    dispatch(ctrl, 2)
    app.refresh()
    assert app._watcher_target == (2, "CLINE_DISPATCHED")

    wt.write_json_atomic(ctrl.expected_cline_report_path(2), report_ok(step_no=2))
    app._refresh_cline_report_status()
    assert "READY" in app.cline_report_var.get()
    assert "step_002_report.json" in app.cline_report_var.get()
    assert str(app.btn_report["state"]) == "normal"


def test_gui_starts_and_stops_watcher(gui):
    app, ctrl = gui
    dispatch(ctrl)
    app._start_report_watcher()
    assert app._watcher_after_id is not None
    app._stop_report_watcher()
    assert app._watcher_after_id is None


def test_gui_poll_reschedules_only_when_dispatched(gui):
    app, ctrl = gui
    dispatch(ctrl)
    app._start_report_watcher()
    assert app._watcher_after_id is not None
    app._poll_cline_report()
    assert app._watcher_after_id is not None  # re-armed while dispatched

    ctrl.db.transition(1, "CLINE_REPORT_RECEIVED")
    app._poll_cline_report()
    assert app._watcher_after_id is None  # retarget/stop when the step changes


def test_gui_poll_is_non_blocking(gui):
    app, ctrl = gui
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())
    app.refresh()
    start = time.perf_counter()
    app._poll_cline_report()
    assert time.perf_counter() - start < 1.0


# --------------------------------------------------------------------------- #
# INVALID/READY cache: unchanged files are not re-parsed every poll cycle
# --------------------------------------------------------------------------- #


def _count_readiness_calls(ctrl):
    calls = {"n": 0}
    real = ctrl.check_cline_report_readiness

    def spy():
        calls["n"] += 1
        return real()

    ctrl.check_cline_report_readiness = spy
    return calls


def test_unchanged_invalid_file_not_reparsed(gui):
    app, ctrl = gui
    dispatch(ctrl)
    ctrl.expected_cline_report_path(1).write_text("{ broken", encoding="utf-8")

    calls = _count_readiness_calls(ctrl)
    app._refresh_cline_report_status()
    app._refresh_cline_report_status()
    app._refresh_cline_report_status()
    assert calls["n"] == 1  # cached INVALID reused; file not re-read/re-parsed
    assert "INVALID" in app.cline_report_var.get()


def test_changed_invalid_file_is_revalidated(gui):
    app, ctrl = gui
    dispatch(ctrl)
    path = ctrl.expected_cline_report_path(1)
    path.write_text("{ broken", encoding="utf-8")

    calls = _count_readiness_calls(ctrl)
    app._refresh_cline_report_status()
    assert calls["n"] == 1
    # A different length/content changes the size+mtime signature.
    path.write_text("{ broken but now longer content", encoding="utf-8")
    app._refresh_cline_report_status()
    assert calls["n"] == 2


def test_ready_file_is_cached(gui):
    app, ctrl = gui
    dispatch(ctrl)
    wt.write_json_atomic(ctrl.expected_cline_report_path(1), report_ok())

    calls = _count_readiness_calls(ctrl)
    app._refresh_cline_report_status()
    app._refresh_cline_report_status()
    assert calls["n"] == 1  # unchanged READY file reused
    assert "READY" in app.cline_report_var.get()


# --------------------------------------------------------------------------- #
# Footer safety: the watcher never clobbers a foreground progress message
# --------------------------------------------------------------------------- #


def test_watcher_does_not_overwrite_footer_while_busy(gui):
    app, ctrl = gui
    dispatch(ctrl)
    app.busy = True
    app.footer_var.set("Codex is preparing PLAN...")
    app._refresh_cline_report_status()
    assert app.footer_var.get() == "Codex is preparing PLAN..."
    # The dedicated status line stays authoritative regardless.
    assert "WAITING" in app.cline_report_var.get()


def test_watcher_does_not_overwrite_foreground_footer(gui):
    app, ctrl = gui
    dispatch(ctrl)
    app.busy = False
    app.footer_var.set("Cline report processed.")
    app._refresh_cline_report_status()
    assert app.footer_var.get() == "Cline report processed."


def test_watcher_sets_footer_when_it_owns_it(gui):
    app, ctrl = gui
    dispatch(ctrl)
    app.footer_var.set("")
    app._refresh_cline_report_status()
    assert app.footer_var.get() == wt.WATCHER_FOOTER_WAITING


