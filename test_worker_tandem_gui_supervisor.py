"""Supervisor provider GUI + policy-control tests (milestone M9).

Pins the M9 contract:

* the Supervisor GUI section exposes Mode (AUTO/MANUAL), a PLAN provider, a
  REVIEW provider, a Budget and the two operator buttons, over a provider list
  of ``AUTO | DeepSeek | OpenAI | Claude | Codex Legacy``;
* the policy is persisted as NON-SECRET json under
  ``<project>/.worker_tandem/supervisor_policy.json`` using atomic writes, and
  is applied to the live router immediately;
* API keys come from the environment only: no key entry field exists and only
  ``NAME: FOUND/MISSING`` may ever be displayed (never a value);
* "Check Providers" probes providers off the Tk thread and runs no completion;
* the workflow buttons say "Supervisor" while every persisted FSM state string
  stays byte-identical to the pre-M9 set;
* an already accepted STEP report (with a Codex-produced PLAN) is reviewable
  through DeepSeek/OpenAI without re-running PLAN and without touching
  ``attempt`` or the accepted report.

No live API, no socket, no real Codex process: HTTP providers are driven by the
M2 ``FakeHttpTransport`` and the Codex health probe is stubbed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

from supervisor import (
    DEFAULT_LEGACY_PROVIDER,
    ID_ANTHROPIC,
    ID_DEEPSEEK,
    ID_OPENAI,
    PROVIDER_CODEX_LEGACY,
    FakeHttpResponse,
    FakeHttpTransport,
    SupervisorRunError,
    fake_review_payload,
)
from supervisor.errors import CATEGORY_PROVIDER_UNAVAILABLE

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    """Load the app under a private module name (no cross-test interference)."""
    spec = importlib.util.spec_from_file_location("worker_tandem_m9", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_m9"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()

#: Realistic-looking credentials: they must never reach a widget or a file.
SECRET_DEEPSEEK = "sk-" + "A" * 24
SECRET_OPENAI = "sk-proj-" + "B" * 24
SECRETS = (SECRET_DEEPSEEK, SECRET_OPENAI)

ENV_WITH_KEYS = {
    "DEEPSEEK_API_KEY": SECRET_DEEPSEEK,
    "OPENAI_API_KEY": SECRET_OPENAI,
}
ENV_WITHOUT_KEYS: dict = {}

PLAN_DEF = {
    "plan_version": "m9-test-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "STEP ONE",
            "description": "m9 integration step one",
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

#: Every persisted FSM state string the app may write (frozen for M9).
FSM_STATES = (
    "READY_FOR_CODEX_PLAN",
    "CODEX_PLAN_RUNNING",
    "PLAN_READY",
    "WAITING_HUMAN_APPROVAL",
    "CLINE_DISPATCHED",
    "CLINE_REPORT_RECEIVED",
    "CODEX_REVIEW_RUNNING",
    "CODEX_REVIEW_RETRYABLE",
    "REVIEW_APPROVED",
    "REVISE",
    "BLOCKED",
    "FAILED",
    "VERIFIED",
    "ABORTED",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def models_response(*names: str) -> FakeHttpResponse:
    return FakeHttpResponse(
        status_code=200,
        payload={"object": "list", "data": [{"id": name} for name in names]},
    )


def chat_response(payload: dict) -> FakeHttpResponse:
    return FakeHttpResponse(
        status_code=200,
        payload={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        },
    )


def ready_transport(**kwargs) -> FakeHttpTransport:
    """A transport that answers every provider health probe with 200 OK."""
    if "default" not in kwargs:
        kwargs["default"] = models_response("model-a", "model-b")
    return FakeHttpTransport(**kwargs)


def controller_for(project, *, transport=None, env=ENV_WITHOUT_KEYS, policy=None):
    controller = wt.TandemController(
        project,
        supervisor_transport=transport if transport is not None else ready_transport(),
        supervisor_env=dict(env),
    )
    controller.db.import_plan(PLAN_DEF)
    if policy is not None:
        controller.save_supervisor_policy(policy)
    return controller


def stub_codex_health(controller, *, available=True):
    """Keep the Codex probe deterministic (no `codex --version` subprocess)."""
    controller.codex.available = lambda: (available, "codex-stub 1.0")


def stub_runner(controller, payload):
    def fake(
        prompt, schema_path, output_path, trace_path, timeout_sec=None, reasoning_effort=None
    ):
        return dict(payload)

    controller.codex.run_structured = fake


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


def make_gui(project, *, transport=None, env=ENV_WITHOUT_KEYS):
    """A withdrawn Tk app bound to a real controller (headless-safe)."""
    try:
        app = wt.TandemGUI()
    except Exception as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk unavailable: {exc}")
    app.root.withdraw()
    controller = controller_for(project, transport=transport, env=env)
    stub_codex_health(controller)
    app.controller = controller
    app.refresh()
    return app, controller


@pytest.fixture
def gui(tmp_path):
    app, controller = make_gui(tmp_path, env=ENV_WITH_KEYS)
    yield app, controller
    try:
        app.on_close()
    except Exception:
        pass
    controller.db.close()


def policy_file(controller) -> pathlib.Path:
    return controller.supervisor_policy_path()


def read_policy(controller) -> dict:
    return json.loads(policy_file(controller).read_text(encoding="utf-8"))


def widget_texts(app) -> list[str]:
    """Every visible text of the app (bound variables + the details pane)."""
    texts: list[str] = []
    for name in dir(app):
        if not name.endswith("_var"):
            continue
        getter = getattr(getattr(app, name, None), "get", None)
        if callable(getter):
            try:
                texts.append(str(getter()))
            except Exception:
                pass
    for attr in ("provider_status_vars", "provider_key_vars", "supervisor_model_vars"):
        for var in getattr(app, attr, {}).values():
            texts.append(str(var.get()))
    try:
        texts.append(app.details.get("1.0", "end"))
    except Exception:
        pass
    return texts


def prepare_accepted_report(controller):
    """A Codex-produced PLAN plus an accepted Cline report for attempt 1."""
    stub_runner(controller, PLAN_OK)
    controller.run_codex_plan()
    controller.approve_plan_and_dispatch_cline(reason="m9 test")
    wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
    controller.process_cline_report()
    return controller.db.get_step(1)


def run_until_idle(app, timeout_ms=20000):
    """Pump Tk until the background job finishes (mirrors the Git GUI test)."""

    def stop_when_done():
        if not app.busy:
            app.root.quit()
        else:
            app.root.after(25, stop_when_done)

    app.root.after(25, stop_when_done)
    app.root.after(timeout_ms, app.root.quit)
    app.root.mainloop()


# --------------------------------------------------------------------------- #
# 1-4  Provider list, labels and key-status contract (no GUI needed)
# --------------------------------------------------------------------------- #


def test_provider_choices_match_the_spec():
    assert wt.SUPERVISOR_PROVIDER_CHOICES == (
        "AUTO",
        ID_DEEPSEEK,
        ID_OPENAI,
        ID_ANTHROPIC,
        PROVIDER_CODEX_LEGACY,
    )
    assert [wt.supervisor_label(p) for p in wt.SUPERVISOR_PROVIDER_CHOICES] == [
        "AUTO",
        "DeepSeek",
        "OpenAI",
        "Claude",
        "Codex Legacy",
    ]
    # "Claude" is the label of the router's canonical anthropic id.
    assert wt.supervisor_provider_from_label("Claude") == ID_ANTHROPIC
    assert wt.supervisor_provider_from_label("Codex Legacy") == PROVIDER_CODEX_LEGACY
    assert wt.SUPERVISOR_MANUAL_CHOICES == (
        ID_DEEPSEEK,
        ID_OPENAI,
        ID_ANTHROPIC,
        PROVIDER_CODEX_LEGACY,
    )
    assert wt.SUPERVISOR_MODE_CHOICES == ("AUTO", "MANUAL")
    assert wt.SUPERVISOR_BUDGET_CHOICES == ("CHEAP", "STANDARD", "PREMIUM")


def test_key_status_is_presence_only():
    status = wt.supervisor_api_key_status(ENV_WITH_KEYS)
    assert status == {
        "DEEPSEEK_API_KEY": "FOUND",
        "OPENAI_API_KEY": "FOUND",
        "ANTHROPIC_API_KEY": "MISSING",
    }
    assert (
        wt.supervisor_api_key_status({"DEEPSEEK_API_KEY": "   "})["DEEPSEEK_API_KEY"]
        == "MISSING"
    )
    # Only the verdict may ever leave the helper - never a value or a prefix.
    assert set(status.values()) == {"FOUND", "MISSING"}
    assert SECRET_DEEPSEEK not in json.dumps(status)


def test_provider_status_labels_cover_the_spec_states():
    assert wt.supervisor_provider_status_label("READY") == "READY"
    assert wt.supervisor_provider_status_label("key_missing") == "KEY MISSING"
    assert wt.supervisor_provider_status_label("READY", "quota_with_reset") == "LIMIT"
    assert wt.supervisor_provider_status_label("READY", "rate_limit") == "LIMIT"
    assert (
        wt.supervisor_provider_status_label("provider_error") == "ERROR (PROVIDER_ERROR)"
    )
    assert wt.supervisor_provider_status_label("") == "NOT CHECKED"


def test_claude_is_listed_but_has_no_adapter_yet():
    available = wt.TandemController.supervisor_provider_available
    assert available(ID_ANTHROPIC) is False
    assert available(ID_DEEPSEEK) is True
    assert available(PROVIDER_CODEX_LEGACY) is True
    assert wt.build_supervisor_provider(ID_ANTHROPIC, transport=ready_transport()) is None


# --------------------------------------------------------------------------- #
# 5-11  Policy persistence (atomic, non-secret, applied immediately)
# --------------------------------------------------------------------------- #


def test_default_project_has_no_policy_file_and_keeps_m4_behaviour(tmp_path):
    controller = controller_for(tmp_path)
    try:
        expected = tmp_path / ".worker_tandem" / "supervisor_policy.json"
        assert policy_file(controller) == expected
        assert expected.exists() is False

        policy = controller.supervisor_policy
        assert policy.mode == "MANUAL"
        assert policy.plan_provider == PROVIDER_CODEX_LEGACY
        assert policy.review_provider == PROVIDER_CODEX_LEGACY
        assert controller.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
        )
    finally:
        controller.db.close()


def test_policy_save_writes_atomically_without_a_temp_file(tmp_path):
    controller = controller_for(tmp_path)
    try:
        payload = controller.save_supervisor_policy(
            wt.SupervisorPolicy(
                mode="MANUAL",
                plan_provider=ID_DEEPSEEK,
                review_provider=ID_OPENAI,
                budget_mode="PREMIUM",
            )
        )
        path = policy_file(controller)
        assert path.exists()
        assert not list(path.parent.glob("*.tmp"))
        stored = read_policy(controller)
        assert stored["version"] == wt.SUPERVISOR_POLICY_VERSION
        assert stored["mode"] == "MANUAL"
        assert stored["plan_provider"] == ID_DEEPSEEK
        assert stored["review_provider"] == ID_OPENAI
        assert stored["budget_mode"] == "PREMIUM"
        assert payload["plan_provider"] == ID_DEEPSEEK
        # Applied immediately, without a restart.
        assert controller.supervisor_policy.plan_provider == ID_DEEPSEEK
        assert controller.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
            ID_DEEPSEEK,
            ID_OPENAI,
        )
    finally:
        controller.db.close()


def test_policy_write_is_atomic_when_the_replace_fails(tmp_path, monkeypatch):
    controller = controller_for(tmp_path)
    try:
        controller.save_supervisor_policy(
            wt.SupervisorPolicy(plan_provider=ID_DEEPSEEK, review_provider=ID_DEEPSEEK)
        )
        before = policy_file(controller).read_text(encoding="utf-8")

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(wt.os, "replace", boom)
        with pytest.raises(OSError):
            controller.save_supervisor_policy(
                wt.SupervisorPolicy(plan_provider=ID_OPENAI, review_provider=ID_OPENAI)
            )

        # The previous policy survived byte-identically and stayed applied.
        assert policy_file(controller).read_text(encoding="utf-8") == before
        assert controller.supervisor_policy.plan_provider == ID_DEEPSEEK
    finally:
        controller.db.close()


def test_policy_round_trips_through_a_new_controller(tmp_path):
    controller = controller_for(tmp_path)
    try:
        controller.save_supervisor_policy(
            wt.SupervisorPolicy(
                mode="MANUAL",
                plan_provider=ID_DEEPSEEK,
                review_provider=ID_ANTHROPIC,
                budget_mode="CHEAP",
            )
        )
    finally:
        controller.db.close()

    reopened = wt.TandemController(
        tmp_path, supervisor_transport=ready_transport(), supervisor_env={}
    )
    try:
        policy = reopened.supervisor_policy
        assert policy.mode == "MANUAL"
        assert policy.plan_provider == ID_DEEPSEEK
        assert policy.review_provider == ID_ANTHROPIC
        assert policy.budget_mode == "CHEAP"
        # Claude has no adapter yet, so it stays in the policy (the router skips
        # unregistered candidates) but is never registered.
        assert reopened.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
            ID_DEEPSEEK,
        )
    finally:
        reopened.db.close()


def test_auto_policy_round_trips_through_a_new_controller(tmp_path):
    controller = controller_for(tmp_path)
    try:
        controller.save_supervisor_policy(wt.default_auto_policy())
    finally:
        controller.db.close()

    reopened = wt.TandemController(
        tmp_path, supervisor_transport=ready_transport(), supervisor_env={}
    )
    try:
        policy = reopened.supervisor_policy
        assert policy.is_auto is True
        assert policy.candidates_for("PLAN", "LOW")[0] == ID_DEEPSEEK
        assert policy.candidates_for("PLAN", "HIGH")[0] == ID_OPENAI
        assert reopened.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
            ID_DEEPSEEK,
            ID_OPENAI,
        )
    finally:
        reopened.db.close()


def test_corrupt_policy_file_falls_back_to_the_default(tmp_path):
    controller = controller_for(tmp_path)
    try:
        policy_file(controller).write_text("{not json", encoding="utf-8")
        reloaded = controller.load_supervisor_policy()
        assert reloaded.mode == "MANUAL"
        assert reloaded.plan_provider == PROVIDER_CODEX_LEGACY
        assert controller.read_supervisor_policy_file() == {}
    finally:
        controller.db.close()

    reopened = wt.TandemController(
        tmp_path, supervisor_transport=ready_transport(), supervisor_env={}
    )
    try:
        assert reopened.supervisor_policy.mode == "MANUAL"
        assert reopened.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
        )
    finally:
        reopened.db.close()


def test_policy_file_never_contains_a_secret(tmp_path):
    controller = controller_for(tmp_path, env=ENV_WITH_KEYS)
    try:
        controller.save_supervisor_policy(
            wt.SupervisorPolicy(plan_provider=ID_DEEPSEEK, review_provider=ID_OPENAI),
            models={ID_DEEPSEEK: "deepseek-chat", ID_OPENAI: "gpt-4o-mini"},
        )
        text = policy_file(controller).read_text(encoding="utf-8")
        for secret in SECRETS:
            assert secret not in text
        for env_name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            assert env_name not in text
        lowered = text.lower()
        assert "api_key" not in lowered
        assert "token" not in lowered
        stored = json.loads(text)
        assert stored["models"] == {
            ID_DEEPSEEK: "deepseek-chat",
            ID_OPENAI: "gpt-4o-mini",
            ID_ANTHROPIC: "",
        }
        assert stored["enabled_providers"]
    finally:
        controller.db.close()


def test_saved_models_reach_the_live_adapter(tmp_path):
    controller = controller_for(tmp_path, env=ENV_WITH_KEYS)
    try:
        controller.save_supervisor_policy(
            wt.SupervisorPolicy(plan_provider=ID_OPENAI, review_provider=ID_OPENAI),
            models={ID_OPENAI: "gpt-4o-test", ID_DEEPSEEK: "deepseek-reasoner"},
        )
        assert controller.supervisor_models()[ID_OPENAI] == "gpt-4o-test"
        adapter = controller.supervisor_provider(ID_OPENAI)
        assert adapter.model == "gpt-4o-test"
        assert adapter.config.model == "gpt-4o-test"
        assert controller.supervisor_provider(ID_DEEPSEEK).model == "deepseek-reasoner"
    finally:
        controller.db.close()


def test_enabled_providers_gates_registration(tmp_path):
    controller = controller_for(tmp_path, env=ENV_WITH_KEYS)
    try:
        controller.save_supervisor_policy(
            wt.SupervisorPolicy(plan_provider=ID_DEEPSEEK, review_provider=ID_DEEPSEEK),
            enabled_providers=(PROVIDER_CODEX_LEGACY,),
        )
        assert controller.supervisor_enabled_providers() == (PROVIDER_CODEX_LEGACY,)
        assert controller.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
        )
    finally:
        controller.db.close()


# --------------------------------------------------------------------------- #
# 12-17  Provider health / route introspection (no GUI needed)
# --------------------------------------------------------------------------- #


def test_provider_status_reports_key_missing_without_any_request(tmp_path):
    transport = ready_transport()
    controller = controller_for(tmp_path, transport=transport, env=ENV_WITHOUT_KEYS)
    try:
        stub_codex_health(controller)
        status = controller.provider_status()

        assert status[ID_DEEPSEEK]["status"] == "KEY MISSING"
        assert status[ID_DEEPSEEK]["key"] == "MISSING"
        assert status[ID_OPENAI]["status"] == "KEY MISSING"
        assert status[ID_DEEPSEEK]["health"] == "KEY_MISSING"
        # A missing key is detected locally: no HTTP request is attempted.
        assert transport.requests == []
        assert controller.supervisor_key_status() == {
            "DEEPSEEK_API_KEY": "MISSING",
            "OPENAI_API_KEY": "MISSING",
            "ANTHROPIC_API_KEY": "MISSING",
        }
    finally:
        controller.db.close()


@pytest.mark.parametrize("provider_id", [ID_DEEPSEEK, ID_OPENAI])
def test_provider_status_reports_ready_on_200(tmp_path, provider_id):
    transport = ready_transport()
    controller = controller_for(tmp_path, transport=transport, env=ENV_WITH_KEYS)
    try:
        stub_codex_health(controller)
        status = controller.provider_status()

        assert status[provider_id]["status"] == "READY"
        assert status[provider_id]["key"] == "FOUND"
        assert "2 models available" in status[provider_id]["detail"]
        probes = [r for r in transport.requests if r["method"] == "GET"]
        assert probes, "the provider should have been probed"
        # A probe is a GET /models, never a completion.
        assert all(r["url"].endswith("/models") for r in probes)
        assert status[PROVIDER_CODEX_LEGACY]["status"] == "READY"
    finally:
        controller.db.close()


def test_provider_status_reports_an_error_and_keeps_going(tmp_path):
    transport = FakeHttpTransport(
        responses=[
            FakeHttpResponse(
                status_code=500,
                payload={"error": {"message": "boom"}},
                error=SupervisorRunError(
                    "provider exploded",
                    category=CATEGORY_PROVIDER_UNAVAILABLE,
                    http_status=500,
                    sanitized_message="provider exploded",
                ),
            )
        ],
        default=models_response("ok"),
    )
    controller = controller_for(tmp_path, transport=transport, env=ENV_WITH_KEYS)
    try:
        stub_codex_health(controller)
        status = controller.provider_status()

        assert status[ID_DEEPSEEK]["status"].startswith("ERROR")
        # A failing provider never aborts the check of the others.
        assert status[PROVIDER_CODEX_LEGACY]["status"] == "READY"
        assert status[ID_ANTHROPIC]["status"] == "UNAVAILABLE"
    finally:
        controller.db.close()


def test_codex_legacy_shows_limit_after_a_quota_failure(tmp_path):
    controller = controller_for(tmp_path, env=ENV_WITH_KEYS)
    try:
        stub_codex_health(controller)
        # Simulate the router's in-memory record of an external quota outage.
        controller.supervisor_router.route_history.append(
            {
                "provider": PROVIDER_CODEX_LEGACY,
                "success": False,
                "failures": [f"{PROVIDER_CODEX_LEGACY}=quota_with_reset"],
            }
        )
        assert (
            controller.last_provider_category(PROVIDER_CODEX_LEGACY)
            == "quota_with_reset"
        )
        assert controller.provider_status()[PROVIDER_CODEX_LEGACY]["status"] == "LIMIT"

        # A later success clears the LIMIT badge again.
        controller.supervisor_router.route_history.append(
            {"provider": PROVIDER_CODEX_LEGACY, "success": True, "failures": []}
        )
        assert controller.last_provider_category(PROVIDER_CODEX_LEGACY) is None
        assert controller.provider_status()[PROVIDER_CODEX_LEGACY]["status"] == "READY"
    finally:
        controller.db.close()


def test_current_routes_follow_the_policy_and_the_step_risk(tmp_path):
    controller = controller_for(
        tmp_path,
        env=ENV_WITH_KEYS,
        policy=wt.SupervisorPolicy(
            mode="MANUAL",
            plan_provider=ID_DEEPSEEK,
            review_provider=ID_OPENAI,
            budget_mode="STANDARD",
        ),
    )
    try:
        routes = controller.current_routes()
        assert routes["plan"]["chain"] == [ID_DEEPSEEK]
        assert routes["plan"]["label"] == "DeepSeek"
        assert routes["review"]["chain"] == [ID_OPENAI]
        assert routes["review"]["label"] == "OpenAI"
        assert routes["plan"]["risk"] == "LOW"
        assert routes["plan"]["mode"] == "MANUAL"
    finally:
        controller.db.close()

    auto = wt.TandemController(
        tmp_path / "auto",
        supervisor_transport=ready_transport(),
        supervisor_env=dict(ENV_WITH_KEYS),
    )
    try:
        auto.db.import_plan(PLAN_DEF)
        auto.save_supervisor_policy(wt.default_auto_policy())
        routes = auto.current_routes()
        assert routes["plan"]["label"] == "DeepSeek -> OpenAI -> Claude -> Codex Legacy"
        assert routes["review"]["primary"] == ID_DEEPSEEK
    finally:
        auto.db.close()


# --------------------------------------------------------------------------- #
# 18-28  GUI: layout, selection, persistence, secrets, health, labels
# --------------------------------------------------------------------------- #


def test_gui_supervisor_section_has_the_specified_controls(gui):
    app, _ = gui

    assert tuple(app.supervisor_mode_combo["values"]) == ("AUTO", "MANUAL")
    assert tuple(app.supervisor_budget_combo["values"]) == (
        "CHEAP",
        "STANDARD",
        "PREMIUM",
    )
    # MANUAL (the default) offers the concrete providers: AUTO is the mode.
    assert tuple(app.supervisor_plan_combo["values"]) == (
        "DeepSeek",
        "OpenAI",
        "Claude",
        "Codex Legacy",
    )
    assert tuple(app.supervisor_review_combo["values"]) == tuple(
        app.supervisor_plan_combo["values"]
    )
    assert app.supervisor_mode_var.get() == "MANUAL"
    assert app.supervisor_plan_var.get() == "Codex Legacy"
    assert app.supervisor_review_var.get() == "Codex Legacy"
    assert app.supervisor_budget_var.get() == "STANDARD"

    assert set(app.provider_status_vars) == set(wt.SUPERVISOR_MANUAL_CHOICES)
    assert set(app.provider_key_vars) == {
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
    assert set(app.supervisor_model_vars) == set(wt.SUPERVISOR_MODEL_PROVIDERS)
    assert app.supervisor_plan_route_var.get() == "Current PLAN route: Codex Legacy"
    assert app.supervisor_review_route_var.get() == "Current REVIEW route: Codex Legacy"
    assert str(app.btn_check_providers["text"]) == "Check Providers"
    assert str(app.btn_save_policy["text"]) == "Save Policy"
    assert str(app.btn_check_providers["state"]) == "normal"
    assert str(app.btn_save_policy["state"]) == "normal"


def test_gui_shows_auto_in_the_provider_combos_in_auto_mode(tmp_path):
    app, controller = make_gui(tmp_path, env=ENV_WITH_KEYS)
    try:
        controller.save_supervisor_policy(wt.default_auto_policy())
        app._supervisor_ui_key = None
        app.refresh()

        assert app.supervisor_mode_var.get() == "AUTO"
        assert tuple(app.supervisor_plan_combo["values"]) == ("AUTO",)
        assert tuple(app.supervisor_review_combo["values"]) == ("AUTO",)
        assert app.supervisor_plan_var.get() == "AUTO"
        assert app.supervisor_review_var.get() == "AUTO"
        assert str(app.supervisor_plan_combo["state"]) == "disabled"
        assert app.supervisor_plan_route_var.get().endswith(
            "DeepSeek -> OpenAI -> Claude -> Codex Legacy"
        )
    finally:
        app.on_close()
        controller.db.close()


def test_gui_manual_selection_is_saved_and_applied(gui):
    app, controller = gui

    app.supervisor_mode_var.set("MANUAL")
    app.supervisor_plan_var.set("DeepSeek")
    app.supervisor_review_var.set("OpenAI")
    app.supervisor_budget_var.set("PREMIUM")
    app.save_policy()

    stored = read_policy(controller)
    assert stored["mode"] == "MANUAL"
    assert stored["plan_provider"] == ID_DEEPSEEK
    assert stored["review_provider"] == ID_OPENAI
    assert stored["budget_mode"] == "PREMIUM"
    # Applied to the live router immediately.
    assert controller.supervisor_policy.plan_provider == ID_DEEPSEEK
    assert controller.supervisor_policy.review_provider == ID_OPENAI
    assert controller.supervisor_router.registered_providers() == (
        PROVIDER_CODEX_LEGACY,
        ID_DEEPSEEK,
        ID_OPENAI,
    )
    assert app.supervisor_plan_route_var.get() == "Current PLAN route: DeepSeek"
    assert app.supervisor_review_route_var.get() == "Current REVIEW route: OpenAI"
    assert "supervisor_policy.json" in app.footer_var.get()


def test_gui_auto_selection_seeds_the_auto_policy(gui):
    app, controller = gui

    app.supervisor_mode_var.set("AUTO")
    app.supervisor_budget_var.set("CHEAP")
    app.save_policy()

    stored = read_policy(controller)
    assert stored["mode"] == "AUTO"
    assert stored["budget_mode"] == "CHEAP"
    # AUTO is seeded from the M8 §2 policy so it is immediately useful.
    assert stored["fallback_chain"] == [
        ID_DEEPSEEK,
        ID_OPENAI,
        ID_ANTHROPIC,
        PROVIDER_CODEX_LEGACY,
    ]
    assert stored["max_provider_retries"] == 1
    assert stored["allow_codex_legacy"] is True
    assert controller.supervisor_policy.is_auto is True
    assert controller.supervisor_router.registered_providers() == (
        PROVIDER_CODEX_LEGACY,
        ID_DEEPSEEK,
        ID_OPENAI,
    )
    assert app.supervisor_mode_var.get() == "AUTO"


def test_gui_review_choice_maps_the_claude_label_to_the_anthropic_id(gui):
    app, controller = gui

    app.supervisor_mode_var.set("MANUAL")
    app.supervisor_plan_var.set("Codex Legacy")
    app.supervisor_review_var.set("Claude")
    app.save_policy()

    stored = read_policy(controller)
    assert stored["review_provider"] == ID_ANTHROPIC
    assert controller.supervisor_policy.review_provider == ID_ANTHROPIC
    # Claude has no adapter before M7: it is never registered, so nothing
    # silently pretends to support it.
    assert controller.supervisor_router.registered_providers() == (
        PROVIDER_CODEX_LEGACY,
    )


def test_gui_plan_choice_is_saved_independently_of_review(gui):
    app, controller = gui

    app.supervisor_plan_var.set("DeepSeek")
    app.supervisor_review_var.set("Codex Legacy")
    app.save_policy()

    stored = read_policy(controller)
    assert stored["plan_provider"] == ID_DEEPSEEK
    assert stored["review_provider"] == PROVIDER_CODEX_LEGACY
    assert controller.supervisor_policy.provider_for("PLAN") == ID_DEEPSEEK
    assert controller.supervisor_policy.provider_for("REVIEW") == PROVIDER_CODEX_LEGACY


def test_gui_model_entries_are_persisted_and_used(gui):
    app, controller = gui

    app.supervisor_model_vars[ID_DEEPSEEK].set("deepseek-reasoner")
    app.supervisor_model_vars[ID_OPENAI].set("gpt-4o-mini-test")
    app.save_policy()

    stored = read_policy(controller)
    assert stored["models"][ID_DEEPSEEK] == "deepseek-reasoner"
    assert stored["models"][ID_OPENAI] == "gpt-4o-mini-test"
    assert controller.supervisor_models()[ID_DEEPSEEK] == "deepseek-reasoner"
    assert controller.supervisor_provider(ID_DEEPSEEK).model == "deepseek-reasoner"
    # Codex Legacy keeps its legacy configuration (never a model override).
    assert PROVIDER_CODEX_LEGACY not in stored["models"]
    assert set(stored["models"]) == set(wt.SUPERVISOR_MODEL_PROVIDERS)


def test_gui_key_lines_show_found_or_missing_only(tmp_path):
    app, controller = make_gui(tmp_path, env={"DEEPSEEK_API_KEY": SECRET_DEEPSEEK})
    try:
        app._provider_status_cache = controller.provider_status()
        app._render_provider_status()

        assert (
            app.provider_key_vars["DEEPSEEK_API_KEY"].get() == "DEEPSEEK_API_KEY: FOUND"
        )
        assert (
            app.provider_key_vars["OPENAI_API_KEY"].get() == "OPENAI_API_KEY: MISSING"
        )
        assert (
            app.provider_key_vars["ANTHROPIC_API_KEY"].get()
            == "ANTHROPIC_API_KEY: MISSING"
        )
        blob = "\n".join(widget_texts(app))
        assert SECRET_DEEPSEEK not in blob
        assert "FOUND" in blob and "MISSING" in blob
    finally:
        app.on_close()
        controller.db.close()


def test_gui_check_providers_runs_off_the_tk_thread(gui):
    app, controller = gui
    stub_codex_health(controller)

    start = time.perf_counter()
    app.check_providers()
    assert time.perf_counter() - start < 1.0
    assert app.busy is True  # the probes run in a worker thread

    run_until_idle(app)

    assert app.busy is False
    assert app.provider_status_vars[ID_DEEPSEEK].get() == "DeepSeek: READY"
    assert app.provider_status_vars[ID_OPENAI].get() == "OpenAI: READY"
    assert app.provider_status_vars[PROVIDER_CODEX_LEGACY].get() == (
        "Codex Legacy: READY"
    )
    assert app.provider_status_vars[ID_ANTHROPIC].get() == "Claude: UNAVAILABLE"
    assert "SUPERVISOR PROVIDERS" in app.details.get("1.0", "end")
    assert "no completion" in app.footer_var.get()


def test_gui_never_exposes_a_key_value_anywhere(gui):
    app, controller = gui  # the injected environment HAS both API keys

    app.save_policy()  # a real policy is persisted while keys are present
    app.check_providers()
    run_until_idle(app)

    blob = "\n".join(widget_texts(app))
    for secret in SECRETS:
        assert secret not in blob
    assert "FOUND" in blob

    for secret in SECRETS:
        assert secret not in str(controller.provider_status()[ID_DEEPSEEK]["detail"])
    assert SECRET_DEEPSEEK not in policy_file(controller).read_text(encoding="utf-8")
    for row in controller.db.db.execute("SELECT details FROM events").fetchall():
        for secret in SECRETS:
            assert secret not in str(row["details"])


def supervisor_frame(app):
    """The "Supervisor" LabelFrame (found by label text, layout-agnostic)."""

    def search(widget):
        for child in widget.winfo_children():
            if isinstance(child, wt.ttk.LabelFrame) and str(child.cget("text")) == (
                "Supervisor"
            ):
                return child
            found = search(child)
            if found is not None:
                return found
        return None

    return search(app.root)


def test_gui_has_no_api_key_entry_field(gui):
    app, _ = gui
    frame = supervisor_frame(app)
    assert frame is not None
    model_vars = {str(var) for var in app.supervisor_model_vars.values()}
    entries: list[str] = []

    def walk(widget):
        for child in widget.winfo_children():
            # ttk.Combobox subclasses ttk.Entry: only true text fields count.
            if isinstance(child, wt.ttk.Entry) and not isinstance(
                child, wt.ttk.Combobox
            ):
                entries.append(str(child.cget("textvariable")))
            walk(child)

    walk(frame)
    # The ONLY text fields in the Supervisor section are the three model
    # overrides: V1 has no API key entry field.
    assert len(entries) == 3
    assert set(entries) == model_vars


def test_gui_button_labels_are_generic_supervisor(gui):
    app, _ = gui
    assert str(app.btn_plan["text"]) == "1. Run Supervisor PLAN"
    assert str(app.btn_review["text"]) == "4. Run Supervisor REVIEW"
    assert str(app.btn_retry_review["text"]) == "Retry Supervisor REVIEW"
    assert str(app.btn_check_providers["text"]) == "Check Providers"
    assert str(app.btn_save_policy["text"]) == "Save Policy"


def test_gui_supervisor_buttons_need_an_open_project():
    try:
        app = wt.TandemGUI()
    except Exception as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk unavailable: {exc}")
    app.root.withdraw()
    try:
        app._set_buttons()
        assert str(app.btn_check_providers["state"]) == "disabled"
        assert str(app.btn_save_policy["state"]) == "disabled"
        app.refresh()  # no project open: must reset, never crash
        assert app.supervisor_plan_route_var.get() == "Current PLAN route: —"
        assert app.supervisor_review_route_var.get() == "Current REVIEW route: —"
        app.check_providers()  # a no-op, never a crash
    finally:
        app.on_close()


def test_fsm_state_strings_and_transitions_are_untouched():
    source = SRC.read_text(encoding="utf-8")
    for state in FSM_STATES:
        assert f'"{state}"' in source

    assert wt.CODEX_REVIEW_RETRYABLE == "CODEX_REVIEW_RETRYABLE"
    assert set(FSM_STATES) <= set(wt.VALID_STATES)
    assert set(wt.ALLOWED_TRANSITIONS) <= set(wt.VALID_STATES)

    # Only the GUI-facing labels changed; the persisted states did not.
    assert '"Run Codex PLAN"' not in source
    assert '"Run Codex REVIEW"' not in source
    assert '"Retry Codex REVIEW"' not in source
    assert '"1. Run Supervisor PLAN"' in source
    assert '"4. Run Supervisor REVIEW"' in source
    assert '"Retry Supervisor REVIEW"' in source


# --------------------------------------------------------------------------- #
# 29-31  Current STEP compatibility: an accepted report + a Codex PLAN are
#        REVIEWable by an API provider (no PLAN re-run, no attempt consumed)
# --------------------------------------------------------------------------- #

APPROVE_REVIEW = {
    "verdict": "APPROVE",
    "issues": [],
    "required_changes": [],
    "checks": ["tests green"],
    "architecture_risk": "none",
    "summary": "api review ok",
}


def plan_rows(controller) -> list[dict]:
    return [
        dict(row)
        for row in controller.db.db.execute(
            "SELECT step_no, attempt, plan_json FROM codex_plans ORDER BY id"
        ).fetchall()
    ]


def event_payloads(controller, event_type: str) -> list[dict]:
    rows = controller.db.db.execute(
        "SELECT details FROM events WHERE event_type=? ORDER BY id", (event_type,)
    ).fetchall()
    return [json.loads(row["details"]) for row in rows]


@pytest.mark.parametrize("provider_id", [ID_DEEPSEEK, ID_OPENAI])
def test_current_report_is_reviewable_by_an_api_provider(tmp_path, provider_id):
    transport = ready_transport(responses=[chat_response(APPROVE_REVIEW)])
    controller = controller_for(tmp_path, transport=transport, env=ENV_WITH_KEYS)
    try:
        # 1) The PLAN is produced by LEGACY CODEX under the default policy.
        stub_runner(controller, PLAN_OK)
        plan = controller.run_codex_plan()
        assert plan["decision"] == "PLAN_READY"
        assert (
            controller.supervisor_router.calls[-1]["provider"]
            == PROVIDER_CODEX_LEGACY
        )
        plans_before = plan_rows(controller)

        # 2) The operator switches REVIEW to an API provider.
        controller.save_supervisor_policy(
            wt.SupervisorPolicy(
                mode="MANUAL",
                plan_provider=PROVIDER_CODEX_LEGACY,
                review_provider=provider_id,
                fallback_chain=(),
                max_provider_retries=0,
            )
        )

        # 3) The accepted Cline report for the current attempt exists.
        controller.approve_plan_and_dispatch_cline(reason="m9 test")
        wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
        controller.process_cline_report()
        step = controller.db.get_step(1)
        assert step.state == "CLINE_REPORT_RECEIVED"
        attempt_before = step.attempt
        report_before = controller.db.cline_report_for_attempt(1, attempt_before)
        assert report_before is not None

        # 4) REVIEW runs through the API provider - never through Codex/PLAN.
        codex_calls: list[str] = []
        original_runner = controller.codex.run_structured

        def guard(*args, **kwargs):
            codex_calls.append("codex")
            return original_runner(*args, **kwargs)

        controller.codex.run_structured = guard
        review = controller.run_supervisor_review()

        assert codex_calls == []  # no PLAN re-run, no second Cline run
        assert review["verdict"] == "APPROVE"
        assert review["step_no"] == 1
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"

        # The provider really was the API one, over HTTP, with its own key/model.
        posts = [r for r in transport.requests if r["method"] == "POST"]
        assert len(posts) == 1
        assert posts[0]["url"].endswith("/chat/completions")
        assert posts[0]["json"]["model"] == controller.supervisor_models()[provider_id]
        headers = {str(k).lower(): v for k, v in posts[0]["headers"].items()}
        expected_key = SECRET_DEEPSEEK if provider_id == ID_DEEPSEEK else SECRET_OPENAI
        other_key = SECRET_OPENAI if provider_id == ID_DEEPSEEK else SECRET_DEEPSEEK
        assert headers["authorization"] == f"Bearer {expected_key}"
        assert other_key not in json.dumps(posts)

        # 5) Identity/attempt semantics: nothing consumed, nothing re-run.
        assert controller.db.get_step(1).attempt == attempt_before
        assert plan_rows(controller) == plans_before
        assert (
            controller.db.cline_report_for_attempt(1, attempt_before) == report_before
        )

        # 6) The routing is audited and carries no credential.
        results = event_payloads(controller, "SUPERVISOR_RESULT")
        assert results
        assert results[-1]["provider"] == provider_id
        assert results[-1]["step_no"] == 1
        for row in controller.db.db.execute("SELECT details FROM events").fetchall():
            for secret in SECRETS:
                assert secret not in str(row["details"])
    finally:
        controller.db.close()


def test_gui_review_button_uses_the_saved_api_provider(tmp_path, monkeypatch):
    """The GUI REVIEW button honours the persisted provider choice (M9)."""
    monkeypatch.setattr(wt.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(wt.messagebox, "showerror", lambda *args, **kwargs: None)

    transport = ready_transport(responses=[chat_response(APPROVE_REVIEW)])
    app, controller = make_gui(tmp_path, transport=transport, env=ENV_WITH_KEYS)
    try:
        stub_runner(controller, PLAN_OK)
        prepare_accepted_report(controller)
        step = controller.db.get_step(1)
        assert step.state == "CLINE_REPORT_RECEIVED"
        attempt_before = step.attempt

        app.supervisor_mode_var.set("MANUAL")
        app.supervisor_plan_var.set("Codex Legacy")
        app.supervisor_review_var.set("OpenAI")
        app.save_policy()

        app.run_codex_review()
        run_until_idle(app)

        assert controller.db.get_step(1).state == "REVIEW_APPROVED"
        assert controller.db.get_step(1).attempt == attempt_before
        assert '"verdict": "APPROVE"' in app.details.get("1.0", "end")
        posts = [r for r in transport.requests if r["method"] == "POST"]
        assert len(posts) == 1
        assert "api.openai.com" in posts[0]["url"]
        assert posts[0]["json"]["model"] == "gpt-4o-mini"
    finally:
        app.on_close()
        controller.db.close()


