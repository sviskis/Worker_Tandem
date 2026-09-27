"""Final OFFLINE supervisor smoke test (multi-provider, zero live calls).

Eight end-to-end scenarios over the M1-M9 supervisor stack with **no** live API,
**no** API keys and **no** paid request:

1. AUTO PLAN (LOW)                       -> deepseek, no fallback
2. AUTO REVIEW fallback (MEDIUM)         -> deepseek 429 -> openai APPROVE
3. HIGH risk                             -> openai (PLAN) / Claude (REVIEW)
4. all providers fail                    -> bounded retries, deferrable, no loss
5. MANUAL mode                           -> only the selected provider
6. secret safety                         -> 0 leaks anywhere
7. no network / no Codex                 -> 0 live calls
8. policy save / load                    -> round-trips, no keys stored

Every provider is a :class:`FakeSupervisorAdapter` or a REAL adapter driven by
``httpx.MockTransport`` (which never opens a socket). A :class:`LiveCallGuard`
fails the test immediately if anything tries to reach the real world:
``socket.create_connection``, the real ``httpx`` transport, ``subprocess`` (the
Codex CLI included), ``shutil.which`` and ``os.system``.

Everything runs inside pytest ``tmp_path`` project fixtures: no live
``.worker_tandem`` runtime, no repository SQLite is touched.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import socket
import sqlite3
import subprocess
import sys

import httpx
import pytest

from supervisor import (
    DEFAULT_ANTHROPIC_MODEL,
    ID_ANTHROPIC,
    ID_DEEPSEEK,
    ID_OPENAI,
    PROVIDER_CODEX_LEGACY,
    RETRYABLE_FALLBACK_CATEGORIES,
    FakeOutcome,
    FakeSupervisorAdapter,
    HttpTransport,
    ProviderConfig,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorReviewRequest,
    SupervisorRouter,
    SupervisorRunError,
    default_auto_policy,
    fake_plan_payload,
    fake_review_payload,
    supervisor_plan_schema,
    supervisor_review_schema,
    validate_plan_payload,
)

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"
REPO = pathlib.Path(__file__).resolve().parent


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem_smoke", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_smoke"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()

#: Fake credentials: they are USED (headers) but must never be persisted/logged.
SECRETS = {
    "DEEPSEEK_API_KEY": "ds-test-secret",
    "OPENAI_API_KEY": "sk-test-secret",
    "ANTHROPIC_API_KEY": "sk-ant-test-secret",
}
SECRET_VALUES = tuple(SECRETS.values())

PLAN_DEF = {
    "plan_version": "smoke-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "SMOKE STEP",
            "description": "offline smoke step",
            "risk": "LOW",
        },
    ],
}
PLAN_DEF_HIGH = {
    "plan_version": "smoke-high-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "SMOKE HIGH STEP",
            "description": "offline high-risk smoke step",
            "risk": "HIGH",
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
REVIEW_APPROVE = {
    "verdict": "APPROVE",
    "issues": [],
    "required_changes": [],
    "checks": ["tests green"],
    "architecture_risk": "none",
    "summary": "approved offline",
}

DEEPSEEK_HOST = "api.deepseek.com"
OPENAI_HOST = "api.openai.com"
ANTHROPIC_HOST = "api.anthropic.com"


# --------------------------------------------------------------------------- #
# Live-call guard (scenario 7) - active for EVERY scenario
# --------------------------------------------------------------------------- #


class LiveCallGuard:
    """Blocks the network outright and keeps the Codex CLI from ever running.

    * ``socket.create_connection``, the real ``httpx`` transport and ``os.system``
      are blocked **unconditionally**: this smoke test must be 100% offline.
    * ``shutil.which`` / ``subprocess`` are blocked (and counted) **only for the
      Codex CLI**. The pre-existing REVIEW context capture legitimately shells
      out to the local ``git`` binary for a read-only diff, so those calls are
      counted separately (``git_calls``) and are never a provider call.
    """

    #: Always fatal: nothing may leave the process.
    ABSOLUTE = (
        "socket.create_connection",
        "httpx.HTTPTransport.handle_request",
        "os.system",
    )
    #: Fatal only for the Codex CLI (never for the local git utility).
    CODEX = ("codex_which", "codex_process")

    def __init__(self, monkeypatch):
        self.counts = {name: 0 for name in self.ABSOLUTE}
        self.counts.update({name: 0 for name in self.CODEX})
        self.counts["git_calls"] = 0
        self._monkeypatch = monkeypatch
        self._project_root: pathlib.Path | None = None
        self.install()

    def _block(self, name):
        def boom(*args, **kwargs):
            self.counts[name] += 1
            raise AssertionError(f"LIVE CALL ATTEMPTED: {name}")

        return boom

    @staticmethod
    def _is_codex(command) -> bool:
        first = command.split()[0] if isinstance(command, str) else None
        if isinstance(command, (list, tuple)) and command:
            first = str(command[0])
        return "codex" in str(first or "").lower()

    def install(self):
        real_which = shutil.which
        real_run = subprocess.run
        real_popen = subprocess.Popen

        def which(name, *args, **kwargs):
            if "codex" in str(name).lower():
                self.counts["codex_which"] += 1
                raise AssertionError("CODEX CLI RESOLUTION ATTEMPTED (forbidden)")
            if str(name) == "git":
                self.counts["git_calls"] += 1
            return real_which(name, *args, **kwargs)

        def run(command, *args, **kwargs):
            if self._is_codex(command):
                self.counts["codex_process"] += 1
                raise AssertionError("CODEX PROCESS EXECUTED (forbidden)")
            cwd = kwargs.get("cwd")
            if cwd is not None and self._project_root is not None:
                assert str(cwd).startswith(str(self._project_root)), cwd
            self.counts["git_calls"] += 1
            return real_run(command, *args, **kwargs)

        def popen(command, *args, **kwargs):
            if self._is_codex(command):
                self.counts["codex_process"] += 1
                raise AssertionError("CODEX PROCESS SPAWNED (forbidden)")
            self.counts["git_calls"] += 1
            return real_popen(command, *args, **kwargs)

        for name, owner, attribute in (
            ("socket.create_connection", socket, "create_connection"),
            (
                "httpx.HTTPTransport.handle_request",
                httpx.HTTPTransport,
                "handle_request",
            ),
            ("os.system", os, "system"),
        ):
            self._monkeypatch.setattr(owner, attribute, self._block(name))
        self._monkeypatch.setattr(shutil, "which", which)
        self._monkeypatch.setattr(subprocess, "run", run)
        self._monkeypatch.setattr(subprocess, "Popen", popen)

    @property
    def codex_calls(self) -> int:
        return sum(self.counts[name] for name in self.CODEX)

    def bind_project(self, path) -> "LiveCallGuard":
        """Scope every allowed local subprocess to the temp project."""
        self._project_root = pathlib.Path(path)
        return self

    def assert_silent(self) -> None:
        """No network call, no Codex CLI, no shell escape."""
        for name in self.ABSOLUTE + self.CODEX:
            assert self.counts[name] == 0, self.counts


@pytest.fixture
def guard(monkeypatch, project):
    return LiveCallGuard(monkeypatch).bind_project(project)


@pytest.fixture
def project(tmp_path):
    """An isolated temp project (never the live .worker_tandem runtime)."""
    root = tmp_path / "smoke_project"
    root.mkdir()
    return root


# --------------------------------------------------------------------------- #
# Offline HTTP doubles: httpx.MockTransport never opens a socket
# --------------------------------------------------------------------------- #


def health_ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [{"id": "offline"}]})


def chat_ok(payload: dict):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
            },
        )

    return handler


def http_error(status: int, message: str, code: str = ""):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message, "code": code}})

    return handler


def http_timeout(_request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("offline timeout")


def messages_ok(payload: dict):
    """The Anthropic Messages envelope: ``content`` is a LIST of typed blocks."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_smoke",
                "type": "message",
                "role": "assistant",
                "model": DEFAULT_ANTHROPIC_MODEL,
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )

    return handler


def provider(post):
    """One endpoint: GET -> models, POST -> the scenario's scripted answer."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return health_ok(request)
        return post(request)

    return handler


class OfflineHttp:
    """Recording, MockTransport-backed stand-in for the API providers."""

    def __init__(self, *, deepseek=None, openai=None, anthropic=None):
        self.requests: list[httpx.Request] = []
        self.deepseek = deepseek or provider(health_ok)
        self.openai = openai or provider(health_ok)
        self.anthropic = anthropic or provider(health_ok)
        self._client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == OPENAI_HOST:
            return self.openai(request)
        if request.url.host == ANTHROPIC_HOST:
            return self.anthropic(request)
        return self.deepseek(request)

    @property
    def transport(self) -> HttpTransport:
        return HttpTransport(client=self._client)

    def close(self) -> None:
        self._client.close()

    def hosts(self, method: str | None = None) -> list[str]:
        return [
            r.url.host for r in self.requests if method is None or r.method == method
        ]

    def bodies(self, host: str | None = None) -> list[dict]:
        return [
            json.loads(r.content.decode("utf-8"))
            for r in self.requests
            if r.method == "POST" and (host is None or r.url.host == host)
        ]

    def authorization(self, host: str) -> list[str]:
        return [
            r.headers.get("authorization", "")
            for r in self.requests
            if r.url.host == host
        ]


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def controller_for(project, http=None, *, policy=None, plan_def=PLAN_DEF, env=None):
    """A controller in a temp project with an injected transport/environment."""
    controller = wt.TandemController(
        project,
        supervisor_transport=(http.transport if http is not None else None),
        supervisor_env=dict(SECRETS if env is None else env),
    )
    controller.db.import_plan(plan_def)
    if policy is not None:
        controller.save_supervisor_policy(policy)
    return controller


def stub_codex_health(controller, *, available=True):
    """The Codex CLI is never resolved or executed (the guard would fire)."""
    controller.codex.available = lambda: (available, "codex-stub (never executed)")


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


def prepare_accepted_report(controller, *, reason="smoke test"):
    """A Codex-produced PLAN plus an accepted Cline report for the attempt."""
    stub_runner(controller, PLAN_OK)
    controller.run_codex_plan()
    controller.approve_plan_and_dispatch_cline(reason=reason)
    wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
    controller.process_cline_report()
    return controller.db.get_step(1)


def events(controller, event_type: str) -> list[dict]:
    rows = controller.db.db.execute(
        "SELECT details FROM events WHERE event_type=? ORDER BY id", (event_type,)
    ).fetchall()
    return [json.loads(row["details"]) for row in rows]


def dir_snapshot(path: pathlib.Path) -> dict:
    return {
        str(p.relative_to(path)): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def spy_on(controller, attribute: str) -> list:
    """Record every call of ``controller.db.<attribute>`` without changing it."""
    calls: list = []
    original = getattr(controller.db, attribute)

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    setattr(controller.db, attribute, spy)
    return calls


def spy_on_runner(controller) -> list:
    """Record every Codex runner call (a Cline redispatch would appear here)."""
    calls: list = []
    original = controller.codex.run_structured

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    controller.codex.run_structured = spy
    return calls


def assert_no_secrets(*blobs) -> None:
    text = json.dumps(blobs, default=str)
    for name, value in SECRETS.items():
        assert value not in text, f"{name} leaked"


def policy_file_text(controller) -> str:
    """The persisted policy file, or "" when the project never saved one."""
    path = controller.supervisor_policy_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def repo_runtime_snapshot() -> dict:
    """The repository's LIVE .worker_tandem runtime (must stay untouched)."""
    root = REPO / ".worker_tandem"
    if not root.exists():
        return {"exists": False}
    return {
        "exists": True,
        "entries": sorted(str(p.relative_to(root)) for p in root.rglob("*")),
    }


def make_gui(project, *, transport=None, env=None):
    """A withdrawn Tk app bound to a temp controller (headless-safe)."""
    try:
        app = wt.TandemGUI()
    except Exception as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk unavailable: {exc}")
    app.root.withdraw()
    controller = wt.TandemController(
        project,
        supervisor_transport=transport,
        supervisor_env=dict(SECRETS if env is None else env),
    )
    controller.db.import_plan(PLAN_DEF)
    controller.codex.available = lambda: (True, "codex-stub (never executed)")
    app.controller = controller
    app.refresh()
    return app, controller


# --------------------------------------------------------------------------- #
# Router-level fakes (used by the HIGH-risk, all-fail and MANUAL scenarios)
# --------------------------------------------------------------------------- #


def fake_provider(provider_id, *, plan=None, review=None, enabled=True):
    """A deterministic fake provider bound to a canonical id."""
    model = f"{provider_id}-model"
    return FakeSupervisorAdapter(
        provider_id,
        model,
        plan_outcomes=plan,
        review_outcomes=review,
        config=ProviderConfig(
            provider_id=provider_id,
            display_name=provider_id,
            model=model,
            enabled=enabled,
        ),
    )


def failing_fake(provider_id, category, *, operation="PLAN", enabled=True):
    error = SupervisorRunError(
        f"{provider_id} failed offline: {category}",
        provider=provider_id,
        model=f"{provider_id}-model",
        operation=operation,
        category=category,
    )
    outcome = FakeOutcome(error=error)
    if operation == "REVIEW":
        return fake_provider(provider_id, review=[outcome], enabled=enabled)
    return fake_provider(provider_id, plan=[outcome], enabled=enabled)


def plan_request(risk="LOW", step_no=1, attempt=1):
    return SupervisorPlanRequest(
        step_no=step_no,
        attempt=attempt,
        risk=risk,
        requires_human=False,
        prompt="PLAN prompt",
        context={},
        schema=supervisor_plan_schema(),
    )


def review_request(risk="LOW", step_no=1, attempt=1):
    return SupervisorReviewRequest(
        step_no=step_no,
        attempt=attempt,
        risk=risk,
        requires_human=False,
        prompt="REVIEW prompt",
        context={},
        schema=supervisor_review_schema(),
    )


class Recording:
    """Collects the router's audit events in memory."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, dict(payload)))

    def of(self, event_type: str) -> list[dict]:
        return [payload for kind, payload in self.events if kind == event_type]

    def types(self) -> list[str]:
        return [kind for kind, _ in self.events]


# --------------------------------------------------------------------------- #
# SCENARIO 1 - AUTO PLAN (LOW): deepseek, canonical, no fallback
# --------------------------------------------------------------------------- #


def test_scenario_1_auto_plan_low_selects_deepseek(guard, project):
    http = OfflineHttp(
        deepseek=provider(chat_ok(PLAN_OK)),
        openai=provider(chat_ok(PLAN_OK)),
    )
    controller = controller_for(project, http, policy=default_auto_policy())
    try:
        stub_codex_health(controller)
        stub_runner(controller, PLAN_OK)
        step = controller.db.get_step(1)

        plan = controller.run_codex_plan()

        # The router selected deepseek for AUTO/LOW ...
        assert controller.supervisor_router.calls[-1]["provider"] == ID_DEEPSEEK
        result = events(controller, "SUPERVISOR_RESULT")[-1]
        assert result["provider"] == ID_DEEPSEEK
        assert result["provider_attempt"] == 1
        assert result["fallback_from"] is None
        assert result["fallback_reason"] is None
        assert result["route_chain"][0] == ID_DEEPSEEK
        assert result["operation"] == "PLAN"

        # ... with no fallback (OpenAI was never contacted).
        assert events(controller, "SUPERVISOR_FALLBACK") == []
        assert http.hosts("POST") == [DEEPSEEK_HOST]
        assert set(http.hosts("GET")) <= {DEEPSEEK_HOST}

        # The provider payload passed canonical validation, unchanged.
        provider_payload = {k: v for k, v in plan.items() if k != "step_no"}
        assert validate_plan_payload(provider_payload) == []
        assert provider_payload == PLAN_OK

        # The controller owns identity: it lives outside the provider payload.
        assert plan == {**PLAN_OK, "step_no": step.step_no}
        assert controller.db.latest_codex_plan(step.step_no) == plan
        body = http.bodies(DEEPSEEK_HOST)[-1]
        assert "step_no" not in body
        assert "attempt" not in body
        assert body["model"] == controller.supervisor_models()[ID_DEEPSEEK]
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is False

        assert controller.db.get_step(1).state == "PLAN_READY"
        assert_no_secrets(plan, events(controller, "SUPERVISOR_SELECTED"))
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# SCENARIO 2 - AUTO REVIEW (MEDIUM): deepseek 429 -> openai APPROVE
# --------------------------------------------------------------------------- #

PLAN_DEF_MEDIUM = {
    "plan_version": "smoke-med-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "SMOKE MEDIUM STEP",
            "description": "offline medium-risk smoke step",
            "risk": "MEDIUM",
        },
    ],
}


def test_scenario_2_auto_review_falls_back_to_openai(guard, project):
    http = OfflineHttp(
        deepseek=provider(
            http_error(
                429, "Rate limit reached for requests", code="rate_limit_exceeded"
            )
        ),
        openai=provider(chat_ok(REVIEW_APPROVE)),
    )
    controller = controller_for(project, http, plan_def=PLAN_DEF_MEDIUM)
    try:
        stub_codex_health(controller)
        # The PLAN is produced by Codex under the default MANUAL policy, so the
        # accepted-report setup stays independent of the REVIEW routing.
        prepare_accepted_report(controller)
        step = controller.db.get_step(1)
        assert step.state == "CLINE_REPORT_RECEIVED"

        # AUTO/REVIEW is what this scenario exercises.
        controller.save_supervisor_policy(default_auto_policy())
        assert controller.supervisor_policy.is_auto is True
        assert controller.current_routes()["review"]["risk"] == "MEDIUM"

        attempt_before = step.attempt
        report_before = controller.db.cline_report_for_attempt(1, attempt_before)
        assert report_before is not None
        to_cline_before = dir_snapshot(controller.db.to_cline)

        increments = spy_on(controller, "increment_attempt")
        runner_calls = spy_on_runner(controller)
        selected_before = len(events(controller, "SUPERVISOR_SELECTED"))
        http.requests.clear()  # only the REVIEW traffic is measured below

        review = controller.run_supervisor_review()

        # 1) deepseek first, 2) one same-provider retry, 3) fallback to openai.
        selected = events(controller, "SUPERVISOR_SELECTED")[selected_before:]
        assert [e["provider"] for e in selected] == [
            ID_DEEPSEEK,
            ID_DEEPSEEK,
            ID_OPENAI,
        ]
        fallback = events(controller, "SUPERVISOR_FALLBACK")[-1]
        assert fallback["fallback_from"] == ID_DEEPSEEK
        assert fallback["fallback_to"] == ID_OPENAI
        assert fallback["category"] in RETRYABLE_FALLBACK_CATEGORIES
        assert fallback["retryable"] is True
        assert [r.url.host for r in http.requests if r.method == "POST"] == [
            DEEPSEEK_HOST,
            DEEPSEEK_HOST,
            OPENAI_HOST,
        ]

        # The verdict came from openai and is the canonical APPROVE.
        assert review["verdict"] == "APPROVE"
        assert review["step_no"] == 1
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"

        # Route metadata records the fallback.
        history = controller.supervisor_router.route_history[-1]
        assert history["provider"] == ID_OPENAI
        assert history["fallback_from"] == ID_DEEPSEEK
        assert history["fallback_reason"] in RETRYABLE_FALLBACK_CATEGORIES
        assert history["route_chain"][0] == ID_DEEPSEEK
        assert history["route_chain"][1] == ID_OPENAI
        assert history["provider_attempt"] == 1
        assert history["latency_ms"] is not None
        assert events(controller, "SUPERVISOR_RESULT")[-1]["fallback_from"] == (
            ID_DEEPSEEK
        )

        # Nothing was consumed, redispatched or mutated.
        assert increments == []
        assert runner_calls == []
        assert controller.db.get_step(1).attempt == attempt_before
        assert (
            controller.db.cline_report_for_attempt(1, attempt_before) == report_before
        )
        assert dir_snapshot(controller.db.to_cline) == to_cline_before
        assert_no_secrets(review, controller.supervisor_router.route_history)
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# SCENARIO 3 - HIGH risk: openai for PLAN, Claude for REVIEW
# --------------------------------------------------------------------------- #


def test_scenario_3_high_risk_declared_route_is_openai_then_claude(guard):
    policy = default_auto_policy()
    assert policy.candidates_for("PLAN", "HIGH")[0] == ID_OPENAI
    assert policy.candidates_for("REVIEW", "HIGH")[0] == ID_ANTHROPIC

    deepseek = fake_provider(ID_DEEPSEEK)
    openai = fake_provider(ID_OPENAI)
    anthropic = fake_provider(ID_ANTHROPIC)  # the M7 slot, exercised with a fake
    legacy = fake_provider(PROVIDER_CODEX_LEGACY)
    router = SupervisorRouter(
        {
            ID_DEEPSEEK: deepseek,
            ID_OPENAI: openai,
            ID_ANTHROPIC: anthropic,
            PROVIDER_CODEX_LEGACY: legacy,
        },
        policy,
    )

    plan = router.plan(plan_request("HIGH"))
    review = router.review(review_request("HIGH"))

    assert plan.provider == ID_OPENAI
    assert review.provider == ID_ANTHROPIC
    assert plan.payload == fake_plan_payload()
    assert review.payload == fake_review_payload()
    assert review.payload["verdict"] == "APPROVE"
    assert plan.fallback_from is None and review.fallback_from is None
    assert [call[0] for call in openai.calls] == ["PLAN"]
    assert [call[0] for call in anthropic.calls] == ["REVIEW"]
    assert deepseek.calls == []
    assert legacy.calls == []
    assert [entry["provider"] for entry in router.route_history] == [
        ID_OPENAI,
        ID_ANTHROPIC,
    ]
    guard.assert_silent()


def test_scenario_3_high_risk_plan_through_the_controller_uses_openai(guard, project):
    http = OfflineHttp(
        deepseek=provider(chat_ok(PLAN_OK)),
        openai=provider(chat_ok(PLAN_OK)),
    )
    controller = controller_for(
        project, http, policy=default_auto_policy(), plan_def=PLAN_DEF_HIGH
    )
    try:
        stub_codex_health(controller)
        stub_runner(controller, PLAN_OK)

        plan = controller.run_codex_plan()

        assert plan["decision"] == "PLAN_READY"
        assert controller.current_routes()["plan"]["risk"] == "HIGH"
        assert controller.supervisor_router.calls[-1]["provider"] == ID_OPENAI
        assert events(controller, "SUPERVISOR_RESULT")[-1]["provider"] == ID_OPENAI
        assert http.hosts("POST") == [OPENAI_HOST]
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_scenario_3_high_risk_review_runs_through_the_registered_claude(guard, project):
    """Claude is the declared HIGH/REVIEW preference and is registered (M7.4).

    M7.4 gave ``anthropic`` a buildable adapter, so the declared preference is now
    the first candidate the router can actually call - the fallback to DeepSeek
    that M7.1 pinned is gone.
    """
    http = OfflineHttp(anthropic=provider(messages_ok(REVIEW_APPROVE)))
    controller = controller_for(project, http, plan_def=PLAN_DEF_HIGH)
    try:
        stub_codex_health(controller)
        prepare_accepted_report(controller)
        controller.save_supervisor_policy(default_auto_policy())

        routes = controller.current_routes()
        assert routes["review"]["risk"] == "HIGH"
        assert routes["review"]["primary"] == ID_ANTHROPIC  # declared preference
        assert ID_ANTHROPIC in controller.supervisor_router.registered_providers()
        assert routes["review"]["chain"][0] == ID_ANTHROPIC

        selected_before = len(events(controller, "SUPERVISOR_SELECTED"))
        review = controller.run_supervisor_review()

        assert review["verdict"] == "APPROVE"
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"
        assert [
            e["provider"]
            for e in events(controller, "SUPERVISOR_SELECTED")[selected_before:]
        ] == [ID_ANTHROPIC]
        # The Messages endpoint - not /chat/completions - and nothing else.
        assert http.hosts("POST") == [ANTHROPIC_HOST]
        assert events(controller, "SUPERVISOR_RESULT")[-1]["route_chain"][0] == (
            ID_ANTHROPIC
        )
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# SCENARIO 5 - MANUAL mode: only the selected provider is called
# --------------------------------------------------------------------------- #


def manual_policy(**overrides):
    values = {
        "mode": "MANUAL",
        "plan_provider": ID_ANTHROPIC,
        "review_provider": ID_ANTHROPIC,
        "fallback_chain": (ID_DEEPSEEK, ID_OPENAI, PROVIDER_CODEX_LEGACY),
        "max_provider_retries": 0,
    }
    values.update(overrides)
    return SupervisorPolicy(**values)


def manual_router(anthropic, policy=None):
    return SupervisorRouter(
        {
            ID_DEEPSEEK: fake_provider(ID_DEEPSEEK),
            ID_OPENAI: fake_provider(ID_OPENAI),
            ID_ANTHROPIC: anthropic,
            PROVIDER_CODEX_LEGACY: fake_provider(PROVIDER_CODEX_LEGACY),
        },
        policy or manual_policy(),
    )


def test_scenario_5_manual_calls_only_the_selected_provider(guard):
    anthropic = fake_provider(ID_ANTHROPIC)
    router = manual_router(anthropic)
    policy = router.policy
    others = [
        router.adapter_for(provider_id)
        for provider_id in (ID_DEEPSEEK, ID_OPENAI, PROVIDER_CODEX_LEGACY)
    ]

    # The chain exists but MANUAL ignores it: exactly one candidate.
    assert policy.candidates_for("PLAN", "HIGH") == (ID_ANTHROPIC,)
    assert policy.fallback_chain

    plan = router.plan(plan_request("LOW"))
    review = router.review(review_request("LOW"))

    assert plan.provider == ID_ANTHROPIC
    assert review.provider == ID_ANTHROPIC
    assert plan.payload == fake_plan_payload()
    assert review.payload["verdict"] == "APPROVE"
    assert plan.fallback_from is None and review.fallback_from is None
    assert [call[0] for call in anthropic.calls] == ["PLAN", "REVIEW"]
    for adapter in others:
        assert adapter.calls == [], adapter.provider_id
    guard.assert_silent()


def test_scenario_5_manual_falls_back_only_when_explicitly_enabled(guard):
    failing_anthropic = failing_fake(ID_ANTHROPIC, "rate_limit")
    router = manual_router(failing_anthropic)

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request("LOW"))

    assert info.value.category == "rate_limit"
    assert info.value.retryable is True
    assert len(failing_anthropic.calls) == 1  # no same-provider retry either
    for provider_id in (ID_DEEPSEEK, ID_OPENAI, PROVIDER_CODEX_LEGACY):
        assert router.adapter_for(provider_id).calls == []

    # Explicit opt-in: now (and only now) the chain is walked.
    opt_in = manual_router(
        failing_fake(ID_ANTHROPIC, "rate_limit"),
        policy=manual_policy(allow_manual_fallback=True),
    )
    result = opt_in.plan(plan_request("LOW"))

    assert result.provider == ID_DEEPSEEK
    assert result.fallback_from == ID_ANTHROPIC
    assert result.fallback_reason == "rate_limit"
    assert opt_in.adapter_for(ID_OPENAI).calls == []
    assert opt_in.adapter_for(PROVIDER_CODEX_LEGACY).calls == []
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# SCENARIO 4 - every provider fails: bounded, deferrable, nothing lost
# --------------------------------------------------------------------------- #


def test_scenario_4_all_providers_fail_is_bounded_and_retryable(guard):
    recorded = Recording()
    deepseek = failing_fake(ID_DEEPSEEK, "timeout")
    openai = failing_fake(ID_OPENAI, "provider_unavailable")
    anthropic = failing_fake(ID_ANTHROPIC, "rate_limit")
    legacy = fake_provider(PROVIDER_CODEX_LEGACY, enabled=False)
    router = SupervisorRouter(
        {
            ID_DEEPSEEK: deepseek,
            ID_OPENAI: openai,
            ID_ANTHROPIC: anthropic,
            PROVIDER_CODEX_LEGACY: legacy,
        },
        default_auto_policy(),
        recorded,
    )

    with pytest.raises(SupervisorRunError) as info:
        router.plan(plan_request("LOW"))

    # Bounded: one retry per provider, then it stops. No infinite loop.
    assert len(deepseek.calls) == 2
    assert len(openai.calls) == 2
    assert len(anthropic.calls) == 2
    assert legacy.calls == []  # disabled -> skipped, never attempted
    assert len(router.calls) == 4  # each candidate considered exactly once

    assert info.value.retryable is True  # provider-side => deferrable
    assert info.value.category == "rate_limit"
    all_failed = recorded.of("SUPERVISOR_ALL_FAILED")[-1]
    assert all_failed["retryable"] is True
    assert all_failed["route_chain"] == [
        ID_DEEPSEEK,
        ID_OPENAI,
        ID_ANTHROPIC,
        PROVIDER_CODEX_LEGACY,
    ]
    assert len(recorded.of("SUPERVISOR_SELECTED")) == 6
    hops = recorded.of("SUPERVISOR_FALLBACK")
    assert [h["fallback_from"] for h in hops] == [ID_DEEPSEEK, ID_OPENAI, ID_ANTHROPIC]
    assert [h["fallback_to"] for h in hops] == [
        ID_OPENAI,
        ID_ANTHROPIC,
        PROVIDER_CODEX_LEGACY,
    ]
    # The last hop targets a DISABLED provider: it is skipped, never attempted.
    assert legacy.calls == []
    assert recorded.of("SUPERVISOR_RESULT") == []
    guard.assert_silent()


def test_scenario_4_review_defers_without_losing_the_accepted_report(guard, project):
    http = OfflineHttp(
        deepseek=provider(http_timeout),
        openai=provider(http_error(503, "provider unavailable")),
    )
    controller = controller_for(project, http)
    try:
        stub_codex_health(controller)
        prepare_accepted_report(controller)

        # AUTO REVIEW over the exact scenario mix: deepseek=timeout,
        # openai=unavailable, Claude=rate_limit, Codex Legacy=disabled.
        controller.save_supervisor_policy(default_auto_policy())
        recorded = Recording()

        def emit(event_type, payload):
            recorded(event_type, payload)
            controller._supervisor_emit(event_type, payload)

        controller.supervisor_router = SupervisorRouter(
            {
                ID_DEEPSEEK: controller.supervisor_provider(ID_DEEPSEEK),
                ID_OPENAI: controller.supervisor_provider(ID_OPENAI),
                ID_ANTHROPIC: failing_fake(
                    ID_ANTHROPIC, "rate_limit", operation="REVIEW"
                ),
                PROVIDER_CODEX_LEGACY: fake_provider(
                    PROVIDER_CODEX_LEGACY, enabled=False
                ),
            },
            controller.supervisor_policy,
            emit,
        )

        step = controller.db.get_step(1)
        attempt_before = step.attempt
        report_before = controller.db.cline_report_for_attempt(1, attempt_before)
        increments = spy_on(controller, "increment_attempt")

        result = controller.run_supervisor_review()

        # Blocking-but-retryable: the existing deferred state, no project failure.
        assert result["deferred"] is True
        assert result["retryable"] is True
        assert result["state"] == "CODEX_REVIEW_RETRYABLE"

        # Nothing was consumed, lost or corrupted.
        assert increments == []
        step_after = controller.db.get_step(1)
        assert step_after.attempt == attempt_before
        assert step_after.state == "CODEX_REVIEW_RETRYABLE"
        assert controller.db.cline_report_for_attempt(1, attempt_before) == report_before
        assert controller.db.has_codex_review(1, attempt_before) is False
        assert step_after.state in wt.VALID_STATES
        assert wt.ALLOWED_TRANSITIONS["CODEX_REVIEW_RETRYABLE"]
        assert (
            controller.db.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        )

        all_failed = recorded.of("SUPERVISOR_ALL_FAILED")
        assert len(all_failed) == 1
        assert all_failed[0]["retryable"] is True
        assert events(controller, "SUPERVISOR_ALL_FAILED")
        assert [r.url.host for r in http.requests if r.method == "POST"] == [
            DEEPSEEK_HOST,
            DEEPSEEK_HOST,
            OPENAI_HOST,
            OPENAI_HOST,
        ]
        assert_no_secrets(result, events(controller, "SUPERVISOR_ALL_FAILED"))
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# SCENARIO 6 - secret safety: 0 leaks in errors/audit/telemetry/health/GUI
# --------------------------------------------------------------------------- #


def exception_text(exc: BaseException) -> dict:
    """Everything an exception (and its whole cause chain) can expose."""
    parts: list = [str(exc), repr(exc), getattr(exc, "sanitized_message", "")]
    to_dict = getattr(exc, "to_dict", None)
    if callable(to_dict):
        parts.append(to_dict())
    seen = {id(exc)}
    current = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        parts.append(repr(current))
        inner = getattr(current, "to_dict", None)
        if callable(inner):
            parts.append(inner())
        current = current.__cause__ or current.__context__
    return {"blob": json.dumps(parts, default=str), "parts": parts}


def secret_bearing_auth_error(request: httpx.Request) -> httpx.Response:
    """A 401 whose body echoes the injected credentials."""
    return httpx.Response(
        401,
        json={
            "error": {
                "message": (
                    "invalid api_key="
                    f"{SECRETS['DEEPSEEK_API_KEY']} / Bearer {SECRETS['OPENAI_API_KEY']} "
                    f"/ x-api-key: {SECRETS['ANTHROPIC_API_KEY']}"
                ),
                "code": "invalid_api_key",
            }
        },
    )


def test_scenario_6_zero_secret_leaks(guard, project, tmp_path):
    http = OfflineHttp(
        # BOTH the completion and the health probe answer with a 401 that echoes
        # the credentials, so every surface is exercised.
        deepseek=secret_bearing_auth_error,
        openai=secret_bearing_auth_error,
    )
    controller = controller_for(
        project,
        http,
        # The health skip is disabled so the auth failure reaches the PLAN call
        # instead of being pre-empted by KEY/AUTH health routing.
        policy=default_auto_policy(skip_unhealthy_providers=False),
    )
    try:
        stub_codex_health(controller)
        stub_runner(controller, PLAN_OK)

        # (1) The credentials really WERE used - otherwise this proves nothing.
        with pytest.raises(Exception) as info:
            controller.run_codex_plan()
        assert http.authorization(DEEPSEEK_HOST)[0] == (
            f"Bearer {SECRETS['DEEPSEEK_API_KEY']}"
        )

        # (2) No leak in the exception chain.
        raised = exception_text(info.value)
        assert_no_secrets(raised)

        # (3) No leak in the audit trail (events table details).
        rows = [
            str(row["details"])
            for row in controller.db.db.execute("SELECT details FROM events").fetchall()
        ]
        assert rows
        assert_no_secrets(rows)

        # (4) Telemetry / result metadata (router traces carry usage + latency).
        assert_no_secrets(
            controller.supervisor_router.calls,
            controller.supervisor_router.route_history,
        )

        # (5) Provider health detail (the data the GUI renders).
        status = controller.provider_status()
        assert status[ID_DEEPSEEK]["health"] == "AUTH_ERROR"
        assert status[ID_DEEPSEEK]["key"] == "FOUND"
        assert "***" in json.dumps(status)  # redacted, not silently dropped
        assert_no_secrets(status)
        # Both credentials were used as headers, and neither was echoed back.
        assert http.authorization(OPENAI_HOST)[0] == (
            f"Bearer {SECRETS['OPENAI_API_KEY']}"
        )
        assert SECRETS["DEEPSEEK_API_KEY"] not in policy_file_text(controller)

        # (6) The GUI must not show it either.
        app, app_controller = make_gui(tmp_path / "gui", transport=http.transport)
        try:
            app._provider_status_cache = app_controller.provider_status()
            app._render_provider_status()
            texts = [str(var.get()) for var in app.provider_key_vars.values()]
            texts += [str(var.get()) for var in app.provider_status_vars.values()]
            texts.append(app.details.get("1.0", "end"))
            assert any("FOUND" in text for text in texts)
            assert_no_secrets(texts)
        finally:
            try:
                app.on_close()
            except Exception:
                pass
            app_controller.db.close()
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# SCENARIO 7 - zero live calls: no network, no Codex CLI
# --------------------------------------------------------------------------- #


def auto_answering_endpoint(request: httpx.Request) -> httpx.Response:
    """GET -> models, POST -> PLAN or REVIEW depending on the prompt."""
    if request.method == "GET":
        return health_ok(request)
    text = request.content.decode("utf-8")
    return chat_ok(REVIEW_APPROVE if "verdict" in text else PLAN_OK)(request)


def test_scenario_7_zero_live_calls_in_a_full_flow(guard, project):
    repo_before = repo_runtime_snapshot()
    http = OfflineHttp(
        deepseek=auto_answering_endpoint,
        openai=auto_answering_endpoint,
    )
    controller = controller_for(
        project, http, policy=default_auto_policy(), plan_def=PLAN_DEF_MEDIUM
    )
    try:
        stub_codex_health(controller)
        # The only transport is the MockTransport-backed one: nothing can dial out.
        assert isinstance(http._client._transport, httpx.MockTransport)

        stub_runner(controller, PLAN_OK)
        plan = controller.run_codex_plan()
        assert plan["decision"] == "PLAN_READY"
        assert controller.supervisor_router.calls[-1]["provider"] == ID_DEEPSEEK

        controller.approve_plan_and_dispatch_cline(reason="smoke")
        wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
        controller.process_cline_report()
        review = controller.run_supervisor_review()
        assert review["verdict"] == "APPROVE"
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"

        # Health probes are offline too.
        status = controller.provider_status()
        assert status[ID_DEEPSEEK]["status"] == "READY"
        assert status[PROVIDER_CODEX_LEGACY]["status"] == "READY"
    finally:
        http.close()
        controller.db.close()
    # Not one byte left the process: no socket, no real httpx, no shell.
    guard.assert_silent()
    assert repo_runtime_snapshot() == repo_before


def test_scenario_7_codex_cli_can_never_be_resolved(guard, project):
    http = OfflineHttp()
    controller = controller_for(project, http, policy=default_auto_policy())
    try:
        # Deliberately NOT stubbing the Codex probe: the guard must intercept the
        # CLI resolution attempt, and no process may be spawned.
        status = controller.provider_status()

        assert guard.counts["codex_which"] >= 1  # the attempt was intercepted
        assert guard.counts["codex_process"] == 0  # ... and nothing ran
        assert status[PROVIDER_CODEX_LEGACY]["health"] == "PROVIDER_ERROR"
        assert "codex" in status[PROVIDER_CODEX_LEGACY]["detail"].lower()
        # The API providers stay healthy: only the CLI probe was blocked.
        assert status[ID_DEEPSEEK]["status"] == "READY"
    finally:
        http.close()
        controller.db.close()
    for name in guard.ABSOLUTE:
        assert guard.counts[name] == 0, guard.counts
    assert guard.counts["codex_process"] == 0


# --------------------------------------------------------------------------- #
# SCENARIO 8 - policy save / load round-trip without any key material
# --------------------------------------------------------------------------- #


def test_scenario_8_policy_round_trip_stores_no_keys(guard, project):
    repo_before = repo_runtime_snapshot()
    http = OfflineHttp()
    controller = controller_for(project, http)
    try:
        controller.save_supervisor_policy(
            SupervisorPolicy(
                mode="MANUAL",
                plan_provider=ID_DEEPSEEK,
                review_provider=ID_OPENAI,
                budget_mode="PREMIUM",
            ),
            models={
                ID_DEEPSEEK: "deepseek-reasoner",
                ID_OPENAI: "gpt-4o-mini-smoke",
            },
        )
        path = controller.supervisor_policy_path()
        assert path.name == "supervisor_policy.json"
        assert path.parent.name == ".worker_tandem"
        assert str(path).startswith(str(project))  # temp project only
        assert not list(path.parent.glob("*.tmp"))  # atomic write left nothing
    finally:
        controller.db.close()

    # Reload in a project that has NO keys at all: the policy still loads.
    reloaded = wt.TandemController(
        project,
        supervisor_transport=http.transport,
        supervisor_env={},
    )
    try:
        policy = reloaded.supervisor_policy
        assert policy.mode == "MANUAL"
        assert policy.plan_provider == ID_DEEPSEEK
        assert policy.review_provider == ID_OPENAI
        assert policy.budget_mode == "PREMIUM"
        assert reloaded.supervisor_models()[ID_DEEPSEEK] == "deepseek-reasoner"
        assert reloaded.supervisor_models()[ID_OPENAI] == "gpt-4o-mini-smoke"
        assert reloaded.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
            ID_DEEPSEEK,
            ID_OPENAI,
        )
        # The persisted file carries names/ids only - never a credential.
        text = policy_file_text(reloaded)
        for name, value in SECRETS.items():
            assert value not in text, f"{name} value stored"
            assert name not in text, f"{name} name stored"
        for forbidden in ("api_key", "token", "secret", "bearer"):
            assert forbidden not in text.lower()
        assert reloaded.supervisor_key_status() == {
            "DEEPSEEK_API_KEY": "MISSING",
            "OPENAI_API_KEY": "MISSING",
            "ANTHROPIC_API_KEY": "MISSING",
        }
    finally:
        reloaded.db.close()

    # The live repository runtime was never written to.
    assert repo_runtime_snapshot() == repo_before
    http.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# Summary matrix: the numbers the report must show
# --------------------------------------------------------------------------- #


def tree_secret_hits(root: pathlib.Path) -> list[str]:
    """Any file under ``root`` (SQLite, JSON, artifacts, logs) holding a secret."""
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_bytes().decode("utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        for name, value in SECRETS.items():
            if value in text:
                hits.append(f"{path.relative_to(root)}::{name}")
    return hits


def test_offline_matrix_summary(guard, project):
    """One pass over the whole stack with the report's counters asserted."""
    repo_before = repo_runtime_snapshot()
    http = OfflineHttp(
        deepseek=auto_answering_endpoint,
        openai=auto_answering_endpoint,
    )
    controller = controller_for(
        project, http, policy=default_auto_policy(), plan_def=PLAN_DEF_MEDIUM
    )
    try:
        stub_codex_health(controller)
        stub_runner(controller, PLAN_OK)
        plan = controller.run_codex_plan()
        controller.approve_plan_and_dispatch_cline(reason="smoke")
        wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
        controller.process_cline_report()
        review = controller.run_supervisor_review()
        status = controller.provider_status()

        assert plan["decision"] == "PLAN_READY"
        assert review["verdict"] == "APPROVE"

        # SECRET_LEAKS = 0 - not in an exception, an audit row, a health detail,
        # nor anywhere in the temp project runtime (SQLite included).
        assert_no_secrets(
            plan,
            review,
            status,
            events(controller, "SUPERVISOR_SELECTED"),
            events(controller, "SUPERVISOR_RESULT"),
            controller.supervisor_router.route_history,
            controller.supervisor_router.calls,
        )
        assert tree_secret_hits(controller.db.project_root) == []
        assert tree_secret_hits(project) == []

        # LIVE_NETWORK_CALLS = 0 (only the MockTransport handler answered) and
        # CODEX_CALLS = 0 (the CLI was never resolved or spawned).
        assert guard.counts["socket.create_connection"] == 0
        assert guard.counts["httpx.HTTPTransport.handle_request"] == 0
        assert guard.counts["os.system"] == 0
        assert guard.codex_calls == 0
        assert http.requests, "the offline transport must have been exercised"
        assert isinstance(http._client._transport, httpx.MockTransport)
    finally:
        http.close()
        controller.db.close()

    assert repo_runtime_snapshot() == repo_before
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# M7.5 FINAL OFFLINE VERIFICATION - all FOUR providers, no live traffic
#
#   codex_legacy + deepseek + openai + anthropic
#
# Scenarios A-G below run the same LiveCallGuard as the rest of this file, so
# LIVE_NETWORK_CALLS / CODEX_CALLS / SECRET_LEAKS are asserted, not hoped for.
# --------------------------------------------------------------------------- #


def messages_error(status: int, error_type: str, message: str = "offline error"):
    """Anthropic error envelope (``authentication_error`` etc.)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"type": "error", "error": {"type": error_type, "message": message}},
        )

    return handler


def review_policy(*candidates: str) -> SupervisorPolicy:
    """An AUTO policy whose REVIEW chain is exactly ``candidates``."""
    routes = {
        risk: {
            "PLAN": (candidates[0],),
            "REVIEW": tuple(candidates),
        }
        for risk in ("LOW", "MEDIUM", "HIGH")
    }
    return SupervisorPolicy(
        mode="AUTO",
        risk_routes=routes,
        fallback_chain=(),
        max_provider_retries=0,
        allow_codex_legacy=False,
    )


def db_count(controller, table: str) -> int:
    return controller.db.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def dispatch_count(controller) -> int:
    """How often Cline was dispatched (a rerun would add one)."""
    return controller.db.db.execute(
        "SELECT COUNT(*) FROM events WHERE event_type LIKE '%->CLINE_DISPATCHED'"
    ).fetchone()[0]


def test_m75_scenario_a_manual_anthropic_plan_is_valid_and_never_falls_back(
    guard, project
):
    """A: MANUAL Anthropic PLAN -> valid canonical PLAN, no fallback."""
    http = OfflineHttp(anthropic=provider(messages_ok(PLAN_OK)))
    controller = controller_for(
        project, http, policy=manual_policy(), plan_def=PLAN_DEF_MEDIUM
    )
    try:
        stub_codex_health(controller)
        assert controller.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
            ID_ANTHROPIC,
        )

        plan = controller.run_codex_plan()

        # A valid, canonical PLAN - exactly the fixture, no provider extras.
        assert plan["decision"] == "PLAN_READY"
        assert plan["goal"] == PLAN_OK["goal"]
        assert set(plan) >= set(PLAN_OK)
        assert "step_no" not in plan or plan["step_no"] == 1  # controller identity

        # Claude did it, on the Messages endpoint, and nothing else was called.
        assert controller.supervisor_router.calls[-1]["provider"] == ID_ANTHROPIC
        assert events(controller, "SUPERVISOR_FALLBACK") == []
        assert [e["provider"] for e in events(controller, "SUPERVISOR_SELECTED")] == [
            ID_ANTHROPIC
        ]
        assert http.hosts("POST") == [ANTHROPIC_HOST]
        assert http.authorization(ANTHROPIC_HOST) == [""]  # x-api-key, not bearer
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_m75_scenario_b_manual_anthropic_review_approves_and_keeps_the_report(
    guard, project
):
    """B: MANUAL Anthropic REVIEW -> APPROVE, report preserved, attempt unchanged."""
    http = OfflineHttp(anthropic=provider(messages_ok(REVIEW_APPROVE)))
    controller = controller_for(project, http)
    try:
        stub_codex_health(controller)
        prepare_accepted_report(controller)  # Codex PLAN + accepted Cline report
        controller.save_supervisor_policy(manual_policy())

        step = controller.db.get_step(1)
        attempt_before = step.attempt
        report_before = controller.db.cline_report_for_attempt(1, attempt_before)
        plans_before = db_count(controller, "codex_plans")
        dispatches_before = dispatch_count(controller)
        increments = spy_on(controller, "increment_attempt")

        review = controller.run_supervisor_review()

        assert review["verdict"] == "APPROVE"
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"
        assert events(controller, "SUPERVISOR_FALLBACK") == []
        assert [e["provider"] for e in events(controller, "SUPERVISOR_SELECTED")[-1:]] == [
            ID_ANTHROPIC
        ]
        assert http.hosts("POST") == [ANTHROPIC_HOST]

        # Nothing consumed, nothing lost, nothing re-run.
        step_after = controller.db.get_step(1)
        assert step_after.attempt == attempt_before == 1
        assert increments == []
        assert controller.db.cline_report_for_attempt(1, attempt_before) == report_before
        assert db_count(controller, "codex_plans") == plans_before
        assert dispatch_count(controller) == dispatches_before
        assert controller.db.has_codex_review(1, attempt_before) is True
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_m75_scenario_c_auto_review_falls_back_from_openai_to_anthropic(guard, project):
    """C: OpenAI temporary failure -> Anthropic -> APPROVE."""
    http = OfflineHttp(
        openai=provider(http_error(429, "rate limited", "rate_limit")),
        anthropic=provider(messages_ok(REVIEW_APPROVE)),
    )
    controller = controller_for(project, http)
    try:
        stub_codex_health(controller)
        prepare_accepted_report(controller)
        controller.save_supervisor_policy(review_policy(ID_OPENAI, ID_ANTHROPIC))

        step = controller.db.get_step(1)
        attempt_before = step.attempt
        report_before = controller.db.cline_report_for_attempt(1, attempt_before)

        review = controller.run_supervisor_review()

        assert review["verdict"] == "APPROVE"
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"

        hops = events(controller, "SUPERVISOR_FALLBACK")
        assert [h["fallback_from"] for h in hops] == [ID_OPENAI]
        assert [h["fallback_to"] for h in hops] == [ID_ANTHROPIC]
        assert events(controller, "SUPERVISOR_RESULT")[-1]["provider"] == ID_ANTHROPIC

        # One POST per provider: exactly one fallback, no retry storm, no loop.
        assert http.hosts("POST") == [OPENAI_HOST, ANTHROPIC_HOST]
        assert controller.db.get_step(1).attempt == attempt_before
        assert controller.db.cline_report_for_attempt(1, attempt_before) == report_before
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_m75_scenario_d_anthropic_temporary_failure_follows_the_policy(guard, project):
    """D: Anthropic temporary failure -> the router falls back per policy."""
    http = OfflineHttp(
        anthropic=provider(messages_error(500, "api_error", "claude exploded")),
        openai=provider(chat_ok(REVIEW_APPROVE)),
    )
    controller = controller_for(project, http)
    try:
        stub_codex_health(controller)
        prepare_accepted_report(controller)
        # Policy: Claude first, OpenAI next, no same-provider retry.
        controller.save_supervisor_policy(review_policy(ID_ANTHROPIC, ID_OPENAI))

        attempt_before = controller.db.get_step(1).attempt
        review = controller.run_supervisor_review()

        assert review["verdict"] == "APPROVE"
        hops = events(controller, "SUPERVISOR_FALLBACK")
        assert [h["fallback_from"] for h in hops] == [ID_ANTHROPIC]
        assert [h["fallback_to"] for h in hops] == [ID_OPENAI]
        assert events(controller, "SUPERVISOR_RESULT")[-1]["provider"] == ID_OPENAI

        assert http.hosts("POST") == [ANTHROPIC_HOST, OPENAI_HOST]
        assert controller.db.get_step(1).attempt == attempt_before
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"

        # A policy that lists Claude ONLY keeps it Claude-only: even though a
        # healthy OpenAI is registered, a temporary Claude failure defers instead
        # of hopping (proved on a fresh step).
        strict_root = project / "strict"
        strict_root.mkdir()
        strict_http = OfflineHttp(
            anthropic=provider(messages_error(503, "overloaded_error")),
            openai=provider(chat_ok(REVIEW_APPROVE)),
        )
        strict_controller = controller_for(strict_root, strict_http)
        try:
            stub_codex_health(strict_controller)
            prepare_accepted_report(strict_controller)
            strict_controller.save_supervisor_policy(review_policy(ID_ANTHROPIC))

            strict = strict_controller.run_supervisor_review()

            assert strict["deferred"] is True
            assert strict["retryable"] is True
            assert strict["provider"] == ID_ANTHROPIC
            assert strict["state"] == "CODEX_REVIEW_RETRYABLE"
            assert events(strict_controller, "SUPERVISOR_FALLBACK") == []
            assert strict_http.hosts("POST") == [ANTHROPIC_HOST]
            assert strict_controller.db.get_step(1).attempt == 1
        finally:
            strict_http.close()
            strict_controller.db.close()
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_m75_scenario_e_anthropic_auth_failure_never_retries_or_falls_back(
    guard, project
):
    """E: Anthropic auth failure -> no automatic retry/fallback when forbidden."""
    http = OfflineHttp(
        anthropic=provider(
            messages_error(401, "authentication_error", "invalid x-api-key")
        ),
        openai=provider(chat_ok(REVIEW_APPROVE)),
        deepseek=provider(chat_ok(REVIEW_APPROVE)),
    )
    controller = controller_for(project, http)
    try:
        stub_codex_health(controller)
        prepare_accepted_report(controller)
        # MANUAL Claude for REVIEW, fallback forbidden, zero provider retries.
        controller.save_supervisor_policy(
            manual_policy(
                plan_provider=PROVIDER_CODEX_LEGACY, review_provider=ID_ANTHROPIC
            )
        )
        selected_before = len(events(controller, "SUPERVISOR_SELECTED"))
        attempt_before = controller.db.get_step(1).attempt
        report_before = controller.db.cline_report_for_attempt(1, attempt_before)

        with pytest.raises(SupervisorRunError) as info:
            controller.run_supervisor_review()

        assert info.value.category == "auth"
        assert info.value.retryable is False

        # One attempt on Claude only: no same-provider retry, no chain walk.
        assert http.hosts("POST") == [ANTHROPIC_HOST]
        assert (
            len(events(controller, "SUPERVISOR_SELECTED")) - selected_before == 1
        )
        assert events(controller, "SUPERVISOR_FALLBACK") == []

        # Nothing consumed, nothing lost.
        assert controller.db.get_step(1).attempt == attempt_before == 1
        assert controller.db.cline_report_for_attempt(1, attempt_before) == report_before
        assert controller.db.has_codex_review(1, attempt_before) is False
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_m75_scenario_f_synthetic_anthropic_key_never_leaks(guard, project):
    """F: the synthetic ANTHROPIC_API_KEY never reaches a file, log or widget."""
    secret = SECRETS["ANTHROPIC_API_KEY"]

    def echoing_error(_request: httpx.Request) -> httpx.Response:
        """A hostile provider: it echoes the credential back in its error body."""
        return httpx.Response(
            401,
            json={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": f"invalid x-api-key: {secret}",
                },
            },
        )

    http = OfflineHttp(
        anthropic=provider(echoing_error),
        openai=provider(chat_ok(REVIEW_APPROVE)),
    )
    controller = controller_for(project, http)
    try:
        stub_codex_health(controller)
        prepare_accepted_report(controller)
        controller.save_supervisor_policy(
            manual_policy(
                plan_provider=PROVIDER_CODEX_LEGACY, review_provider=ID_ANTHROPIC
            )
        )
        status = controller.provider_status()

        with pytest.raises(SupervisorRunError) as info:
            controller.run_supervisor_review()

        # The key really was used as the credential on the one POST ...
        posts = [r for r in http.requests if r.method == "POST"]
        assert [r.headers.get("x-api-key", "") for r in posts] == [secret]

        # ... and it is nowhere else: not in the failure, telemetry, health, the
        # audit rows, the policy file, nor anywhere in the project runtime.
        assert_no_secrets(
            str(info.value),
            info.value.sanitized_message,
            events(controller, "SUPERVISOR_SELECTED"),
            events(controller, "SUPERVISOR_RESULT"),
            events(controller, "SUPERVISOR_ALL_FAILED"),
            status,
            controller.supervisor_router.route_history,
            controller.supervisor_router.calls,
            policy_file_text(controller),
        )
        assert tree_secret_hits(controller.db.project_root) == []
        assert tree_secret_hits(project) == []
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_m75_scenario_g_forbidden_paths_are_hard_blocks(guard, project):
    """G: socket / real httpx / Codex subprocess / shell are HARD failures.

    This test deliberately trips every trap, so it must NOT call
    ``guard.assert_silent()`` - it asserts the counters instead, which proves the
    traps fire rather than merely counting.
    """
    # 1) socket.create_connection - nothing may leave the process.
    with pytest.raises(AssertionError, match="socket.create_connection"):
        socket.create_connection(("api.anthropic.com", 443), timeout=1)
    assert guard.counts["socket.create_connection"] == 1

    # 2) the REAL httpx network transport (httpx.MockTransport stays allowed).
    with pytest.raises(Exception):
        httpx.Client(timeout=1).post(
            "https://api.anthropic.com/v1/messages",
            json={"model": "claude", "max_tokens": 1, "messages": []},
        )
    assert guard.counts["httpx.HTTPTransport.handle_request"] == 1

    # 3) the Codex CLI: resolution AND process spawn (plus a list-form command).
    with pytest.raises(AssertionError, match="CODEX CLI RESOLUTION"):
        shutil.which("codex")
    with pytest.raises(AssertionError, match="CODEX PROCESS EXECUTED"):
        subprocess.run(["codex", "--version"], check=False)
    with pytest.raises(AssertionError, match="CODEX PROCESS SPAWNED"):
        subprocess.Popen(["codex", "exec"])
    assert guard.counts["codex_which"] == 1
    assert guard.counts["codex_process"] == 2
    assert guard.codex_calls == 3

    # 4) no shell escape either.
    with pytest.raises(AssertionError, match="os.system"):
        os.system("codex --version")
    assert guard.counts["os.system"] == 1

    # 5) ... while the OFFLINE transport keeps working: only the real network is
    #    forbidden, which is exactly what makes the smoke suite meaningful.
    http = OfflineHttp(anthropic=provider(messages_ok(PLAN_OK)))
    try:
        assert isinstance(http._client._transport, httpx.MockTransport)
        assert http._client.get("https://api.anthropic.com/v1/models").status_code == 200
    finally:
        http.close()


def four_provider_endpoint(request: httpx.Request) -> httpx.Response:
    """GET -> models; POST -> PLAN/REVIEW in the caller's provider dialect."""
    if request.method == "GET":
        return health_ok(request)
    text = request.content.decode("utf-8")
    payload = REVIEW_APPROVE if "verdict" in text else PLAN_OK
    if "/messages" in request.url.path:  # Anthropic Messages API
        return messages_ok(payload)(request)
    return chat_ok(payload)(request)


def test_m75_final_offline_summary_all_four_providers(guard, project):
    """The M7.5 counters: LIVE_NETWORK_CALLS / CODEX_CALLS / SECRET_LEAKS = 0."""
    repo_before = repo_runtime_snapshot()
    http = OfflineHttp(
        deepseek=four_provider_endpoint,
        openai=four_provider_endpoint,
        anthropic=four_provider_endpoint,
    )
    controller = controller_for(project, http, plan_def=PLAN_DEF_MEDIUM)
    try:
        stub_codex_health(controller)

        # 1) codex_legacy PLAN under the M4 default policy: nothing but the stub.
        assert controller.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
        )
        stub_runner(controller, PLAN_OK)
        plan = controller.run_codex_plan()
        assert plan["decision"] == "PLAN_READY"
        assert controller.supervisor_router.calls[-1]["provider"] == (
            PROVIDER_CODEX_LEGACY
        )

        # 2) AUTO registers all four; MANUAL Claude then performs the REVIEW.
        controller.save_supervisor_policy(default_auto_policy())
        assert set(controller.supervisor_router.registered_providers()) == {
            PROVIDER_CODEX_LEGACY,
            ID_DEEPSEEK,
            ID_OPENAI,
            ID_ANTHROPIC,
        }

        controller.approve_plan_and_dispatch_cline(reason="m7.5")
        wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
        controller.process_cline_report()
        controller.save_supervisor_policy(
            manual_policy(
                plan_provider=PROVIDER_CODEX_LEGACY, review_provider=ID_ANTHROPIC
            )
        )
        review = controller.run_supervisor_review()
        assert review["verdict"] == "APPROVE"
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"

        # 3) Health of all four: three HTTP probes (MockTransport) + the stub CLI.
        status = controller.provider_status()
        assert status[ID_DEEPSEEK]["status"] == "READY"
        assert status[ID_OPENAI]["status"] == "READY"
        assert status[ID_ANTHROPIC]["status"] == "READY"
        assert status[PROVIDER_CODEX_LEGACY]["status"] == "READY"

        # 4) The three M7.5 counters.
        assert ANTHROPIC_HOST in http.hosts("POST")
        assert_no_secrets(
            plan,
            review,
            status,
            events(controller, "SUPERVISOR_SELECTED"),
            events(controller, "SUPERVISOR_RESULT"),
            controller.supervisor_router.route_history,
            controller.supervisor_router.calls,
        )
        counters = {
            "LIVE_NETWORK_CALLS": sum(guard.counts[name] for name in guard.ABSOLUTE),
            "CODEX_CALLS": guard.codex_calls,
            "SECRET_LEAKS": len(tree_secret_hits(controller.db.project_root))
            + len(tree_secret_hits(project)),
        }
        assert isinstance(http._client._transport, httpx.MockTransport)
        assert http.requests, "the offline transport must have been exercised"
        assert counters == {
            "LIVE_NETWORK_CALLS": 0,
            "CODEX_CALLS": 0,
            "SECRET_LEAKS": 0,
        }, counters
        print(
            "M7.5 SUMMARY: "
            f"LIVE_NETWORK_CALLS={counters['LIVE_NETWORK_CALLS']} "
            f"CODEX_CALLS={counters['CODEX_CALLS']} "
            f"SECRET_LEAKS={counters['SECRET_LEAKS']} "
            f"POST hosts={http.hosts('POST')}"
        )
    finally:
        http.close()
        controller.db.close()

    assert repo_runtime_snapshot() == repo_before
    guard.assert_silent()
