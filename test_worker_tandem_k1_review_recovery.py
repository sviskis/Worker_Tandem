"""K1: interrupted supervisor REVIEW recovery (CODEX_REVIEW_RUNNING -> CLINE_REPORT_RECEIVED).

The K1 defect
-------------
A hard interrupt while the supervisor REVIEW runs (the process dies after the step
moved to ``CODEX_REVIEW_RUNNING`` but before any review result was persisted) left
the project in a state with no legal way forward and no automatic recovery. The
task is to make the single explicit, audited, human-triggered recovery
``CODEX_REVIEW_RUNNING -> CLINE_REPORT_RECEIVED`` available - and to keep every
other guarantee intact (no provider call, no Cline redispatch, no PLAN rerun, no
attempt/step change, report and hash untouched, secret-free audit).

Design notes
------------
* DELIBERATELY SELF-CONTAINED: this module imports no shared/private test helper
  (in particular not the separate offline deep-validation helper module), so the
  K1 regression suite can be committed and run on its own. Everything it needs is
  defined here.
* Strictly OFFLINE: real provider adapters over ``httpx.MockTransport`` (no socket
  is ever opened), a stubbed Codex runner (no CLI, no subprocess), temporary SQLite
  projects and :class:`OfflineGuard`, which fails on any live attempt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import httpx
import pytest

from supervisor import (
    ID_ANTHROPIC,
    ID_DEEPSEEK,
    ID_OPENAI,
    PROVIDER_CODEX_LEGACY,
    AnthropicSupervisorAdapter,
    DeepSeekSupervisorAdapter,
    OpenAISupervisorAdapter,
    ProviderConfig,
    SupervisorPolicy,
    SupervisorRouter,
    default_anthropic_config,
    default_auto_policy,
    default_deepseek_config,
    default_openai_config,
    fake_plan_payload,
    fake_review_payload,
)
from supervisor.models import PLAN_OPERATION, REVIEW_OPERATION

# --------------------------------------------------------------------------- #
# The production module under test (the filename is not importable)
# --------------------------------------------------------------------------- #

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem(name: str = "worker_tandem_k1") -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SRC)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


wt = _load_worker_tandem()

#: Synthetic (FAKE) credentials - never real, never valid.
SECRET_DEEPSEEK = "sk-" + "d" * 30
SECRET_OPENAI = "sk-proj-" + "o" * 30
SECRET_ANTHROPIC = "sk-ant-" + "a" * 30

FAKE_SECRETS: dict[str, str] = {
    "DEEPSEEK_API_KEY": SECRET_DEEPSEEK,
    "OPENAI_API_KEY": SECRET_OPENAI,
    "ANTHROPIC_API_KEY": SECRET_ANTHROPIC,
}
SECRET_VALUES = tuple(FAKE_SECRETS.values())

DEEPSEEK_HOST = "api.deepseek.com"
OPENAI_HOST = "api.openai.com"
ANTHROPIC_HOST = "api.anthropic.com"

API_PROVIDERS = (ID_DEEPSEEK, ID_OPENAI, ID_ANTHROPIC)
ALL_PROVIDERS = API_PROVIDERS + (PROVIDER_CODEX_LEGACY,)

#: Anything that would have gone live; every test asserts these stay 0.
TALLY: dict[str, int] = {
    "live_network_calls": 0,
    "codex_process_calls": 0,
    "blocked_attempts": 0,
}


class OfflineGuard:
    """Blocks the network, the shell and the Codex CLI outright.

    Same semantics as the guard used by the other offline supervisor suites:
    ``socket.create_connection`` / the real ``httpx`` transport / ``os.system``
    are fatal, and so is any Codex CLI resolution or process. Local ``git`` reads
    are counted separately (the pre-existing REVIEW context capture shells out to
    ``git``), but only ever inside the temporary project.
    """

    ABSOLUTE = (
        "socket.create_connection",
        "httpx.HTTPTransport.handle_request",
        "os.system",
    )
    CODEX = ("codex_which", "codex_process")

    def __init__(self, monkeypatch, project_root=None):
        self.counts = {name: 0 for name in self.ABSOLUTE + self.CODEX}
        self.counts["git_calls"] = 0
        self._monkeypatch = monkeypatch
        self._project_root = pathlib.Path(project_root) if project_root else None
        self.install()

    def _block(self, name: str):
        def boom(*_args, **_kwargs):
            self.counts[name] += 1
            TALLY["blocked_attempts"] += 1
            if name in ("socket.create_connection", "httpx.HTTPTransport.handle_request"):
                TALLY["live_network_calls"] += 1
            if name.startswith("codex"):
                TALLY["codex_process_calls"] += 1
            raise AssertionError(f"LIVE CALL ATTEMPTED: {name}")

        return boom

    @staticmethod
    def _is_codex(command: Any) -> bool:
        first: Any = None
        if isinstance(command, (list, tuple)) and command:
            first = str(command[0])
        elif isinstance(command, str):
            parts = command.split()
            first = parts[0] if parts else ""
        return "codex" in str(first or "").lower()

    def install(self) -> None:
        real_which = shutil.which
        real_run = subprocess.run
        real_popen = subprocess.Popen

        def which(name, *args, **kwargs):
            if "codex" in str(name).lower():
                self.counts["codex_which"] += 1
                TALLY["codex_process_calls"] += 1
                raise AssertionError("CODEX CLI RESOLUTION ATTEMPTED (forbidden)")
            if str(name) == "git":
                self.counts["git_calls"] += 1
            return real_which(name, *args, **kwargs)

        def run(command, *args, **kwargs):
            if self._is_codex(command):
                self.counts["codex_process"] += 1
                TALLY["codex_process_calls"] += 1
                raise AssertionError("CODEX PROCESS EXECUTED (forbidden)")
            cwd = kwargs.get("cwd")
            if cwd is not None and self._project_root is not None:
                assert str(cwd).startswith(str(self._project_root)), cwd
            self.counts["git_calls"] += 1
            return real_run(command, *args, **kwargs)

        def popen(command, *args, **kwargs):
            if self._is_codex(command):
                self.counts["codex_process"] += 1
                TALLY["codex_process_calls"] += 1
                raise AssertionError("CODEX PROCESS SPAWNED (forbidden)")
            self.counts["git_calls"] += 1
            return real_popen(command, *args, **kwargs)

        for name, owner, attribute in (
            ("socket.create_connection", socket, "create_connection"),
            ("httpx.HTTPTransport.handle_request", httpx.HTTPTransport, "handle_request"),
            ("os.system", os, "system"),
        ):
            self._monkeypatch.setattr(owner, attribute, self._block(name))
        self._monkeypatch.setattr(shutil, "which", which)
        self._monkeypatch.setattr(subprocess, "run", run)
        self._monkeypatch.setattr(subprocess, "Popen", popen)

    def assert_silent(self) -> None:
        for name in self.ABSOLUTE + self.CODEX:
            assert self.counts[name] == 0, self.counts
        assert TALLY["live_network_calls"] == 0, TALLY
        assert TALLY["codex_process_calls"] == 0, TALLY


# --------------------------------------------------------------------------- #
# Canonical payloads, policy and provider failures
# --------------------------------------------------------------------------- #


def plan_payload(decision: str = "PLAN_READY", **overrides: Any) -> dict:
    payload = fake_plan_payload(decision=decision)
    payload.update(overrides)
    return payload


def review_payload(verdict: str = "APPROVE", **overrides: Any) -> dict:
    payload = fake_review_payload(verdict=verdict)
    payload.update(overrides)
    return payload


PLAN_OK = plan_payload()
REVIEW_APPROVE = review_payload("APPROVE")

PLAN_DEF = {
    "plan_version": "k1-offline-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "K1 STEP",
            "description": "interrupted-review recovery step",
            "risk": "LOW",
        },
    ],
}


def auto_policy(**overrides: Any) -> SupervisorPolicy:
    """The CURRENT production AUTO policy (``default_auto_policy``)."""
    return default_auto_policy(**overrides)


def codex_run_error(category: str = "timeout", *, retryable: bool = True):
    """A legacy-Codex failure object (no CLI, no process, no socket)."""
    return wt.CodexRunError(
        f"codex stub failure: {category}",
        category=category,
        retryable=retryable,
        returncode=1,
        provider="codex-stub",
        sanitized_error=f"codex stub failure: {category}",
        source="jsonl",
    )


def success_scripts() -> dict[str, dict]:
    """Every provider answers both operations successfully."""
    return {
        provider_id: {"plan": plan_payload(), "review": review_payload()}
        for provider_id in ALL_PROVIDERS
    }


# --------------------------------------------------------------------------- #
# Offline HTTP doubles (httpx.MockTransport: no socket is ever opened)
# --------------------------------------------------------------------------- #

USAGE_CHAT = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
USAGE_MESSAGES = {"input_tokens": 11, "output_tokens": 7}


def health_ok(_request=None) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [{"id": "offline"}]})


def status_error(status: int, message: str = ""):
    """A callable handler returning an HTTP error (never a socket)."""

    def handler(_request=None) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message}})

    return handler


def _request_operation(request: httpx.Request) -> str:
    """PLAN or REVIEW, inferred from the canonical schema named in the request."""
    text = request.content.decode("utf-8", "replace") if request.content else ""
    return REVIEW_OPERATION if "verdict" in text else PLAN_OPERATION


def chat_envelope(text: str, *, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": dict(USAGE_CHAT),
        },
    )


def messages_envelope(text: str, *, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "id": "msg_offline",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": dict(USAGE_MESSAGES),
        },
    )


def _call_handler(spec, request):
    """Call a handler with the request when it accepts one, else without."""
    try:
        count = len(inspect.signature(spec).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        count = 1
    return spec() if count == 0 else spec(request)


class Scripted:
    """One provider endpoint: GET -> health probe, POST -> scripted response.

    A POST spec may be a callable handler, a ``dict`` payload (canonical
    envelope), a ``str`` (raw model text inside the envelope), ``bytes`` (raw HTTP
    body) or an ``httpx.Response``. When only one operation is scripted, it
    answers both.
    """

    def __init__(
        self,
        *,
        plan: Any = None,
        review: Any = None,
        anthropic: bool = False,
        health: Any = None,
    ) -> None:
        self.plan = plan
        self.review = review
        self.anthropic = anthropic
        self.health = health if health is not None else health_ok
        self.calls: list[tuple[str, Optional[httpx.Request]]] = []

    @property
    def operations(self) -> list[str]:
        return [op for op, _ in self.calls]

    def _envelope(self, text: str) -> httpx.Response:
        if self.anthropic:
            return messages_envelope(text)
        return chat_envelope(text)

    def _respond(self, spec: Any) -> httpx.Response:
        if callable(spec):
            return _call_handler(spec, None)
        if isinstance(spec, (bytes, bytearray)):
            return httpx.Response(200, content=bytes(spec))
        if isinstance(spec, httpx.Response):
            return spec
        if isinstance(spec, str):
            return self._envelope(spec)
        if isinstance(spec, dict):
            return self._envelope(json.dumps(spec))
        raise AssertionError(f"unsupported scripted spec: {spec!r}")

    def __call__(self, request: Optional[httpx.Request] = None) -> httpx.Response:
        if request is None or request.method == "GET":
            self.calls.append(("HEALTH", request))
            return _call_handler(self.health, request)
        operation = _request_operation(request)
        self.calls.append((operation, request))
        spec = self.plan if operation == PLAN_OPERATION else self.review
        if spec is None:
            spec = self.review if operation == PLAN_OPERATION else self.plan
        if spec is None:
            spec = plan_payload() if operation == PLAN_OPERATION else review_payload()
        return self._respond(spec)


class OfflineHttp:
    """Host-routed, recording, MockTransport-backed stand-in for the providers."""

    def __init__(self, *, deepseek=None, openai=None, anthropic=None) -> None:
        self.requests: list[httpx.Request] = []
        self.deepseek = deepseek if deepseek is not None else Scripted()
        self.openai = openai if openai is not None else Scripted()
        self.anthropic = anthropic if anthropic is not None else Scripted(anthropic=True)
        self._client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == OPENAI_HOST:
            return self.openai(request)
        if request.url.host == ANTHROPIC_HOST:
            return self.anthropic(request)
        assert request.url.host == DEEPSEEK_HOST, request.url.host
        return self.deepseek(request)

    @property
    def transport(self):
        return wt.HttpTransport(client=self._client)

    def close(self) -> None:
        self._client.close()

    def scripts(self) -> dict[str, "Scripted"]:
        return {
            ID_DEEPSEEK: self.deepseek,
            ID_OPENAI: self.openai,
            ID_ANTHROPIC: self.anthropic,
        }


# --------------------------------------------------------------------------- #
# Real adapters over the offline transport + the controller's router
# --------------------------------------------------------------------------- #


class StubRunner:
    """A duck-typed structured-runner port: no Codex CLI, no subprocess."""

    def __init__(self, *, plan=None, review=None, model="gpt-5-codex", error=None) -> None:
        self.plan = plan if plan is not None else plan_payload()
        self.review = review if review is not None else review_payload()
        self.model = model
        self._error = error
        self.calls: list[dict] = []

    def available(self):
        return True, "codex-stub (never executed)"

    def run_structured(
        self,
        prompt,
        schema_path,
        output_path,
        trace_path,
        timeout_sec=None,
        reasoning_effort=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "schema_path": str(schema_path),
                "output_path": str(output_path),
                "trace_path": str(trace_path),
                "reasoning_effort": reasoning_effort,
            }
        )
        if self._error is not None:
            raise self._error
        payload = self.plan if "_plan" in str(output_path) else self.review
        if isinstance(payload, (dict, list)):
            return json.loads(json.dumps(payload))
        return payload


def codex_adapter(
    *,
    workdir,
    plan=None,
    review=None,
    error=None,
    model=None,
    enabled: bool = True,
):
    """The legacy Codex adapter over a stubbed runner (no CLI, no subprocess)."""
    return wt.LegacyCodexSupervisorAdapter(
        StubRunner(plan=plan, review=review, model=model or "gpt-5-codex", error=error),
        provider_id=PROVIDER_CODEX_LEGACY,
        model=model,
        workdir=pathlib.Path(workdir),
        identity_policy="strip_step_no",
        validation_mode="adapter",
        config=ProviderConfig(
            provider_id=PROVIDER_CODEX_LEGACY,
            display_name="Codex Legacy",
            model=model,
            enabled=enabled,
            supports_structured_output=True,
            cost_policy="free",
            priority=99,
        ),
    )


def real_adapter(provider_id: str, transport, *, env=None):
    """A REAL provider adapter over an injected (offline) transport."""
    maker, cls = {
        ID_DEEPSEEK: (default_deepseek_config, DeepSeekSupervisorAdapter),
        ID_OPENAI: (default_openai_config, OpenAISupervisorAdapter),
        ID_ANTHROPIC: (default_anthropic_config, AnthropicSupervisorAdapter),
    }[provider_id]
    return cls(maker(), transport, env=dict(FAKE_SECRETS if env is None else env))


def controller_router(controller, http: "OfflineHttp", policy, *, scripts=None, emit=None):
    """Point a controller at a scripted, offline provider router.

    ``scripts`` maps a provider id to ``{"plan": spec, "review": spec,
    "error": exception, "model": str, "enabled": bool}``. The controller's own
    audit hook is used by default, so routing events land in the project's
    ``events`` table. Returns ``(router, adapters)``.
    """
    base = controller.db.workdir / "providers"
    base.mkdir(parents=True, exist_ok=True)
    scripts = dict(scripts or {})
    adapters: dict[str, Any] = {}
    for provider_id in ALL_PROVIDERS:
        spec = dict(scripts.get(provider_id) or {})
        if provider_id == PROVIDER_CODEX_LEGACY:
            adapters[provider_id] = codex_adapter(
                workdir=base,
                plan=spec.get("plan"),
                review=spec.get("review"),
                error=spec.get("error"),
                model=spec.get("model"),
                enabled=bool(spec.get("enabled", True)),
            )
            continue
        script = http.scripts()[provider_id]
        if "plan" in spec:
            script.plan = spec["plan"]
        if "review" in spec:
            script.review = spec["review"]
        adapters[provider_id] = real_adapter(provider_id, http.transport)
    router = SupervisorRouter(
        dict(adapters), policy, emit if emit is not None else controller._supervisor_emit
    )
    controller.supervisor_policy = policy
    controller.supervisor_router = router
    return router, adapters


# --------------------------------------------------------------------------- #
# Controller plumbing over a temporary project
# --------------------------------------------------------------------------- #


def make_controller(
    project: pathlib.Path,
    *,
    transport: Any = None,
    env: Optional[Mapping[str, str]] = None,
    policy: Optional[SupervisorPolicy] = None,
):
    """A controller in a temp project: no live transport, no Codex CLI."""
    controller = wt.TandemController(
        project,
        supervisor_transport=transport,
        supervisor_env=None if env is None else dict(env),
    )
    controller.db.import_plan(PLAN_DEF)
    # The real Codex CLI is never resolved or executed by this suite.
    controller.codex.available = lambda: (True, "codex-stub (never executed)")
    stub_runner(controller, PLAN_OK)
    if policy is not None:
        controller.save_supervisor_policy(policy)
    return controller


def stub_runner(controller, payload: Any, *, review: Any = None) -> list:
    """Replace ``CodexRunner.run_structured`` with a deterministic, offline stub."""
    calls: list = []
    review_value = review if review is not None else review_payload()

    def fake(
        prompt,
        schema_path,
        output_path,
        trace_path,
        timeout_sec=None,
        reasoning_effort=None,
    ):
        calls.append(str(output_path))
        chosen = payload if "_plan" in str(output_path) else review_value
        if isinstance(chosen, (dict, list)):
            return json.loads(json.dumps(chosen))
        return chosen

    controller.codex.run_structured = fake
    return calls


def offline_controller(project, *, policy=None, transport=None):
    """A fresh controller over an EXISTING project (a simulated restart).

    Deliberately does NOT re-import the plan: reopening a project must preserve
    steps, attempts, reports and the policy exactly as they were left.
    """
    controller = wt.TandemController(
        project,
        supervisor_transport=transport,
        supervisor_env=dict(FAKE_SECRETS),
    )
    controller.codex.available = lambda: (True, "codex-stub (never executed)")
    stub_runner(controller, PLAN_OK)
    if policy is not None:
        controller.save_supervisor_policy(policy)
    return controller


def report_ok(step_no: int = 1, status: str = "DONE", **overrides: Any) -> dict:
    payload = {
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
    payload.update(overrides)
    return payload


DISPATCH_PATH = (
    "CODEX_PLAN_RUNNING",
    "PLAN_READY",
    "WAITING_HUMAN_APPROVAL",
    "CLINE_DISPATCHED",
)


def to_cline_dispatched(controller, step_no: int = 1, attempt: int = 1):
    """Move ``step_no`` to CLINE_DISPATCHED using only legal transitions."""
    for state in DISPATCH_PATH:
        controller.db.transition(step_no, state)
    while controller.db.get_step(step_no).attempt < attempt:
        controller.db.increment_attempt(step_no)
    return controller.db.get_step(step_no)


def write_report(controller, step_no: int = 1, status: str = "DONE", **overrides: Any):
    """Write the expected Cline report artifact atomically (no FSM change)."""
    path = controller.expected_cline_report_path(step_no)
    wt.write_json_atomic(path, report_ok(step_no, status, **overrides))
    return path


def accept_report(controller, step_no: int = 1, status: str = "DONE", **overrides: Any):
    write_report(controller, step_no, status, **overrides)
    return controller.process_cline_report()


def to_review_received(controller, step_no: int = 1, attempt: int = 1, status: str = "DONE"):
    to_cline_dispatched(controller, step_no, attempt)
    return accept_report(controller, step_no, status)


def report_bytes(controller, step_no: int = 1) -> bytes:
    return controller.expected_cline_report_path(step_no).read_bytes()


def events(controller, event_type: Optional[str] = None, step_no: Optional[int] = None):
    """Parsed audit events (``details`` is stored as JSON text)."""
    sql = "SELECT step_no, event_type, details FROM events"
    clauses: list[str] = []
    params: list[Any] = []
    if event_type is not None:
        clauses.append("event_type=?")
        params.append(event_type)
    if step_no is not None:
        clauses.append("step_no=?")
        params.append(step_no)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id"
    rows = controller.db.db.execute(sql, tuple(params)).fetchall()
    out: list[dict] = []
    for row in rows:
        try:
            parsed = json.loads(row["details"])
        except Exception:
            parsed = {"_raw": row["details"]}
        out.append(
            {
                "step_no": row["step_no"],
                "event_type": row["event_type"],
                "details": parsed,
                "raw": row["details"],
            }
        )
    return out


def _table_names(controller) -> list[str]:
    rows = controller.db.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(row["name"]) for row in rows]


def database_dump(controller) -> str:
    """Every row of every table as text (for secret/consistency scanning)."""
    chunks: list[str] = []
    for table in _table_names(controller):
        try:
            rows = controller.db.db.execute(f"SELECT * FROM {table}").fetchall()
        except Exception:
            continue
        for row in rows:
            chunks.append(f"{table}:{tuple(row)}")
    return "\n".join(chunks)


def table_counts(controller) -> dict[str, int]:
    """Row count per table (used to prove 'no unintended write')."""
    counts: dict[str, int] = {}
    for table in _table_names(controller):
        try:
            counts[table] = int(
                controller.db.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
        except Exception:
            counts[table] = -1
    return counts


# --------------------------------------------------------------------------- #
# Frozen-state snapshot + structural invariants + secret scanning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateSnapshot:
    """Everything a supervisor/provider operation must never change."""

    step_no: Optional[int]
    state: Optional[str]
    attempt: Optional[int]
    current_steps: tuple[int, ...]
    report_hash: Optional[str]
    report_file_sha: Optional[str]
    report_file_bytes: Optional[int]
    accepted_attempts: tuple
    review_attempts: tuple
    event_count: int
    policy: tuple
    models: tuple
    enabled: tuple
    routes: tuple

    def frozen(self) -> dict:
        return {
            "step_no": self.step_no,
            "state": self.state,
            "attempt": self.attempt,
            "current_steps": self.current_steps,
            "report_hash": self.report_hash,
            "report_file_sha": self.report_file_sha,
            "report_file_bytes": self.report_file_bytes,
            "accepted_attempts": self.accepted_attempts,
            "review_attempts": self.review_attempts,
            "event_count": self.event_count,
            "policy": self.policy,
            "models": self.models,
            "enabled": self.enabled,
            "routes": self.routes,
        }


def snapshot(controller) -> StateSnapshot:
    step = controller.db.current_step()
    current_steps = tuple(
        int(row["step_no"])
        for row in controller.db.db.execute(
            "SELECT step_no FROM steps WHERE state NOT IN ('VERIFIED','SKIPPED','ABORTED')"
        ).fetchall()
    )
    report_hash = None
    report_sha = None
    report_len = None
    if step is not None:
        stored = controller.db.cline_report_for_attempt(step.step_no, step.attempt)
        report_hash = stored[1] if stored else None
        path = controller.expected_cline_report_path(step.step_no)
        if path.exists():
            data = path.read_bytes()
            report_sha = hashlib.sha256(data).hexdigest()
            report_len = len(data)
    accepted = tuple(
        (int(row["step_no"]), int(row["attempt"]))
        for row in controller.db.db.execute(
            "SELECT step_no, attempt FROM cline_reports ORDER BY id"
        ).fetchall()
    )
    reviews = tuple(
        (int(row["step_no"]), int(row["attempt"]))
        for row in controller.db.db.execute(
            "SELECT step_no, attempt FROM codex_reviews ORDER BY id"
        ).fetchall()
    )
    event_count = int(controller.db.db.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    policy = controller.supervisor_policy
    return StateSnapshot(
        step_no=step.step_no if step else None,
        state=step.state if step else None,
        attempt=step.attempt if step else None,
        current_steps=current_steps,
        report_hash=report_hash,
        report_file_sha=report_sha,
        report_file_bytes=report_len,
        accepted_attempts=accepted,
        review_attempts=reviews,
        event_count=event_count,
        policy=tuple(sorted(policy.to_dict().items(), key=lambda item: item[0])),
        models=tuple(sorted(controller.supervisor_models().items())),
        enabled=tuple(controller.supervisor_enabled_providers()),
        routes=tuple(
            (name, tuple(route["chain"]))
            for name, route in sorted(controller.current_routes().items())
        ),
    )


def assert_frozen(before: StateSnapshot, after: StateSnapshot, **kwargs) -> None:
    """Assert the frozen state is identical (state/attempt optionally excepted)."""
    left, right = before.frozen(), after.frozen()
    if kwargs.get("allow_state"):
        left.pop("state", None)
        right.pop("state", None)
    if kwargs.get("allow_attempt"):
        left.pop("attempt", None)
        right.pop("attempt", None)
    assert left == right, f"unexpected mutation:\n{left}\n{right}"


def invariants(controller, *, require_attempt_started: bool = False) -> list[str]:
    """Structural invariants that must hold after every scenario."""
    problems: list[str] = []
    current = controller.db.current_step()
    current_steps = [
        int(row["step_no"])
        for row in controller.db.db.execute(
            "SELECT step_no FROM steps WHERE state NOT IN ('VERIFIED','SKIPPED','ABORTED')"
        ).fetchall()
    ]
    if len(current_steps) > 1:
        problems.append(f"more than one current step: {current_steps}")
    for row in controller.db.db.execute("SELECT step_no, state, attempt FROM steps"):
        state = str(row["state"])
        if state not in wt.VALID_STATES:
            problems.append(f"illegal FSM state persisted: {state}")
        if int(row["attempt"]) < 0:
            problems.append(f"negative attempt on step {row['step_no']}")
    if current is None:
        states = {
            str(row["state"]) for row in controller.db.db.execute("SELECT state FROM steps")
        }
        if states - set(wt.TERMINAL_STATES):
            problems.append("no current step while steps are still non-terminal")
    else:
        if current.state not in wt.VALID_STATES:
            problems.append(f"current state invalid: {current.state}")
        if (
            require_attempt_started
            and current.attempt < 1
            and current.state not in ("PENDING", "READY_FOR_CODEX_PLAN")
        ):
            problems.append(
                f"step {current.step_no} state {current.state} has attempt "
                f"{current.attempt} (< 1)"
            )
        stored = controller.db.cline_report_for_attempt(current.step_no, current.attempt)
        if stored is not None:
            report, digest = stored
            if report.get("step_no") != current.step_no:
                problems.append("accepted report does not belong to the current step")
            expected = hashlib.sha256(
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            ).hexdigest()
            if digest != expected:
                problems.append("stored report hash does not match stored report")
    # Provider payloads must never carry controller-owned FSM state.
    for table, column in (
        ("codex_plans", "plan_json"),
        ("codex_reviews", "review_json"),
    ):
        for row in controller.db.db.execute(f"SELECT {column} AS payload FROM {table}"):
            if '"state"' in str(row["payload"]):
                problems.append(f"{table}: provider payload carries controller state")
    integrity = controller.db.db.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        problems.append(f"sqlite integrity_check={integrity}")
    return problems


def assert_invariants(controller, **kwargs) -> None:
    problems = invariants(controller, **kwargs)
    assert not problems, "; ".join(problems)


def leaked_secrets(blob: Any, secrets: Sequence[str] = SECRET_VALUES) -> list[str]:
    text = (
        blob
        if isinstance(blob, str)
        else json.dumps(blob, ensure_ascii=False, default=repr)
    )
    return [secret for secret in secrets if secret and secret in text]


def scan_tree(root: pathlib.Path, secrets: Sequence[str] = SECRET_VALUES) -> list[str]:
    """Every file under ``root`` scanned for the synthetic secret values."""
    leaks: list[str] = []
    for path in sorted(pathlib.Path(root).rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:  # pragma: no cover - defensive
            continue
        for secret in secrets:
            if secret.encode("utf-8") in data:
                leaks.append(f"{path.name}: {secret[:6]}...")
    return leaks


# --------------------------------------------------------------------------- #
# Tk binding (headless-safe: the caller skips when Tk is unavailable)
# --------------------------------------------------------------------------- #


def make_gui(controller, *, offline_transport_http: Optional["OfflineHttp"] = None):
    """A withdrawn, offline-bound Tk app for ``controller``.

    Returns ``(app, http)``; the caller closes ``http``. Tk failures propagate so
    a test can ``pytest.skip("Tk unavailable: ...")``.
    """
    http = offline_transport_http if offline_transport_http is not None else OfflineHttp()
    controller._supervisor_transport = http.transport
    controller._supervisor_owned_transport = False
    app = wt.TandemGUI()
    app.root.withdraw()
    app.controller = controller
    app.refresh()
    return app, http


# --------------------------------------------------------------------------- #
# Fixtures + the K1 crash/scenario plumbing
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "k1_project"
    root.mkdir()
    return root


@pytest.fixture
def guard(monkeypatch, project):
    return OfflineGuard(monkeypatch, project)


@pytest.fixture
def controller(project, guard):
    ctrl = make_controller(project, env=FAKE_SECRETS)
    yield ctrl
    ctrl.close()


#: The reason text and audit event the recovery records (production constants).
RECOVER_REASON = wt.RECOVER_REVIEW_RUNNING_REASON
RECOVER_EVENT = wt.RECOVER_REVIEW_RUNNING_EVENT


def _interrupt_during_provider_call(project, *, where: str):
    """Run a REVIEW that dies at ``where`` and return the crashed controller."""
    controller = make_controller(project, env=FAKE_SECRETS)
    to_review_received(controller)
    http = OfflineHttp()
    controller_router(controller, http, auto_policy())

    adapter = controller.supervisor_router.adapter_for(ID_DEEPSEEK)
    original_review = adapter.review

    def exploding(*_args, **_kwargs):
        raise KeyboardInterrupt("simulated hard interrupt")

    if where == "provider-selected":
        adapter.review = exploding
    elif where == "before-persist":
        adapter.review = original_review
        controller.db.save_codex_review = exploding  # type: ignore[assignment]
    else:  # pragma: no cover - programming error
        raise AssertionError(where)

    with pytest.raises(KeyboardInterrupt):
        controller.run_supervisor_review()
    return controller, http


def crash_into_review_running(project):
    """Leave the project in CODEX_REVIEW_RUNNING with the report still accepted.

    (The K1 crash: the state moved, the process died before any review result was
    persisted.)
    """
    controller, http = _interrupt_during_provider_call(project, where="provider-selected")
    assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
    assert controller.db.has_codex_review(1, 1) is False
    return controller, http


# --------------------------------------------------------------------------- #
# The recovery itself
# --------------------------------------------------------------------------- #


def test_k1_valid_interrupted_review_recovery(project, guard):
    """The exact K1 scenario: recover, review, accept - no manual DB repair."""
    controller, http = crash_into_review_running(project)
    report_before = report_bytes(controller)
    stored_hash = controller.db.cline_report_for_attempt(1, 1)[1]
    attempt_before = controller.db.get_step(1).attempt
    http.close()
    controller.close()  # destroy the controller instance

    reopened = offline_controller(project)  # reopen the project
    try:
        # reopening runs NOTHING automatically
        assert snapshot(reopened).state == "CODEX_REVIEW_RUNNING"
        assert reopened.db.get_step(1).attempt == attempt_before == 1
        assert reopened.review_running_recovery_qualified() is True
        assert report_bytes(reopened) == report_before

        # the EXPLICIT human recovery
        recovered = reopened.recover_review_running_to_received(
            actor="human", reason=RECOVER_REASON
        )

        # the state is restored; step/attempt/report/hash are untouched
        assert recovered["state"] == "CLINE_REPORT_RECEIVED"
        assert recovered["from_state"] == "CODEX_REVIEW_RUNNING"
        assert recovered["to_state"] == "CLINE_REPORT_RECEIVED"
        assert recovered["step_no"] == 1
        assert recovered["attempt"] == 1
        assert recovered["report_hash"] == stored_hash
        assert reopened.db.get_step(1).state == "CLINE_REPORT_RECEIVED"
        assert reopened.db.get_step(1).attempt == 1
        assert report_bytes(reopened) == report_before
        assert reopened.db.cline_report_for_attempt(1, 1)[1] == stored_hash
        assert snapshot(reopened).accepted_attempts == ((1, 1),)

        # the normal REVIEW flow can now run - and it approves
        http_ok = OfflineHttp()
        controller_router(reopened, http_ok, auto_policy())
        review = reopened.run_supervisor_review()
        assert review["verdict"] == "APPROVE"
        assert reopened.db.get_step(1).state == "REVIEW_APPROVED"

        # accepting advances EXACTLY once (single-step project: no next step)
        assert reopened.accept_review(reason="accepted after recovery") is None
        assert reopened.db.get_step(1).state == "VERIFIED"
        with pytest.raises(RuntimeError):
            reopened.accept_review(reason="again")
        assert report_bytes(reopened) == report_before
        assert_invariants(reopened, require_attempt_started=True)
        http_ok.close()
    finally:
        reopened.close()
    guard.assert_silent()


def test_k1_recovery_rejected_in_other_states(project, guard):
    """State not CODEX_REVIEW_RUNNING => fail closed, nothing written."""
    controller = make_controller(project, env=FAKE_SECRETS)
    try:
        to_review_received(controller)  # CLINE_REPORT_RECEIVED
        for state in (
            "READY_FOR_CODEX_PLAN",
            "CLINE_REPORT_RECEIVED",
            "CODEX_REVIEW_RETRYABLE",
            "REVIEW_APPROVED",
            "FAILED",
            "VERIFIED",
        ):
            controller.db.db.execute(
                "UPDATE steps SET state = ? WHERE step_no = 1", (state,)
            )
            before = table_counts(controller)
            assert controller.review_running_recovery_qualified() is False
            with pytest.raises(RuntimeError):
                controller.recover_review_running_to_received(
                    actor="human", reason="should fail"
                )
            assert table_counts(controller) == before  # nothing written
            assert controller.db.get_step(1).state == state
    finally:
        controller.close()
    guard.assert_silent()


def test_k1_recovery_rejected_without_an_accepted_report(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        controller.db.db.execute("DELETE FROM cline_reports")
        before = table_counts(controller)

        assert controller.review_running_recovery_qualified() is False
        with pytest.raises(RuntimeError) as info:
            controller.recover_review_running_to_received()
        assert "no accepted Cline report" in str(info.value)

        assert table_counts(controller) == before
        assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
        assert_invariants(controller, require_attempt_started=True)
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_rejected_for_a_report_of_another_step(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        report, _hash = controller.db.cline_report_for_attempt(1, 1)
        other = dict(report, step_no=2)
        text = json.dumps(other, ensure_ascii=False, sort_keys=True, indent=2)
        controller.db.db.execute(
            "UPDATE cline_reports SET report_json=?, report_hash=? "
            "WHERE step_no=1 AND attempt=1",
            (text, wt.sha256_text(text)),
        )
        before = table_counts(controller)

        assert controller.review_running_recovery_qualified() is False
        with pytest.raises(RuntimeError) as info:
            controller.recover_review_running_to_received()
        message = str(info.value)
        assert "does not belong to the current step" in message
        assert "not valid" in message  # the step_no mismatch is reported too

        assert table_counts(controller) == before
        assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_rejected_for_a_report_of_another_attempt(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        # The accepted report belongs to attempt 1 while the step is on attempt 1:
        # moving the row to attempt 2 must make the recovery fail closed.
        controller.db.db.execute(
            "UPDATE cline_reports SET attempt=2 WHERE step_no=1 AND attempt=1"
        )
        before = table_counts(controller)

        assert controller.db.get_step(1).attempt == 1
        assert controller.review_running_recovery_qualified() is False
        with pytest.raises(RuntimeError, match="no accepted Cline report"):
            controller.recover_review_running_to_received()

        assert table_counts(controller) == before
        assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_rejected_for_an_invalid_accepted_report(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        report, _hash = controller.db.cline_report_for_attempt(1, 1)
        broken = dict(report, status="ALMOST")
        text = json.dumps(broken, ensure_ascii=False, sort_keys=True, indent=2)
        controller.db.db.execute(
            "UPDATE cline_reports SET report_json=?, report_hash=? "
            "WHERE step_no=1 AND attempt=1",
            (text, wt.sha256_text(text)),
        )
        before = table_counts(controller)

        assert controller.review_running_recovery_qualified() is False
        with pytest.raises(RuntimeError) as info:
            controller.recover_review_running_to_received()
        assert "not valid" in str(info.value)

        assert table_counts(controller) == before
        assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_rejected_for_a_hash_mismatch(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        controller.db.db.execute(
            "UPDATE cline_reports SET report_hash=? WHERE step_no=1 AND attempt=1",
            ("0" * 64,),
        )
        before = table_counts(controller)

        assert controller.review_running_recovery_qualified() is False
        with pytest.raises(RuntimeError, match="does not match its stored hash"):
            controller.recover_review_running_to_received()

        assert table_counts(controller) == before
        assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_rejected_when_a_review_result_is_already_persisted(project, guard):
    """A persisted REVIEW result must never be walked back (no duplicate accept)."""
    controller = make_controller(project, env=FAKE_SECRETS)
    to_review_received(controller)
    http = OfflineHttp()
    controller_router(controller, http, auto_policy())
    real_transition = controller.db.transition

    def exploding_transition(step_no, to_state, detail=""):
        if to_state in {"REVIEW_APPROVED", "REVISE", "BLOCKED"}:
            raise KeyboardInterrupt("interrupted after the review was persisted")
        return real_transition(step_no, to_state, detail)

    controller.db.transition = exploding_transition
    with pytest.raises(KeyboardInterrupt):
        controller.run_supervisor_review()
    assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
    assert controller.db.has_codex_review(1, 1) is True  # the result IS persisted

    before = snapshot(controller)
    counts = table_counts(controller)
    assert controller.review_running_recovery_qualified() is False
    with pytest.raises(RuntimeError) as info:
        controller.recover_review_running_to_received()
    assert "already persisted" in str(info.value)

    assert_frozen(before, snapshot(controller))
    assert table_counts(controller) == counts
    assert controller.db.latest_codex_review(1)["verdict"] == "APPROVE"  # untouched
    assert_invariants(controller, require_attempt_started=True)
    http.close()
    controller.close()
    guard.assert_silent()


def test_k1_recovery_is_one_time_and_idempotent_safe(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        controller.recover_review_running_to_received(actor="human", reason="first")
        assert controller.db.has_event(1, RECOVER_EVENT) is True
        assert (
            len(events(controller, "STATE:CODEX_REVIEW_RUNNING->CLINE_REPORT_RECEIVED")) == 1
        )

        # (a) a second call is refused because the state already moved
        assert controller.review_running_recovery_qualified() is False
        with pytest.raises(RuntimeError, match="CODEX_REVIEW_RUNNING"):
            controller.recover_review_running_to_received(actor="human", reason="again")

        # (b) even a forced-back state cannot replay it (one-time audit guard)
        controller.db.db.execute(
            "UPDATE steps SET state='CODEX_REVIEW_RUNNING' WHERE step_no=1"
        )
        assert controller.review_running_recovery_qualified() is False
        with pytest.raises(RuntimeError, match="already used"):
            controller.recover_review_running_to_received(actor="human", reason="again")

        # no duplicate terminal transition was ever written
        assert len(events(controller, RECOVER_EVENT)) == 1
        assert (
            len(events(controller, "STATE:CODEX_REVIEW_RUNNING->CLINE_REPORT_RECEIVED")) == 1
        )
        assert_invariants(controller, require_attempt_started=True)
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_makes_no_provider_plan_or_cline_activity(project, guard):
    """No provider call, no PLAN re-run, no Cline redispatch during recovery."""
    controller, http = crash_into_review_running(project)
    try:
        requests_before = len(http.requests)
        counts_before = table_counts(controller)
        plan_path = controller.db.from_codex / "step_001_plan.json"
        plan_bytes = plan_path.read_bytes() if plan_path.exists() else None
        accepted_before = snapshot(controller).accepted_attempts
        tasks_before = sorted(p.name for p in controller.db.to_cline.glob("*"))
        events_before = len(events(controller))
        report_before = report_bytes(controller)

        controller.recover_review_running_to_received(actor="human", reason="k1")

        assert len(http.requests) == requests_before  # NO provider call
        assert table_counts(controller)["codex_plans"] == counts_before["codex_plans"]
        if plan_bytes is None:
            assert not plan_path.exists()
        else:
            assert plan_path.read_bytes() == plan_bytes
        assert snapshot(controller).accepted_attempts == accepted_before
        assert sorted(p.name for p in controller.db.to_cline.glob("*")) == tasks_before
        assert report_bytes(controller) == report_before  # report untouched
        # never CLINE_DISPATCHED: no Cline redispatch happened
        assert controller.db.get_step(1).state == "CLINE_REPORT_RECEIVED"
        # exactly the two audit rows of the recovery were appended
        assert len(events(controller)) == events_before + 2
        assert_invariants(controller, require_attempt_started=True)
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_audit_content_and_secret_safety(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        stored_hash = controller.db.cline_report_for_attempt(1, 1)[1]

        controller.recover_review_running_to_received(
            actor="operator-bob", reason="restarted the laptop mid-review"
        )

        rows = events(controller, RECOVER_EVENT)
        assert len(rows) == 1
        details = rows[0]["details"]
        assert details["operation"] == RECOVER_EVENT
        assert details["from_state"] == "CODEX_REVIEW_RUNNING"
        assert details["to_state"] == "CLINE_REPORT_RECEIVED"
        assert details["step_no"] == 1
        assert details["attempt"] == 1
        assert details["actor"] == "operator-bob"
        assert "restarted the laptop mid-review" in details["reason"]
        assert details["accepted_report_hash"] == stored_hash
        assert details["timestamp"]
        assert rows[0]["step_no"] == 1  # attached to the step, not global

        # The standard FSM transition row exists alongside the dedicated event.
        assert (
            len(events(controller, "STATE:CODEX_REVIEW_RUNNING->CLINE_REPORT_RECEIVED")) == 1
        )

        # No secret can reach the audit trail or the project files.
        assert leaked_secrets([row["raw"] for row in events(controller)]) == []
        assert scan_tree(project) == []
        assert_invariants(controller, require_attempt_started=True)
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_records_the_default_human_reason(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        controller.recover_review_running_to_received()  # no explicit reason
        details = events(controller, RECOVER_EVENT)[0]["details"]
        assert details["reason"] == RECOVER_REASON
        assert details["actor"] == "human"
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_k1_recovery_then_retryable_outage_behaves_normally(project, guard):
    """After recovery the normal retryable/deferred behaviour is unchanged."""
    controller, http = crash_into_review_running(project)
    http.close()
    try:
        controller.recover_review_running_to_received()

        http_fail = OfflineHttp()
        scripts = success_scripts()
        scripts[ID_DEEPSEEK] = {"review": status_error(429, "rate limited")}
        scripts[ID_OPENAI] = {"review": status_error(503, "overloaded")}
        scripts[ID_ANTHROPIC] = {"review": status_error(429, "rate limited")}
        scripts[PROVIDER_CODEX_LEGACY] = {"error": codex_run_error("timeout")}
        controller_router(controller, http_fail, auto_policy(), scripts=scripts)

        deferred = controller.run_supervisor_review()
        assert deferred["deferred"] is True
        assert controller.db.get_step(1).state == wt.CODEX_REVIEW_RETRYABLE
        assert controller.db.get_step(1).attempt == 1

        http_ok = OfflineHttp()
        controller_router(controller, http_ok, auto_policy())
        review = controller.retry_codex_review()
        assert review["verdict"] == "APPROVE"
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"
        assert controller.db.get_step(1).attempt == 1  # never consumed
        assert_invariants(controller, require_attempt_started=True)
        http_fail.close()
        http_ok.close()
    finally:
        controller.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# GUI: the human-triggered button (skipped when Tk is unavailable)
# --------------------------------------------------------------------------- #


def test_k1_gui_button_is_enabled_only_in_review_running(project, guard):
    controller, http = crash_into_review_running(project)
    try:
        app, gui_http = make_gui(controller, offline_transport_http=http)
    except Exception as exc:  # pragma: no cover - headless CI
        http.close()
        controller.close()
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        assert str(app.btn_recover_review["text"]) == "Recover Interrupted REVIEW"
        assert (
            str(app.btn_recover_review["command"]).endswith("recover_interrupted_review")
        )

        # Enabled in CODEX_REVIEW_RUNNING, with a clear recoverable status.
        app.refresh()
        assert str(app.btn_recover_review["state"]) == "normal"
        assert "interrupted REVIEW" in app.recovery_var.get()
        assert "STEP 1" in app.recovery_var.get()

        # Disabled (and status cleared) in every other state.
        for state in (
            "CLINE_REPORT_RECEIVED",
            "CODEX_REVIEW_RETRYABLE",
            "REVIEW_APPROVED",
            "FAILED",
            "CLINE_DISPATCHED",
        ):
            controller.db.db.execute(
                "UPDATE steps SET state = ? WHERE step_no = 1", (state,)
            )
            app.refresh()
            assert str(app.btn_recover_review["state"]) == "disabled", state
            assert app.recovery_var.get() == "Recovery: —", state
    finally:
        app.on_close()
        gui_http.close()
        controller.close()
    guard.assert_silent()


def test_k1_gui_recovery_requires_confirmation_and_never_runs_review(
    project, guard, monkeypatch
):
    controller, http = crash_into_review_running(project)
    try:
        app, gui_http = make_gui(controller, offline_transport_http=http)
    except Exception as exc:  # pragma: no cover - headless CI
        http.close()
        controller.close()
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        requests_before = len(http.requests)

        # (a) declining the explicit confirmation changes NOTHING
        monkeypatch.setattr(wt.messagebox, "askyesno", lambda *a, **k: False)
        app.recover_interrupted_review()
        assert controller.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
        assert controller.db.has_event(1, RECOVER_EVENT) is False
        assert len(http.requests) == requests_before

        # (b) confirming performs the recovery - and NEVER starts REVIEW
        monkeypatch.setattr(wt.messagebox, "askyesno", lambda *a, **k: True)
        app.recover_interrupted_review()
        assert controller.db.get_step(1).state == "CLINE_REPORT_RECEIVED"
        assert "recovered to CLINE_REPORT_RECEIVED" in app.footer_var.get()
        assert len(http.requests) == requests_before  # no provider request at all
        assert controller.db.has_codex_review(1, 1) is False
        assert controller.db.get_step(1).attempt == 1
        # (c) the human now has the normal REVIEW button back
        assert str(app.btn_recover_review["state"]) == "disabled"
        assert str(app.btn_review["state"]) == "normal"
        assert_invariants(controller, require_attempt_started=True)
    finally:
        app.on_close()
        gui_http.close()
        controller.close()
    guard.assert_silent()


def test_k1_gui_reopen_shows_the_recoverable_status_without_actions(project, guard):
    controller, http = crash_into_review_running(project)
    controller.close()
    http.close()

    reopened = offline_controller(project)
    http2 = OfflineHttp()
    try:
        app, gui_http = make_gui(reopened, offline_transport_http=http2)
    except Exception as exc:  # pragma: no cover - headless CI
        http2.close()
        reopened.close()
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        requests_before = len(http2.requests)

        app.refresh()

        assert "interrupted REVIEW" in app.recovery_var.get()
        assert str(app.btn_recover_review["state"]) == "normal"
        assert str(app.btn_review["state"]) == "disabled"  # REVIEW is not running yet
        # Nothing was executed automatically by the reopen.
        assert len(http2.requests) == requests_before
        assert reopened.db.has_codex_review(1, 1) is False
        assert reopened.db.get_step(1).state == "CODEX_REVIEW_RUNNING"
        assert reopened.db.get_step(1).attempt == 1
    finally:
        app.on_close()
        gui_http.close()
        reopened.close()
    guard.assert_silent()


# --------------------------------------------------------------------------- #
# Source-level guards
# --------------------------------------------------------------------------- #


def test_k1_recovery_edge_is_only_reachable_through_the_controller():
    """No provider/router code may perform the recovery (source-level guard)."""
    app_source = pathlib.Path(wt.__file__).read_text(encoding="utf-8")
    package = pathlib.Path(wt.__file__).resolve().parent / "supervisor"
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "recover_review_running_to_received" not in text, path.name
        assert "RECOVER_REVIEW_RUNNING" not in text, path.name
        assert "recover_interrupted_review" not in text, path.name

    # The DB write of the recovery state exists exactly ONCE per dedicated
    # recovery method (FAILED and interrupted-REVIEW): no other code path may
    # move a running REVIEW backwards.
    ddl = "SET state='CLINE_REPORT_RECEIVED'"
    assert app_source.count(ddl) == 2
    for marker in (
        "def recover_review_running_to_received(",
        "def recover_failed_review_to_received(",
    ):
        body = app_source[app_source.index(marker):]
        body = body[: body.index("\n    def ", 1)]
        assert ddl in body, marker

    # The interrupted-REVIEW recovery is guarded by its own state precondition and
    # by the terminal-result guard (a persisted review must never be walked back).
    running = app_source[app_source.index("def recover_review_running_to_received("):]
    running = running[: running.index("\n    def ", 1)]
    assert "CODEX_REVIEW_RUNNING" in running
    assert "RECOVER_REVIEW_RUNNING_EVENT" in running
    assert "has_codex_review" in running

    # The GUI handler never writes the database directly.
    handler = app_source[app_source.index("def recover_interrupted_review("):]
    handler = handler[: handler.index("\n    def ", 1)]
    assert "self.controller.recover_review_running_to_received(" in handler
    assert "db.execute" not in handler
    assert "db.transition" not in handler


# --------------------------------------------------------------------------- #
# Regression: the audit reason goes through the STRONG supervisor sanitizer
# --------------------------------------------------------------------------- #


def test_k1_recovery_audit_redacts_a_key_shaped_secret_in_the_reason(project, guard):
    """REGRESSION: a key-shaped token in the human reason must never be persisted.

    The module used to shadow the strong ``supervisor.errors.sanitize_diagnostic``
    with a weaker module-local copy, so a human-supplied reason reached the audit
    trail (and the DB) verbatim. The recovery audit must stay sanitized.
    """
    controller, http = crash_into_review_running(project)
    secret = "sk-" + "z" * 30
    try:
        controller.recover_review_running_to_received(
            actor="human", reason=f"restarted the laptop; key={secret}"
        )

        raw = "\n".join(row["raw"] for row in events(controller))
        assert secret not in raw
        assert "restarted the laptop" in raw  # the useful part is preserved
        assert leaked_secrets(raw) == []
        assert secret not in database_dump(controller)
        assert secret not in json.dumps(
            controller.supervisor_router.route_history, default=str
        )
        assert scan_tree(project) == []
    finally:
        controller.close()
        http.close()
    guard.assert_silent()


def test_module_sanitizer_is_the_strong_supervisor_implementation():
    """REGRESSION: no weaker local copy may shadow the supervisor sanitizer."""
    from supervisor import sanitize_diagnostic as strong

    key = "sk-" + "q" * 30
    text = f"key={key} Authorization: Bearer abcDEF123456 x-api-key: {key}"

    assert wt.sanitize_diagnostic(text) == strong(text)
    assert key not in wt.sanitize_diagnostic(text)
    assert "abcDEF123456" not in wt.sanitize_diagnostic(text)
    assert wt.sanitize_diagnostic("plain message") == "plain message"
