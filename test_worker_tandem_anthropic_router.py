"""M7.4 - Anthropic ROUTER INTEGRATION (registration, routing, safety, parity).

``anthropic``/Claude stops being an "adapter without a home" and becomes a
registered provider: MANUAL selection, the AUTO route the M8 policy already
declares, and fallback targets/sources all work through it.

What this file pins:

* REGISTRATION - the app builds and registers Claude exactly like the other API
  providers, and registering it changes **no** default (no default provider
  switch, no FSM state string, no SQLite schema change).
* MANUAL - a MANUAL policy that selects Claude really does PLAN and REVIEW
  through ``POST /messages``.
* FALLBACK - OpenAI -> Anthropic and DeepSeek -> Anthropic, Anthropic -> the next
  provider when the policy allows it, and **no** fallback on an auth failure.
* SAFETY - a Claude failure never consumes a project/Cline attempt, never reruns
  Cline/PLAN, never deletes the accepted report and never alters the step state.
* PARITY - Anthropic/OpenAI/DeepSeek/Codex-Legacy normalize to the SAME
  controller-relevant PLAN and REVIEW (provider metadata is ignored).

Everything is OFFLINE: the HTTP providers are driven by ``httpx.MockTransport``
(no socket, no key, no paid call), the Codex Legacy slot is a stub runner and a
:class:`LiveNetworkGuard` fails the test if anything reaches the real world.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import socket
import sys

import httpx
import pytest

from supervisor import (
    ANTHROPIC_API_KEY_ENV,
    BUDGET_STANDARD,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_LEGACY_PROVIDER,
    DEEPSEEK_API_KEY_ENV,
    ID_ANTHROPIC,
    ID_DEEPSEEK,
    ID_OPENAI,
    OPENAI_API_KEY_ENV,
    PLAN_OPERATION,
    PROVIDER_ANTHROPIC,
    PROVIDER_CODEX_LEGACY,
    REVIEW_OPERATION,
    AnthropicSupervisorAdapter,
    DeepSeekSupervisorAdapter,
    HttpTransport,
    LegacyCodexSupervisorAdapter,
    OpenAISupervisorAdapter,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorReviewRequest,
    SupervisorRouter,
    SupervisorRunError,
    default_anthropic_config,
    default_auto_policy,
    default_deepseek_config,
    default_openai_config,
    fake_plan_payload,
    fake_review_payload,
    supervisor_plan_schema,
    supervisor_review_schema,
)
from supervisor.errors import (
    CATEGORY_AUTH,
    CATEGORY_CONTRACT,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_RATE_LIMIT,
)

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"

#: Obviously synthetic credentials: used in headers, never persisted or logged.
TEST_KEY_ANTHROPIC = "test-anthropic-key-not-a-real-credential"
TEST_KEY_DEEPSEEK = "test-deepseek-key-not-a-real-credential"
TEST_KEY_OPENAI = "test-openai-key-not-a-real-credential"
SECRET_VALUES = (TEST_KEY_ANTHROPIC, TEST_KEY_DEEPSEEK, TEST_KEY_OPENAI)

#: The injected environment (names only ever reach a widget or a file).
KEYS = {
    ANTHROPIC_API_KEY_ENV: TEST_KEY_ANTHROPIC,
    DEEPSEEK_API_KEY_ENV: TEST_KEY_DEEPSEEK,
    OPENAI_API_KEY_ENV: TEST_KEY_OPENAI,
}

ANTHROPIC_HOST = "api.anthropic.com"
OPENAI_HOST = "api.openai.com"
DEEPSEEK_HOST = "api.deepseek.com"

PLAN_DEF = {
    "plan_version": "m74-1",
    "steps": [
        {
            "step_no": 1,
            "phase": "FOUNDATION",
            "title": "M7.4 STEP",
            "description": "router integration step",
            "risk": "LOW",
        },
    ],
}

PLAN_OK = fake_plan_payload(goal="m7.4 plan")
REVIEW_APPROVE = fake_review_payload(verdict="APPROVE", summary="m7.4 review")
REVIEW_REVISE = fake_review_payload(verdict="REVISE", summary="m7.4 revise")
def _load_worker_tandem(name: str):
    spec = importlib.util.spec_from_file_location(name, SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem("worker_tandem_m74")




# --------------------------------------------------------------------------- #
# Offline doubles (MockTransport never opens a socket)
# --------------------------------------------------------------------------- #


def _models_ok() -> httpx.Response:
    return httpx.Response(
        200, json={"object": "list", "data": [{"id": "offline-model"}]}
    )


def chat_ok(payload: dict):
    """OpenAI/DeepSeek chat-completions envelope carrying the JSON payload."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-m74",
                "object": "chat.completion",
                "model": "offline-chat-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(payload)},
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


def chat_error(status: int, message: str, code: str = ""):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message, "code": code}})

    return handler


def messages_ok(payload: dict):
    """Anthropic Messages envelope: ``content`` is a LIST of typed blocks."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_m74",
                "type": "message",
                "role": "assistant",
                "model": DEFAULT_ANTHROPIC_MODEL,
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    return handler


def messages_error(status: int, error_type: str, message: str = "offline error"):
    """Anthropic error envelope (``authentication_error`` etc.)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"type": "error", "error": {"type": error_type, "message": message}},
        )

    return handler


def _load_worker_tandem(name: str):
    spec = importlib.util.spec_from_file_location(name, SRC)
    module = importlib.util.module_from_spec(spec)


class RouterHttp:
    """Recording MockTransport shared by every HTTP provider (never a socket)."""

    def __init__(self, *, anthropic=None, openai=None, deepseek=None):
        self.calls: list[dict] = []
        self.anthropic = anthropic or messages_ok(PLAN_OK)
        self.openai = openai or chat_ok(PLAN_OK)
        self.deepseek = deepseek or chat_ok(PLAN_OK)
        self._client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        payload: dict = {}
        if request.method == "POST":
            payload = json.loads(request.content.decode("utf-8") or "{}")
        self.calls.append(
            {
                "method": request.method,
                "host": request.url.host,
                "url": str(request.url),
                "headers": {str(k).lower(): v for k, v in request.headers.items()},
                "json": payload,
            }
        )
        if request.method == "GET":
            return _models_ok()
        if request.url.host == ANTHROPIC_HOST:
            return self.anthropic(request)
        if request.url.host == OPENAI_HOST:
            return self.openai(request)
        return self.deepseek(request)

    @property
    def transport(self) -> HttpTransport:
        return HttpTransport(client=self._client)

    def posted(self, host: str | None = None) -> list[dict]:
        return [
            call
            for call in self.calls
            if call["method"] == "POST" and (host is None or call["host"] == host)
        ]

    def posted_hosts(self) -> list[str]:
        return [call["host"] for call in self.posted()]

    def close(self) -> None:
        self._client.close()


def anthropic_adapter(http: RouterHttp) -> AnthropicSupervisorAdapter:
    return AnthropicSupervisorAdapter(
        default_anthropic_config(), http.transport, env=dict(KEYS)
    )


def deepseek_adapter(http: RouterHttp) -> DeepSeekSupervisorAdapter:
    return DeepSeekSupervisorAdapter(
        default_deepseek_config(), http.transport, env=dict(KEYS)
    )


def openai_adapter(http: RouterHttp) -> OpenAISupervisorAdapter:
    return OpenAISupervisorAdapter(
        default_openai_config(), http.transport, env=dict(KEYS)
    )


class _StubCodexRunner:
    """Minimal ``StructuredRunnerPort`` double for the Codex Legacy parity check."""

    model = "codex-model"

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def run_structured(
        self,
        prompt,
        schema_path,
        output_path,
        trace_path,
        timeout_sec=None,
        reasoning_effort=None,
    ):
        return dict(self._payload)

    def available(self):
        return True, "codex-cli test"


def codex_adapter(payload: dict, workdir) -> LegacyCodexSupervisorAdapter:
    return LegacyCodexSupervisorAdapter(_StubCodexRunner(payload), workdir=workdir)


def plan_request(risk: str = "LOW", prompt: str = "PLAN prompt") -> SupervisorPlanRequest:
    return SupervisorPlanRequest(
        step_no=1,
        attempt=1,
        risk=risk,
        requires_human=False,
        prompt=prompt,
        context={},
        schema=supervisor_plan_schema(),
    )


def review_request(
    risk: str = "LOW", prompt: str = "REVIEW prompt"
) -> SupervisorReviewRequest:
    return SupervisorReviewRequest(
        step_no=1,
        attempt=1,
        risk=risk,
        requires_human=False,
        prompt=prompt,
        context={},
        schema=supervisor_review_schema(),
    )


def policy_for(
    *, plan: tuple[str, ...], review: tuple[str, ...], chain: tuple[str, ...] = ()
) -> SupervisorPolicy:
    """An AUTO policy whose every risk level uses the same candidate lists.

    ``max_provider_retries=0`` keeps the expectations exact: every candidate is
    attempted exactly once, so a repeated provider visit would be a loop.
    """
    routes = {
        risk: {PLAN_OPERATION: tuple(plan), REVIEW_OPERATION: tuple(review)}
        for risk in ("LOW", "MEDIUM", "HIGH")
    }
    return SupervisorPolicy(
        mode="AUTO",
        risk_routes=routes,
        fallback_chain=tuple(chain),
        max_provider_retries=0,
        allow_codex_legacy=False,
        budget_mode=BUDGET_STANDARD,
    )


def router_for(http: RouterHttp, policy: SupervisorPolicy) -> SupervisorRouter:
    """A router with all four providers wired to the offline transport."""
    return SupervisorRouter(
        {
            ID_ANTHROPIC: anthropic_adapter(http),
            ID_DEEPSEEK: deepseek_adapter(http),
            ID_OPENAI: openai_adapter(http),
            PROVIDER_CODEX_LEGACY: codex_adapter(PLAN_OK, pathlib.Path(".")),
        },
        policy,
    )


class LiveNetworkGuard:
    """Fails the test if anything tries to reach the real world."""

    def __init__(self, monkeypatch):
        self.counts = {"socket.create_connection": 0, "httpx_transport": 0}
        for name, owner, attribute in (
            ("socket.create_connection", socket, "create_connection"),
            ("httpx_transport", httpx.HTTPTransport, "handle_request"),
        ):
            monkeypatch.setattr(owner, attribute, self._block(name))

    def _block(self, name):
        def boom(*args, **kwargs):
            self.counts[name] += 1
            raise AssertionError(f"LIVE CALL ATTEMPTED: {name}")

        return boom

    def assert_silent(self):
        assert self.counts == {"socket.create_connection": 0, "httpx_transport": 0}



# --------------------------------------------------------------------------- #
# App-level helpers (real controller, offline transport, stub Codex)
# --------------------------------------------------------------------------- #


def sequence(*handlers):
    """Answer each request with the next handler (the last one repeats)."""

    pending = list(handlers)

    def handler(request: httpx.Request) -> httpx.Response:
        chosen = pending[0] if len(pending) == 1 else pending.pop(0)
        return chosen(request)

    return handler


def controller_for(project, http: RouterHttp, *, policy=None):
    """A real controller in a temp project with an injected transport/env."""
    controller = wt.TandemController(
        project, supervisor_transport=http.transport, supervisor_env=dict(KEYS)
    )
    controller.db.import_plan(PLAN_DEF)
    if policy is not None:
        controller.save_supervisor_policy(policy)
    return controller


def stub_codex(controller, payload: dict = PLAN_OK) -> None:
    """Keep the Codex CLI from ever being resolved or executed."""

    def fake(
        prompt,
        schema_path,
        output_path,
        trace_path,
        timeout_sec=None,
        reasoning_effort=None,
    ):
        return dict(payload)

    controller.codex.available = lambda: (True, "codex-stub (never executed)")
    controller.codex.run_structured = fake


def report_ok(step_no: int = 1, status: str = "DONE") -> dict:
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


def prepare_accepted_report(controller, *, reason: str = "m7.4 test"):
    """A Codex-produced PLAN plus an accepted Cline report for attempt 1."""
    stub_codex(controller)
    controller.run_codex_plan()
    controller.approve_plan_and_dispatch_cline(reason=reason)
    wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
    controller.process_cline_report()
    return controller.db.get_step(1)


def event_payloads(controller, event_type: str) -> list[dict]:
    rows = controller.db.db.execute(
        "SELECT details FROM events WHERE event_type=? ORDER BY id", (event_type,)
    ).fetchall()
    return [json.loads(row["details"]) for row in rows]


def plan_rows(controller) -> list[dict]:
    return [
        dict(row)
        for row in controller.db.db.execute(
            "SELECT step_no, attempt, plan_json FROM codex_plans ORDER BY id"
        ).fetchall()
    ]


def dispatch_count(controller) -> int:
    """How often Cline was dispatched (a rerun would add one)."""
    rows = controller.db.db.execute(
        "SELECT id FROM events WHERE event_type LIKE '%->CLINE_DISPATCHED'"
    ).fetchall()
    return len(rows)


def _schema_of(controller) -> dict:
    """SQLite schema fingerprint: table -> (column, type) pairs (no data)."""
    tables = [
        row["name"]
        for row in controller.db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    return {
        table: tuple(
            (column["name"], column["type"])
            for column in controller.db.db.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        )
        for table in tables
    }


def safety_snapshot(controller) -> dict:
    """Everything a provider failure must never touch."""
    step = controller.db.get_step(1)
    report = controller.db.cline_report_for_attempt(1, step.attempt)
    return {
        "attempt": step.attempt,
        "state": step.state,
        "plans": plan_rows(controller),
        "report": report,
        "dispatches": dispatch_count(controller),
    }


def metadata_free(result):
    """A provider result minus the fields the controller must ignore.

    ``provider``/``model``/``usage`` are per-provider metadata by definition, and
    ``route_chain`` is routing provenance (it names the providers tried) - none of
    them may influence the controller-relevant outcome.
    """
    data = dataclasses.asdict(result)
    for key in ("provider", "model", "usage", "route_chain"):
        data.pop(key, None)
    return data


# --------------------------------------------------------------------------- #
# 1  REGISTRATION (anthropic is registered; NO default changes)
# --------------------------------------------------------------------------- #


def test_anthropic_is_a_buildable_provider():
    http = RouterHttp()
    try:
        assert PROVIDER_ANTHROPIC == "anthropic" == ID_ANTHROPIC
        assert PROVIDER_ANTHROPIC in wt.SUPERVISOR_BUILDABLE_PROVIDERS
        assert (
            wt.TandemController.supervisor_provider_available(PROVIDER_ANTHROPIC)
            is True
        )

        adapter = wt.build_supervisor_provider(
            PROVIDER_ANTHROPIC, transport=http.transport, env=dict(KEYS), models={}
        )
        assert isinstance(adapter, AnthropicSupervisorAdapter)
        assert adapter.provider_id == PROVIDER_ANTHROPIC
        assert adapter.model == DEFAULT_ANTHROPIC_MODEL
        assert adapter.api_key_source == f"env:{ANTHROPIC_API_KEY_ENV}"

        # A configured model wins; an unknown id stays unsupported.
        custom = wt.build_supervisor_provider(
            PROVIDER_ANTHROPIC,
            transport=http.transport,
            env=dict(KEYS),
            models={ID_ANTHROPIC: "claude-configured-model"},
        )
        assert custom.model == "claude-configured-model"
        assert wt.build_supervisor_provider("nope", transport=http.transport) is None
        # Building an adapter is I/O-free: no health probe, no completion.
        assert http.calls == []
    finally:
        http.close()


def test_manual_selection_registers_anthropic(tmp_path):
    http = RouterHttp()
    controller = controller_for(
        tmp_path,
        http,
        policy=SupervisorPolicy(
            mode="MANUAL", plan_provider=ID_ANTHROPIC, review_provider=ID_ANTHROPIC
        ),
    )
    try:
        assert controller.supervisor_policy.plan_provider == ID_ANTHROPIC
        assert controller.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
            ID_ANTHROPIC,
        )
        # The operator-facing lists are unchanged; registering is not a call.
        assert wt.SUPERVISOR_MANUAL_CHOICES == (
            ID_DEEPSEEK,
            ID_OPENAI,
            ID_ANTHROPIC,
            PROVIDER_CODEX_LEGACY,
        )
        assert http.calls == []
    finally:
        controller.db.close()
        http.close()


def test_auto_policy_registers_anthropic_and_keeps_the_default(tmp_path):
    http = RouterHttp()
    controller = controller_for(tmp_path, http, policy=default_auto_policy())
    try:
        assert controller.supervisor_router.registered_providers() == (
            PROVIDER_CODEX_LEGACY,
            ID_DEEPSEEK,
            ID_OPENAI,
            ID_ANTHROPIC,
        )
        # The M8 §2 AUTO policy really declares Claude for HIGH/REVIEW.
        assert (
            controller.supervisor_policy.candidates_for(REVIEW_OPERATION, "HIGH")[0]
            == ID_ANTHROPIC
        )
        assert http.calls == []
    finally:
        controller.db.close()
        http.close()

    # Registering Anthropic changes NO default (M4 policy is still the default).
    assert SupervisorPolicy().mode == "MANUAL"
    assert SupervisorPolicy().plan_provider == DEFAULT_LEGACY_PROVIDER
    assert SupervisorPolicy().review_provider == DEFAULT_LEGACY_PROVIDER
    assert SupervisorPolicy().fallback_chain == ()
    assert wt.default_supervisor_policy().plan_provider == DEFAULT_LEGACY_PROVIDER


def test_registration_changes_no_fsm_state_string_and_no_sqlite_schema(tmp_path):
    # Frozen M4/M9 FSM: registration adds no state and no transition.
    assert wt.VALID_STATES == {
        "PENDING",
        "READY_FOR_CODEX_PLAN",
        "CODEX_PLAN_RUNNING",
        "PLAN_READY",
        "WAITING_HUMAN_APPROVAL",
        "CLINE_DISPATCHED",
        "CLINE_REPORT_RECEIVED",
        "CODEX_REVIEW_RUNNING",
        wt.CODEX_REVIEW_RETRYABLE,
        "REVIEW_APPROVED",
        "REVISE",
        "BLOCKED",
        "FAILED",
        "VERIFIED",
        "SKIPPED",
        "ABORTED",
    }
    assert wt.ALLOWED_TRANSITIONS["CLINE_REPORT_RECEIVED"] == {
        "CODEX_REVIEW_RUNNING",
        "BLOCKED",
        "FAILED",
    }

    http = RouterHttp()
    first = tmp_path / "default"
    second = tmp_path / "claude"
    first.mkdir()
    second.mkdir()
    default_controller = wt.TandemController(
        first, supervisor_transport=http.transport, supervisor_env=dict(KEYS)
    )
    claude_controller = wt.TandemController(
        second, supervisor_transport=http.transport, supervisor_env=dict(KEYS)
    )
    try:
        claude_controller.save_supervisor_policy(
            SupervisorPolicy(
                mode="MANUAL", plan_provider=ID_ANTHROPIC, review_provider=ID_ANTHROPIC
            )
        )
        # Selecting Claude persists a provider id - never a schema change.
        assert _schema_of(claude_controller) == _schema_of(default_controller)
    finally:
        default_controller.db.close()
        claude_controller.db.close()
        http.close()



# --------------------------------------------------------------------------- #
# 2  MANUAL selection (Claude really performs PLAN and REVIEW)
# --------------------------------------------------------------------------- #


def test_manual_anthropic_plan_and_review_through_the_controller(tmp_path, monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(
        anthropic=sequence(messages_ok(PLAN_OK), messages_ok(REVIEW_APPROVE))
    )
    controller = controller_for(
        tmp_path,
        http,
        policy=SupervisorPolicy(
            mode="MANUAL", plan_provider=ID_ANTHROPIC, review_provider=ID_ANTHROPIC
        ),
    )
    try:
        # Legacy Codex must never run while Claude is the selected provider.
        codex_calls: list[str] = []
        stub_codex(controller)
        original_runner = controller.codex.run_structured

        def guarded(*args, **kwargs):
            codex_calls.append("codex")
            return original_runner(*args, **kwargs)

        controller.codex.run_structured = guarded

        # 1) PLAN through Claude.
        plan = controller.run_codex_plan()
        assert plan["decision"] == "PLAN_READY"
        assert controller.supervisor_router.calls[-1]["provider"] == ID_ANTHROPIC
        assert codex_calls == []

        # 2) The human approves; Cline runs; the report is accepted.
        controller.approve_plan_and_dispatch_cline(reason="m7.4")
        wt.write_json_atomic(controller.expected_cline_report_path(1), report_ok())
        controller.process_cline_report()
        assert controller.db.get_step(1).state == "CLINE_REPORT_RECEIVED"
        before = safety_snapshot(controller)

        # 3) REVIEW through Claude.
        review = controller.run_supervisor_review()
        assert review["verdict"] == "APPROVE"
        # The controller owns the identity - never the provider payload.
        assert review["step_no"] == 1
        assert controller.db.get_step(1).state == "REVIEW_APPROVED"
        assert [e["provider"] for e in event_payloads(controller, "SUPERVISOR_SELECTED")] == [
            ID_ANTHROPIC,
            ID_ANTHROPIC,
        ]

        # 4) Only the Messages endpoint was used, with x-api-key and no bearer.
        posts = http.posted(ANTHROPIC_HOST)
        assert len(posts) == 2
        assert http.posted_hosts() == [ANTHROPIC_HOST, ANTHROPIC_HOST]
        for call in posts:
            assert call["url"].endswith("/messages")
            assert call["headers"]["x-api-key"] == TEST_KEY_ANTHROPIC
            assert "authorization" not in call["headers"]
            body = call["json"]
            assert set(body) <= {
                "model",
                "max_tokens",
                "system",
                "messages",
                "temperature",
            }
            assert body["model"] == DEFAULT_ANTHROPIC_MODEL
            assert isinstance(body["system"], str)  # top-level system turn
            assert body["messages"][0]["role"] == "user"
            assert set(body["messages"][0]) == {"role", "content"}

        # 5) No attempt consumed, no PLAN rerun, no second dispatch, same report.
        after = safety_snapshot(controller)
        assert before["attempt"] == after["attempt"] == 1
        assert before["plans"] == after["plans"]
        assert before["dispatches"] == after["dispatches"] == 1
        assert before["report"] == after["report"]
        assert codex_calls == []
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()
# --------------------------------------------------------------------------- #
# 3  FALLBACK + loop guards (router level, real Claude adapter)
# --------------------------------------------------------------------------- #


def test_fallback_openai_to_anthropic(monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(
        openai=chat_error(429, "rate limited", "rate_limit"),
        anthropic=messages_ok(PLAN_OK),
    )
    router = router_for(
        http,
        policy_for(plan=(ID_OPENAI,), review=(ID_OPENAI,), chain=(ID_ANTHROPIC,)),
    )
    try:
        result = router.plan(plan_request())

        assert result.provider == ID_ANTHROPIC
        assert result.model == DEFAULT_ANTHROPIC_MODEL
        assert result.fallback_from == ID_OPENAI
        assert result.fallback_reason == CATEGORY_RATE_LIMIT
        assert result.payload == PLAN_OK
        assert result.route_chain == (ID_OPENAI, ID_ANTHROPIC)
        # Exactly one attempt per provider: no same-provider retry, no loop.
        assert len(http.posted(OPENAI_HOST)) == 1
        assert len(http.posted(ANTHROPIC_HOST)) == 1
        assert len(router.calls) == 2
    finally:
        http.close()
    guard.assert_silent()


def test_fallback_deepseek_to_anthropic(monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(
        deepseek=chat_error(429, "rate limited", "rate_limit"),
        anthropic=messages_ok(REVIEW_APPROVE),
    )
    router = router_for(
        http,
        policy_for(plan=(ID_DEEPSEEK,), review=(ID_DEEPSEEK,), chain=(ID_ANTHROPIC,)),
    )
    try:
        result = router.review(review_request())

        assert result.provider == ID_ANTHROPIC
        assert result.fallback_from == ID_DEEPSEEK
        assert result.fallback_reason == CATEGORY_RATE_LIMIT
        assert result.payload == REVIEW_APPROVE
        assert result.route_chain == (ID_DEEPSEEK, ID_ANTHROPIC)
        assert len(http.posted(DEEPSEEK_HOST)) == 1
        assert len(http.posted(ANTHROPIC_HOST)) == 1
        assert len(router.calls) == 2
    finally:
        http.close()
    guard.assert_silent()


def test_anthropic_failure_falls_forward_when_the_policy_allows_it(monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(
        anthropic=messages_error(500, "api_error", "claude exploded"),
        deepseek=chat_ok(PLAN_OK),
    )
    router = router_for(
        http,
        policy_for(plan=(ID_ANTHROPIC,), review=(ID_ANTHROPIC,), chain=(ID_DEEPSEEK,)),
    )
    try:
        result = router.plan(plan_request())

        assert result.provider == ID_DEEPSEEK
        assert result.fallback_from == ID_ANTHROPIC
        # Anthropic's ``api_error`` type maps to provider_unavailable.
        assert result.fallback_reason == CATEGORY_PROVIDER_UNAVAILABLE
        assert result.payload == PLAN_OK
        assert result.route_chain == (ID_ANTHROPIC, ID_DEEPSEEK)
        assert len(http.posted(ANTHROPIC_HOST)) == 1
        assert len(http.posted(DEEPSEEK_HOST)) == 1
        assert len(router.calls) == 2
    finally:
        http.close()
    guard.assert_silent()


def test_no_fallback_on_anthropic_auth_failure(monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(
        anthropic=messages_error(401, "authentication_error", "invalid x-api-key"),
        deepseek=chat_ok(PLAN_OK),
    )
    router = router_for(
        http,
        policy_for(plan=(ID_ANTHROPIC,), review=(ID_ANTHROPIC,), chain=(ID_DEEPSEEK,)),
    )
    try:
        with pytest.raises(SupervisorRunError) as info:
            router.plan(plan_request())

        assert info.value.category == CATEGORY_AUTH
        assert info.value.retryable is False
        assert info.value.provider == ID_ANTHROPIC
        # A permanent credential failure never walks the chain ...
        assert http.posted(DEEPSEEK_HOST) == []
        assert len(router.calls) == 1
        # ... and is not retried on the same provider either.
        assert len(http.posted(ANTHROPIC_HOST)) == 1
    finally:
        http.close()
    guard.assert_silent()
def test_no_provider_loop_when_a_provider_repeats_in_the_chain(monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(
        anthropic=messages_error(500, "api_error"),
        deepseek=chat_error(429, "rate limited"),
        openai=chat_error(429, "rate limited"),
    )
    recorded: list[tuple[str, dict]] = []
    router = SupervisorRouter(
        {
            ID_ANTHROPIC: anthropic_adapter(http),
            ID_DEEPSEEK: deepseek_adapter(http),
            ID_OPENAI: openai_adapter(http),
            PROVIDER_CODEX_LEGACY: codex_adapter(PLAN_OK, pathlib.Path(".")),
        },
        policy_for(
            plan=(ID_ANTHROPIC,),
            review=(ID_ANTHROPIC,),
            chain=(ID_ANTHROPIC, ID_OPENAI, ID_DEEPSEEK, ID_OPENAI),
        ),
        lambda kind, payload: recorded.append((kind, payload)),
    )
    try:
        with pytest.raises(SupervisorRunError) as info:
            router.plan(plan_request())

        assert info.value.retryable is True
        chain = [
            payload["route_chain"]
            for kind, payload in recorded
            if kind == "SUPERVISOR_ALL_FAILED"
        ][-1]
        assert chain == [ID_ANTHROPIC, ID_DEEPSEEK, ID_OPENAI]  # deduplicated
        assert len(chain) == len(set(chain))  # no provider visited twice
        hops = [
            payload["fallback_from"]
            for kind, payload in recorded
            if kind == "SUPERVISOR_FALLBACK"
        ]
        assert hops == [ID_ANTHROPIC, ID_DEEPSEEK]

        # Each provider was called exactly once: bounded, no loop.
        assert len(http.posted(ANTHROPIC_HOST)) == 1
        assert len(http.posted(DEEPSEEK_HOST)) == 1
        assert len(http.posted(OPENAI_HOST)) == 1
        assert len(router.calls) == 3
    finally:
        http.close()
    guard.assert_silent()


def test_no_secret_in_router_metadata(monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    secret = "sk-ant-" + "A" * 24
    http = RouterHttp(
        anthropic=messages_error(
            401, "authentication_error", f"your key {secret} is invalid"
        ),
        deepseek=chat_error(429, "rate limited"),
    )
    adapter = AnthropicSupervisorAdapter(
        default_anthropic_config(), http.transport, env={ANTHROPIC_API_KEY_ENV: secret}
    )
    recorded: list[tuple[str, dict]] = []
    router = SupervisorRouter(
        {ID_ANTHROPIC: adapter, ID_DEEPSEEK: deepseek_adapter(http)},
        policy_for(plan=(ID_ANTHROPIC,), review=(ID_ANTHROPIC,), chain=(ID_DEEPSEEK,)),
        lambda kind, payload: recorded.append((kind, payload)),
    )
    try:
        with pytest.raises(SupervisorRunError) as info:
            router.plan(plan_request())

        # The key really was used - and the provider even echoed it back.
        assert http.posted(ANTHROPIC_HOST)[0]["headers"]["x-api-key"] == secret

        metadata = json.dumps(
            [dict(entry) for entry in router.route_history], default=str
        )
        metadata += json.dumps(recorded, default=str)
        metadata += f"{info.value} {info.value.sanitized_message}"
        assert secret not in metadata
        assert secret[:9] not in metadata
        assert "x-api-key" not in metadata.lower()
        assert "authorization" not in metadata.lower()
        # Metadata stays useful: provider + category are recorded.
        assert ID_ANTHROPIC in metadata
        assert CATEGORY_AUTH in metadata
    finally:
        http.close()
    guard.assert_silent()
# --------------------------------------------------------------------------- #
# 4  ATTEMPT + accepted-report safety (controller level)
# --------------------------------------------------------------------------- #


def messages_junk(_request: httpx.Request) -> httpx.Response:
    """A 200 Message whose text block is not a JSON object at all."""

    return httpx.Response(
        200,
        json={
            "id": "msg_junk",
            "type": "message",
            "role": "assistant",
            "model": DEFAULT_ANTHROPIC_MODEL,
            "content": [{"type": "text", "text": "not json at all"}],
            "stop_reason": "end_turn",
        },
    )


def spy_on(controller, attribute: str) -> list:
    """Record every call of ``controller.db.<attribute>`` without changing it."""
    calls: list = []
    original = getattr(controller.db, attribute)

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    setattr(controller.db, attribute, spy)
    return calls


def spy_runner(controller) -> list:
    """Record every Codex runner call (a PLAN/Cline rerun would appear here)."""
    calls: list = []
    original = controller.codex.run_structured

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    controller.codex.run_structured = spy
    return calls


@pytest.mark.parametrize(
    "handler, expected_category",
    [
        (lambda: messages_error(401, "authentication_error"), CATEGORY_AUTH),
        (lambda: messages_junk, CATEGORY_CONTRACT),
    ],
    ids=["auth", "malformed_contract"],
)
def test_non_retryable_anthropic_failure_keeps_the_accepted_report(
    tmp_path, monkeypatch, handler, expected_category
):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(anthropic=handler())
    controller = controller_for(tmp_path, http)
    try:
        prepare_accepted_report(controller)
        controller.save_supervisor_policy(
            SupervisorPolicy(
                mode="MANUAL",
                plan_provider=ID_ANTHROPIC,
                review_provider=ID_ANTHROPIC,
            )
        )
        before = safety_snapshot(controller)
        increments = spy_on(controller, "increment_attempt")
        runner_calls = spy_runner(controller)

        with pytest.raises(SupervisorRunError) as info:
            controller.run_supervisor_review()

        assert info.value.category == expected_category
        assert info.value.retryable is False

        after = safety_snapshot(controller)
        # FAILED is the pre-existing, provider-agnostic non-retryable state
        # (the parity test below proves OpenAI/DeepSeek end up identically).
        assert after["state"] == "FAILED"
        assert increments == []
        assert runner_calls == []
        assert after["attempt"] == before["attempt"] == 1
        assert after["dispatches"] == before["dispatches"] == 1
        assert after["plans"] == before["plans"]
        assert after["report"] == before["report"]
        assert controller.db.has_codex_review(1, after["attempt"]) is False
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


def test_anthropic_failure_semantics_match_the_other_providers(tmp_path, monkeypatch):
    """The SAME fault leaves the SAME controller state for every API provider."""
    guard = LiveNetworkGuard(monkeypatch)
    outcomes: dict[str, dict] = {}
    for provider in (ID_ANTHROPIC, ID_OPENAI, ID_DEEPSEEK):
        project = tmp_path / provider
        project.mkdir()
        handler = (
            messages_error(401, "authentication_error", "bad key")
            if provider == ID_ANTHROPIC
            else chat_error(401, "invalid api key")
        )
        http = RouterHttp(anthropic=handler, openai=handler, deepseek=handler)
        controller = controller_for(project, http)
        try:
            prepare_accepted_report(controller)
            controller.save_supervisor_policy(
                SupervisorPolicy(
                    mode="MANUAL", plan_provider=provider, review_provider=provider
                )
            )
            before = safety_snapshot(controller)
            increments = spy_on(controller, "increment_attempt")
            raised = None
            try:
                controller.run_supervisor_review()
            except SupervisorRunError as exc:
                raised = (exc.category, exc.retryable)
            after = safety_snapshot(controller)
            outcomes[provider] = {
                "raised": raised,
                "attempt": (before["attempt"], after["attempt"]),
                "state": after["state"],
                "report_same": after["report"] == before["report"],
                "dispatches": (before["dispatches"], after["dispatches"]),
                "plans_same": after["plans"] == before["plans"],
                "increments": increments,
                "review_row": controller.db.has_codex_review(1, after["attempt"]),
            }
        finally:
            http.close()
            controller.db.close()

    first = outcomes[ID_ANTHROPIC]
    assert first["raised"] == (CATEGORY_AUTH, False)
    assert first["attempt"] == (1, 1)
    assert first["state"] == "FAILED"
    assert first["report_same"] is True
    assert first["dispatches"] == (1, 1)
    assert first["plans_same"] is True
    assert first["increments"] == []
    assert first["review_row"] is False
    # Anthropic behaves exactly like the providers that were already registered.
    for provider, outcome in outcomes.items():
        assert outcome == first, provider
    guard.assert_silent()


def test_successful_anthropic_review_records_nothing_extra(tmp_path, monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(anthropic=messages_ok(REVIEW_APPROVE))
    controller = controller_for(tmp_path, http)
    try:
        prepare_accepted_report(controller)
        controller.save_supervisor_policy(
            SupervisorPolicy(
                mode="MANUAL",
                plan_provider=ID_ANTHROPIC,
                review_provider=ID_ANTHROPIC,
            )
        )
        before = safety_snapshot(controller)
        increments = spy_on(controller, "increment_attempt")
        runner_calls = spy_runner(controller)

        review = controller.run_supervisor_review()

        assert review["verdict"] == "APPROVE"
        assert review["step_no"] == 1  # identity comes from the controller
        after = safety_snapshot(controller)
        assert after["state"] == "REVIEW_APPROVED"
        assert after["attempt"] == before["attempt"] == 1
        assert after["dispatches"] == before["dispatches"] == 1
        assert after["plans"] == before["plans"]
        assert after["report"] == before["report"]
        assert increments == []
        assert runner_calls == []
        assert controller.db.has_codex_review(1, 1) is True
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()


@pytest.mark.parametrize(
    "handler, expected_category",
    [
        (lambda: messages_error(429, "rate_limit_error"), CATEGORY_RATE_LIMIT),
        (lambda: messages_error(503, "overloaded_error"), CATEGORY_PROVIDER_UNAVAILABLE),
    ],
    ids=["rate_limit", "provider_unavailable"],
)
def test_retryable_anthropic_failure_defers_without_touching_the_project(
    tmp_path, monkeypatch, handler, expected_category
):
    guard = LiveNetworkGuard(monkeypatch)
    http = RouterHttp(anthropic=handler())
    controller = controller_for(tmp_path, http)
    try:
        prepare_accepted_report(controller)
        controller.save_supervisor_policy(
            SupervisorPolicy(
                mode="MANUAL",
                plan_provider=ID_ANTHROPIC,
                review_provider=ID_ANTHROPIC,
            )
        )
        before = safety_snapshot(controller)
        assert before["state"] == "CLINE_REPORT_RECEIVED"
        increments = spy_on(controller, "increment_attempt")
        runner_calls = spy_runner(controller)

        result = controller.run_supervisor_review()

        # Blocking-but-retryable: the pre-existing deferred state, no raise.
        assert result["deferred"] is True
        assert result["retryable"] is True
        assert result["category"] == expected_category
        assert result["provider"] == ID_ANTHROPIC
        assert result["state"] == "CODEX_REVIEW_RETRYABLE"

        after = safety_snapshot(controller)
        assert increments == []  # no project/Cline attempt consumed
        assert runner_calls == []  # Cline and PLAN were not rerun
        assert after["attempt"] == before["attempt"] == 1
        assert after["dispatches"] == before["dispatches"] == 1
        assert after["plans"] == before["plans"]
        assert after["report"] == before["report"]  # accepted report intact
        assert controller.db.has_codex_review(1, after["attempt"]) is False
    finally:
        http.close()
        controller.db.close()
    guard.assert_silent()
# --------------------------------------------------------------------------- #
# 5  CANONICAL PARITY: Anthropic == OpenAI == DeepSeek == Codex Legacy
# --------------------------------------------------------------------------- #


def test_parity_plan_payload_across_all_four_providers(tmp_path, monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    fixture = fake_plan_payload(goal="m7.4 parity")
    http = RouterHttp(
        anthropic=messages_ok(fixture),
        openai=chat_ok(fixture),
        deepseek=chat_ok(fixture),
    )
    try:
        results = {
            ID_ANTHROPIC: anthropic_adapter(http).plan(plan_request()),
            ID_OPENAI: openai_adapter(http).plan(plan_request()),
            ID_DEEPSEEK: deepseek_adapter(http).plan(plan_request()),
            PROVIDER_CODEX_LEGACY: codex_adapter(fixture, tmp_path / ".wt").plan(
                plan_request()
            ),
        }

        canonical = results[ID_ANTHROPIC].payload
        for provider, result in results.items():
            # The controller-relevant artifact is identical ...
            assert result.payload == canonical == fixture, provider
            # ... carries no provider-owned identity ...
            assert "step_no" not in result.payload, provider
            assert "attempt" not in result.payload, provider
            assert "state" not in result.payload, provider
            assert result.fallback_from is None, provider
            assert result.fallback_reason is None, provider
            # ... and lets no Anthropic-specific envelope field escape.
            for field in ("stop_reason", "content", "usage", "id", "type"):
                assert field not in result.payload, (provider, field)

        # Ignoring provider metadata, the four results are indistinguishable.
        reference = metadata_free(results[ID_ANTHROPIC])
        for provider, result in results.items():
            assert metadata_free(result) == reference, provider

        # Each result is still honestly attributed to its own provider.
        assert results[ID_ANTHROPIC].provider == ID_ANTHROPIC
        assert results[ID_ANTHROPIC].model == DEFAULT_ANTHROPIC_MODEL
        assert results[ID_OPENAI].provider == ID_OPENAI
        assert results[ID_DEEPSEEK].provider == ID_DEEPSEEK
        assert results[PROVIDER_CODEX_LEGACY].provider == DEFAULT_LEGACY_PROVIDER
    finally:
        http.close()
    guard.assert_silent()


def test_parity_review_payload_across_all_four_providers(tmp_path, monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    fixture = fake_review_payload(verdict="REVISE", summary="m7.4 parity")
    http = RouterHttp(
        anthropic=messages_ok(fixture),
        openai=chat_ok(fixture),
        deepseek=chat_ok(fixture),
    )
    try:
        results = {
            ID_ANTHROPIC: anthropic_adapter(http).review(review_request()),
            ID_OPENAI: openai_adapter(http).review(review_request()),
            ID_DEEPSEEK: deepseek_adapter(http).review(review_request()),
            PROVIDER_CODEX_LEGACY: codex_adapter(fixture, tmp_path / ".wt").review(
                review_request()
            ),
        }

        canonical = results[ID_ANTHROPIC].payload
        assert canonical["verdict"] == "REVISE"
        for provider, result in results.items():
            assert result.payload == canonical == fixture, provider
            assert result.payload["verdict"] == "REVISE", provider
            assert "step_no" not in result.payload, provider
            assert "attempt" not in result.payload, provider
            for field in ("stop_reason", "content", "usage", "id", "type"):
                assert field not in result.payload, (provider, field)

        reference = metadata_free(results[ID_ANTHROPIC])
        for provider, result in results.items():
            assert metadata_free(result) == reference, provider

        # Provider metadata (the only difference) is per-provider, never shared.
        models = {result.model for result in results.values()}
        assert len(models) == 4
    finally:
        http.close()
    guard.assert_silent()