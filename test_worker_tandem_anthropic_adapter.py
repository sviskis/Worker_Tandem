"""Anthropic supervisor adapter tests (milestone M7.1 - foundation).

The adapter is driven entirely through injected transport doubles: no test opens
a socket, calls a paid API or needs a real API key.

Two harnesses, as in M1-M6:

* ``FakeHttpTransport`` - request-shape assertions and scripted raised errors;
* ``HttpTransport`` + ``httpx.MockTransport`` - because only the REAL transport
  maps non-2xx statuses and httpx exceptions into ``SupervisorRunError``.

M7.2 pinned the deterministic content contract, the strict JSON envelope and the
canonical PLAN/REVIEW normalization. M7.4 changed the MILESTONE BOUNDARY again:
the adapter is now **registered** in the app (MANUAL selection, the AUTO route and
fallback targets/sources can all use it) while the DEFAULT provider stays the M4
``MANUAL``/``codex_legacy`` policy.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import socket
import sys

import httpx
import pytest

from supervisor import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_API_VERSION_ENV,
    ANTHROPIC_TEMPERATURE_RANGE,
    DEFAULT_ANTHROPIC_API_BASE,
    DEFAULT_ANTHROPIC_API_VERSION,
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ANTHROPIC_TIMEOUT_SEC,
    DEFAULT_LEGACY_PROVIDER,
    ID_ANTHROPIC,
    MESSAGES_PATH,
    PROVIDER_ANTHROPIC,
    REVIEW_VERDICTS,
    AnthropicSupervisorAdapter,
    FakeHttpResponse,
    FakeHttpTransport,
    HttpTransport,
    ProviderConfig,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorPort,
    SupervisorReviewRequest,
    SupervisorRunError,
    anthropic_endpoint,
    anthropic_strict_json_object,
    anthropic_usage_from_payload,
    build_messages_body,
    classify_anthropic_error,
    content_text,
    default_anthropic_config,
    fake_plan_payload,
    fake_review_payload,
    message_text,
    messages_system_prompt,
    supervisor_plan_schema,
    supervisor_review_schema,
)
from supervisor.errors import (
    CATEGORY_AUTH,
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_API_KEY,
    CATEGORY_INVALID_MODEL,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_TEMPORARY_5XX,
)
from supervisor.models import (
    HEALTH_AUTH_ERROR,
    HEALTH_KEY_MISSING,
    HEALTH_RATE_LIMIT,
    HEALTH_READY,
)

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"
ADAPTER_SRC = (
    pathlib.Path(__file__).resolve().parent / "supervisor" / "anthropic_adapter.py"
)

#: Obviously synthetic: never a real credential, and not in any key format.
TEST_KEY = "test-anthropic-key-not-a-real-credential"
OTHER_MODEL = "claude-test-model-1"


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem_m71", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_m71"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def plan_request(prompt: str = "PLAN prompt") -> SupervisorPlanRequest:
    return SupervisorPlanRequest(
        step_no=1,
        attempt=1,
        risk="LOW",
        requires_human=False,
        prompt=prompt,
        context={},
        schema=supervisor_plan_schema(),
    )


def review_request(prompt: str = "REVIEW prompt") -> SupervisorReviewRequest:
    return SupervisorReviewRequest(
        step_no=1,
        attempt=1,
        risk="LOW",
        requires_human=False,
        prompt=prompt,
        context={},
        schema=supervisor_review_schema(),
    )


def messages_envelope(
    content, *, usage=None, model: str = DEFAULT_ANTHROPIC_MODEL
) -> dict:
    """Anthropic Messages envelope: ``content`` is a LIST of typed blocks."""
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": (
            content if isinstance(content, list) else [{"type": "text", "text": content}]
        ),
        "stop_reason": "end_turn",
    }
    if usage is not None:
        body["usage"] = usage
    return body


def json_response(body, *, status_code: int = 200) -> FakeHttpResponse:
    return FakeHttpResponse(status_code=status_code, payload=body, text=json.dumps(body))


def config(**overrides) -> ProviderConfig:
    return default_anthropic_config(**overrides)


def adapter_with(transport, *, cfg=None, env=None, clock=None):
    """Adapter + an injected transport double (never the network)."""
    return AnthropicSupervisorAdapter(
        cfg or config(),
        transport,
        env={ANTHROPIC_API_KEY_ENV: TEST_KEY} if env is None else env,
        clock=clock,
    )


def fake_transport_adapter(body, *, cfg=None, env=None, clock=None, transport=None):
    """Adapter + ``FakeHttpTransport`` for success-path tests."""
    transport = transport or FakeHttpTransport([json_response(body)])
    return adapter_with(transport, cfg=cfg, env=env, clock=clock), transport


def mock_transport_adapter(handler, *, cfg=None, env=None):
    """Adapter + the REAL ``HttpTransport`` over ``httpx.MockTransport``."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(client=client)
    return adapter_with(transport, cfg=cfg, env=env), transport


def json_handler(body, *, status_code: int = 200):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=body)

    return handler


def ticking_clock(first: float, second: float):
    values = iter([first, second])
    last = [second]

    def clock() -> float:
        try:
            return next(values)
        except StopIteration:
            return last[0]

    return clock


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
# 1-10  Config factory, identity, endpoints, headers, API version
# --------------------------------------------------------------------------- #


def test_provider_id_and_display_name():
    cfg = default_anthropic_config()
    assert cfg.provider_id == PROVIDER_ANTHROPIC == "anthropic"
    assert cfg.display_name == "Claude"
    # The router already reserved the same id for AUTO routing (M8).
    assert PROVIDER_ANTHROPIC == ID_ANTHROPIC
    assert cfg.model == DEFAULT_ANTHROPIC_MODEL


def test_config_factory_defaults():
    cfg = default_anthropic_config()
    assert cfg.api_base == DEFAULT_ANTHROPIC_API_BASE == "https://api.anthropic.com/v1"
    assert cfg.api_key_source == f"env:{ANTHROPIC_API_KEY_ENV}"
    assert cfg.api_key_source == "env:ANTHROPIC_API_KEY"
    assert cfg.timeout_sec == DEFAULT_ANTHROPIC_TIMEOUT_SEC
    assert cfg.max_output_tokens == DEFAULT_ANTHROPIC_MAX_TOKENS
    assert cfg.temperature is None
    assert cfg.enabled is True
    # Anthropic has no response_format field: JSON is prompt-enforced.
    assert cfg.supports_structured_output is False
    assert cfg.priority == 3
    assert cfg.cost_policy == "premium"


def test_config_factory_overrides():
    cfg = default_anthropic_config(
        model=OTHER_MODEL, api_base="https://proxy.example/v1/", timeout_sec=7
    )
    assert cfg.model == OTHER_MODEL
    assert cfg.api_base == "https://proxy.example/v1/"
    assert cfg.timeout_sec == 7
    adapter = adapter_with(FakeHttpTransport(), cfg=cfg)
    assert adapter.model == OTHER_MODEL
    # A trailing slash is normalised away for endpoint construction.
    assert adapter.api_base == "https://proxy.example/v1"
    assert adapter._timeout_sec() == 7


def test_api_key_source_is_a_reference_and_env_name_is_exposed():
    adapter = adapter_with(FakeHttpTransport())
    assert adapter.api_key_source == "env:ANTHROPIC_API_KEY"
    assert adapter.api_key_env() == ANTHROPIC_API_KEY_ENV

    custom = adapter_with(
        FakeHttpTransport(),
        cfg=config(api_key_source="env:MY_CLAUDE_KEY"),
        env={"MY_CLAUDE_KEY": TEST_KEY},
    )
    assert custom.api_key_env() == "MY_CLAUDE_KEY"
    assert custom._api_key() == TEST_KEY


def test_endpoint_construction_default_and_configured():
    adapter = adapter_with(FakeHttpTransport())
    assert adapter.messages_endpoint() == (
        f"{DEFAULT_ANTHROPIC_API_BASE}{MESSAGES_PATH}"
    )
    assert adapter.health_endpoint() == f"{DEFAULT_ANTHROPIC_API_BASE}/models"
    assert MESSAGES_PATH == "/messages"

    custom = adapter_with(
        FakeHttpTransport(), cfg=config(api_base="https://proxy.example/v1")
    )
    assert custom.messages_endpoint() == "https://proxy.example/v1/messages"
    assert anthropic_endpoint(None, MESSAGES_PATH) == (
        f"{DEFAULT_ANTHROPIC_API_BASE}{MESSAGES_PATH}"
    )


def test_headers_use_x_api_key_and_never_a_bearer_token():
    adapter = adapter_with(FakeHttpTransport())
    headers = adapter._headers(TEST_KEY)
    assert headers["x-api-key"] == TEST_KEY
    assert headers["anthropic-version"] == DEFAULT_ANTHROPIC_API_VERSION
    assert headers["Content-Type"] == "application/json"
    # Anthropic does NOT use Authorization: Bearer.
    assert "Authorization" not in headers
    assert not any(str(value).startswith("Bearer") for value in headers.values())


def test_api_version_default_env_override_and_config_override():
    assert DEFAULT_ANTHROPIC_API_VERSION == "2023-06-01"
    assert adapter_with(FakeHttpTransport()).api_version() == (
        DEFAULT_ANTHROPIC_API_VERSION
    )

    from_env = adapter_with(
        FakeHttpTransport(),
        env={
            ANTHROPIC_API_KEY_ENV: TEST_KEY,
            ANTHROPIC_API_VERSION_ENV: "2099-01-01",
        },
    )
    assert from_env.api_version() == "2099-01-01"
    assert from_env._headers(TEST_KEY)["anthropic-version"] == "2099-01-01"

    # An explicit config attribute wins over the environment: a future
    # ProviderConfig.api_version field would be honoured through getattr, so the
    # shared frozen dataclass does not have to change for M7.1.
    explicit = config()
    object.__setattr__(explicit, "api_version", "2000-01-01")
    adapter = adapter_with(
        FakeHttpTransport(),
        cfg=explicit,
        env={
            ANTHROPIC_API_KEY_ENV: TEST_KEY,
            ANTHROPIC_API_VERSION_ENV: "2099-01-01",
        },
    )
    assert adapter.api_version() == "2000-01-01"


def test_body_skeleton_has_top_level_system_and_mandatory_max_tokens():
    body = build_messages_body(
        model=OTHER_MODEL, system_prompt="SYS", prompt="USER", max_output_tokens=None
    )
    assert body["model"] == OTHER_MODEL
    assert body["system"] == "SYS"  # top-level, NOT a message turn
    assert body["messages"] == [{"role": "user", "content": "USER"}]
    assert body["max_tokens"] == DEFAULT_ANTHROPIC_MAX_TOKENS  # always present
    assert "response_format" not in body  # no such field exists
    assert "stream" not in body
    assert "temperature" not in body

    capped = build_messages_body(
        model=OTHER_MODEL,
        system_prompt="SYS",
        prompt="USER",
        max_output_tokens=1234,
        temperature=0.2,
    )
    assert capped["max_tokens"] == 1234
    assert capped["temperature"] == 0.2


def test_system_prompt_states_the_schema_and_the_identity_rule():
    for operation, schema in (
        ("PLAN", supervisor_plan_schema()),
        ("REVIEW", supervisor_review_schema()),
    ):
        prompt = messages_system_prompt(operation)
        assert "ONE JSON object" in prompt
        assert "step_no, attempt or state" in prompt
        assert json.dumps(schema, sort_keys=True, separators=(",", ":")) in prompt
    assert "Planner/Architect" in messages_system_prompt("PLAN")
    assert "Reviewer/Supervisor" in messages_system_prompt("REVIEW")


# --------------------------------------------------------------------------- #
# 11-17  API key + local preconditions: non-retryable, and NO HTTP call
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("operation", ["PLAN", "REVIEW"])
def test_missing_key_fails_locally_without_any_http_call(operation):
    transport = FakeHttpTransport()
    adapter = adapter_with(transport, env={})

    request = plan_request() if operation == "PLAN" else review_request()
    call = adapter.plan if operation == "PLAN" else adapter.review

    with pytest.raises(SupervisorRunError) as info:
        call(request)

    exc = info.value
    assert exc.category == CATEGORY_INVALID_API_KEY
    assert exc.retryable is False  # fail closed: never a fallback trigger
    assert exc.provider == PROVIDER_ANTHROPIC
    assert exc.operation == operation
    assert exc.model == DEFAULT_ANTHROPIC_MODEL
    assert ANTHROPIC_API_KEY_ENV in exc.sanitized_message  # NAME only
    assert TEST_KEY not in exc.sanitized_message
    assert transport.requests == []  # local precondition: no request at all


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", None])
def test_blank_key_is_treated_as_missing_without_any_http_call(blank):
    transport = FakeHttpTransport()
    adapter = adapter_with(transport, env={ANTHROPIC_API_KEY_ENV: blank})

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_API_KEY
    assert info.value.retryable is False
    assert transport.requests == []
    # Health agrees, and also performs no request.
    health = adapter.health_check()
    assert health.status == HEALTH_KEY_MISSING
    assert health.ready is False
    assert ANTHROPIC_API_KEY_ENV in health.detail
    assert transport.requests == []


@pytest.mark.parametrize("operation", ["PLAN", "REVIEW"])
def test_empty_model_fails_locally_without_any_http_call(operation):
    transport = FakeHttpTransport()
    adapter = adapter_with(transport, cfg=config(model=""), env=None)
    assert adapter.model is None

    request = plan_request() if operation == "PLAN" else review_request()
    call = adapter.plan if operation == "PLAN" else adapter.review

    with pytest.raises(SupervisorRunError) as info:
        call(request)

    exc = info.value
    assert exc.category == CATEGORY_INVALID_REQUEST
    assert exc.retryable is False
    assert "model" in exc.sanitized_message.lower()
    assert transport.requests == []


@pytest.mark.parametrize("temperature", [-0.1, 1.5, "hot", [0.5]])
def test_out_of_range_temperature_fails_locally_without_any_http_call(temperature):
    """Anthropic accepts temperature in [0, 1] only - reject it locally."""
    low, high = ANTHROPIC_TEMPERATURE_RANGE
    assert (low, high) == (0.0, 1.0)
    transport = FakeHttpTransport()
    adapter = adapter_with(transport, cfg=config(temperature=temperature))

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False
    assert "temperature" in info.value.sanitized_message
    assert transport.requests == []


def test_boundary_temperatures_are_accepted_offline():
    for value in (0.0, 0.5, 1.0):
        body = build_messages_body(
            model=OTHER_MODEL, system_prompt="S", prompt="P", temperature=value
        )
        assert body["temperature"] == value


def test_no_plaintext_key_is_stored_or_displayed_on_the_adapter(monkeypatch):
    adapter = adapter_with(FakeHttpTransport())  # the key IS in the env

    assert adapter._api_key() == TEST_KEY  # resolved at call time only

    # The adapter caches NO secret field: its attribute set is exactly this.
    assert set(vars(adapter)) == {
        "config",
        "provider_id",
        "model",
        "api_base",
        "api_key_source",
        "_transport",
        "_env",
        "_clock",
        "_api_version_override",
    }
    assert TEST_KEY not in str(adapter.config)
    assert TEST_KEY not in str(adapter.api_key_source)
    assert TEST_KEY not in str(vars(adapter)["api_key_source"])

    # In the production path (no injected env) nothing on the adapter - and no
    # output it produces - can contain the process credential.
    process_key = "test-anthropic-process-credential"
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, process_key)
    live = AnthropicSupervisorAdapter(config(), FakeHttpTransport())
    assert set(vars(live)) == set(vars(adapter))
    assert live._api_key() == process_key
    for name, value in vars(live).items():
        assert process_key not in str(value), name
    assert process_key not in str(live)
    assert process_key not in json.dumps(vars(live), default=str)

    # Health detail and errors name only the env var.
    healthy = adapter_with(
        FakeHttpTransport(default=json_response({"data": [{"id": "claude-test"}]}))
    )
    assert TEST_KEY not in str(healthy.health_check())
    failing = adapter_with(FakeHttpTransport(), env={})
    with pytest.raises(SupervisorRunError) as info:
        failing.review(review_request())
    assert info.value.category == CATEGORY_INVALID_API_KEY
    assert TEST_KEY not in str(info.value)
    assert TEST_KEY not in json.dumps(info.value.to_dict(), default=str)


# --------------------------------------------------------------------------- #
# 18-22  Transport contract: endpoint, method, headers, health probe
# --------------------------------------------------------------------------- #


def test_plan_posts_to_messages_with_anthropic_headers_and_body():
    plan = fake_plan_payload()
    adapter, transport = fake_transport_adapter(
        messages_envelope(json.dumps(plan)), clock=ticking_clock(10.0, 10.25)
    )

    result = adapter.plan(plan_request("MY PROMPT"))

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == adapter.messages_endpoint()
    assert request["url"].endswith("/v1/messages")
    assert request["url"] != f"{DEFAULT_ANTHROPIC_API_BASE}/chat/completions"
    assert request["timeout_sec"] == DEFAULT_ANTHROPIC_TIMEOUT_SEC
    assert request["headers"]["x-api-key"] == TEST_KEY
    assert request["headers"]["anthropic-version"] == DEFAULT_ANTHROPIC_API_VERSION

    body = request["json"]
    assert body["model"] == DEFAULT_ANTHROPIC_MODEL
    assert body["system"] == messages_system_prompt("PLAN")
    assert body["messages"] == [{"role": "user", "content": "MY PROMPT"}]
    assert body["max_tokens"] == DEFAULT_ANTHROPIC_MAX_TOKENS
    assert "step_no" not in body and "attempt" not in body

    # Canonical validation + honest telemetry (Claude reported no usage: the
    # token counts stay None while the measured latency survives).
    assert result.payload == plan
    assert result.provider == PROVIDER_ANTHROPIC
    assert result.model == DEFAULT_ANTHROPIC_MODEL
    assert result.usage is not None and result.usage.latency_ms == 250
    assert result.usage.input_tokens is None
    assert result.usage.total_tokens is None
    assert result.fallback_from is None


def test_review_posts_to_messages_and_returns_the_verdict():
    review = fake_review_payload()
    adapter, transport = fake_transport_adapter(
        messages_envelope(json.dumps(review))
    )

    result = adapter.review(review_request())

    assert result.payload == review
    assert result.payload["verdict"] == "APPROVE"
    assert transport.requests[0]["json"]["system"] == messages_system_prompt("REVIEW")


def test_configured_timeout_and_model_reach_the_request():
    adapter, transport = fake_transport_adapter(
        messages_envelope(json.dumps(fake_plan_payload())),
        cfg=config(model=OTHER_MODEL, timeout_sec=11),
    )
    adapter.plan(plan_request())
    assert transport.requests[0]["timeout_sec"] == 11
    assert transport.requests[0]["json"]["model"] == OTHER_MODEL


def test_health_probe_is_a_get_on_models_and_never_a_completion():
    transport = FakeHttpTransport(
        [json_response({"data": [{"id": "claude-a"}, {"id": "claude-b"}]})]
    )
    adapter = adapter_with(transport)

    health = adapter.health_check()

    assert health.status == HEALTH_READY
    assert health.ready is True
    assert health.provider_id == PROVIDER_ANTHROPIC
    assert health.detail == "2 models available"
    assert health.model == DEFAULT_ANTHROPIC_MODEL
    assert len(transport.requests) == 1
    assert transport.requests[0]["method"] == "GET"
    assert transport.requests[0]["url"] == adapter.health_endpoint()
    assert transport.requests[0]["url"].endswith("/v1/models")
    assert transport.requests[0]["headers"]["x-api-key"] == TEST_KEY


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (401, {"type": "error", "error": {"type": "authentication_error"}}, HEALTH_AUTH_ERROR),
        (403, {"type": "error", "error": {"type": "permission_error"}}, HEALTH_AUTH_ERROR),
        (429, {"type": "error", "error": {"type": "rate_limit_error"}}, HEALTH_RATE_LIMIT),
        (500, {"type": "error", "error": {"type": "api_error"}}, "PROVIDER_ERROR"),
        (
            529,
            {"type": "error", "error": {"type": "overloaded_error"}},
            "PROVIDER_ERROR",
        ),
    ],
)
def test_health_maps_real_statuses_to_provider_health(status_code, body, expected):
    adapter, _ = mock_transport_adapter(json_handler(body, status_code=status_code))
    health = adapter.health_check()
    assert health.status == expected


def test_health_key_missing_detail_names_only_the_variable():
    adapter = adapter_with(FakeHttpTransport(), env={})
    health = adapter.health_check()
    assert health.status == HEALTH_KEY_MISSING
    assert health.detail == f"{ANTHROPIC_API_KEY_ENV} is not set"
    assert health.model == DEFAULT_ANTHROPIC_MODEL


# --------------------------------------------------------------------------- #
# 23-30  Parsing, canonical validation, usage, error classification
# --------------------------------------------------------------------------- #


def test_plan_succeeds_through_the_real_transport_and_maps_usage():
    plan = fake_plan_payload()
    envelope = messages_envelope(
        json.dumps(plan), usage={"input_tokens": 12, "output_tokens": 5}
    )
    adapter, _ = mock_transport_adapter(json_handler(envelope))

    result = adapter.plan(plan_request())

    assert result.payload == plan
    assert result.usage is not None
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 17
    assert result.usage.estimated_cost is None


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '   {"a": 1}   ',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
    ],
)
def test_strict_envelope_accepts_one_object(text):
    assert anthropic_strict_json_object(text) == {"a": 1}


@pytest.mark.parametrize(
    "text",
    [
        'prose before {"a": 1}',
        '{"a": 1} trailing prose',
        '{"a": 1}{"b": 2}',
        '[{"a": 1}]',
        '"just a string"',
        "not json at all",
        "",
        None,
    ],
)
def test_strict_envelope_rejects_anything_but_one_object(text):
    assert anthropic_strict_json_object(text) is None


def test_prose_content_fails_closed_as_contract():
    adapter, _ = fake_transport_adapter(
        messages_envelope('Here it is: {"decision": "PLAN_READY"}')
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False


def test_one_text_block_with_non_text_blocks_is_accepted():
    """Deterministic content: exactly one text block, extras carry no payload."""
    payload = {
        "content": [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": '{"decision": "PLAN_READY"}'},
            {"type": "tool_use", "id": "toolu_1", "name": "noop", "input": {}},
        ]
    }
    text, problem = content_text(payload)
    assert problem is None
    assert text == '{"decision": "PLAN_READY"}'
    # The non-raising convenience view agrees with the strict decision.
    assert message_text(payload) == text


def test_multiple_text_blocks_are_ambiguous_and_are_rejected():
    """M7.2: unrelated text blocks are NEVER concatenated."""
    payload = {
        "content": [
            {"type": "text", "text": '{"decision":'},
            {"type": "text", "text": ' "PLAN_READY"}'},
        ]
    }
    text, problem = content_text(payload)
    assert text is None
    assert "2 text blocks" in problem and "ambiguous" in problem
    assert message_text(payload) is None

    # ... and the adapter fails closed instead of stitching them together.
    adapter, _ = fake_transport_adapter(payload)
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False
    assert "ambiguous" in info.value.sanitized_message


@pytest.mark.parametrize(
    ("payload", "expected_problem"),
    [
        (None, "not a JSON object"),
        ({"content": "raw"}, "no content block list"),
        ({"content": []}, "block list is empty"),
        ({"content": ["text"]}, "block 0 is not an object"),
        ({"content": [{"text": "no type"}]}, "block 0 has no type"),
        ({"content": [{"type": "  "}]}, "block 0 has no type"),
        ({"content": [{"type": "text"}]}, "no string text"),
        ({"content": [{"type": "text", "text": 5}]}, "no string text"),
        ({"content": [{"type": "text", "text": "   "}]}, "text block is empty"),
        ({"content": [{"type": "thinking", "thinking": "x"}]}, "no text block"),
    ],
)
def test_every_content_problem_fails_closed(payload, expected_problem):
    text, problem = content_text(payload)
    assert text is None
    assert expected_problem in problem
    assert message_text(payload) is None

    adapter, _ = fake_transport_adapter(payload)
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False
    assert expected_problem in info.value.sanitized_message
    # A rejection never echoes provider content.
    assert info.value.sanitized_message.startswith("Anthropic content rejected:")


def test_canonically_invalid_payload_fails_closed():
    adapter, _ = fake_transport_adapter(
        messages_envelope(json.dumps({"verdict": "MAYBE"}))
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False


@pytest.mark.parametrize(
    ("payload", "expected_total"),
    [
        ({"usage": {"input_tokens": 3, "output_tokens": 4}}, 7),
        ({"usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 99}}, 99),
        ({"usage": {"input_tokens": 3}}, 3),
    ],
)
def test_usage_mapping_unit(payload, expected_total):
    usage = anthropic_usage_from_payload(payload, latency_ms=250)
    assert usage is not None
    assert usage.total_tokens == expected_total
    assert usage.latency_ms == 250
    assert usage.estimated_cost is None


def test_usage_mapping_stays_honest_when_nothing_is_reported():
    # A ProviderUsage is ALWAYS returned (so latency is never lost), but no
    # token count is ever invented and cost is never guessed.
    empty = anthropic_usage_from_payload({}, latency_ms=None)
    assert empty.input_tokens is None and empty.output_tokens is None
    assert empty.total_tokens is None and empty.latency_ms is None
    assert empty.estimated_cost is None

    no_usage = anthropic_usage_from_payload(None, latency_ms=42)
    assert no_usage.total_tokens is None
    assert no_usage.latency_ms == 42

    # Provider numeric quirks are coerced exactly like the shared mappers ...
    coerced = anthropic_usage_from_payload({"usage": {"input_tokens": "5"}})
    assert coerced.input_tokens == 5 and coerced.total_tokens == 5
    # ... but booleans are never treated as counts.
    boolean = anthropic_usage_from_payload({"usage": {"input_tokens": True}})
    assert boolean.input_tokens is None


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (401, {"error": {"type": "authentication_error"}}, CATEGORY_AUTH),
        (403, {"error": {"type": "permission_error"}}, CATEGORY_AUTH),
        (400, {"error": {"type": "invalid_request_error"}}, CATEGORY_INVALID_REQUEST),
        (
            400,
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": "unknown model claude-x",
                }
            },
            CATEGORY_INVALID_MODEL,
        ),
        (
            404,
            {"error": {"type": "not_found_error", "message": "model not found"}},
            CATEGORY_INVALID_MODEL,
        ),
        (429, {"error": {"type": "rate_limit_error"}}, CATEGORY_RATE_LIMIT),
        (
            429,
            {"error": {"type": "rate_limit_error", "message": "try again at 12:00"}},
            CATEGORY_QUOTA_WITH_RESET,
        ),
        (500, {"error": {"type": "api_error"}}, CATEGORY_PROVIDER_UNAVAILABLE),
        (529, {"error": {"type": "overloaded_error"}}, CATEGORY_PROVIDER_UNAVAILABLE),
        (402, {"error": {"type": "billing_error"}}, CATEGORY_PAYMENT_REQUIRED),
    ],
)
def test_error_classification_uses_anthropic_error_types(status_code, body, expected):
    from supervisor.errors import is_retryable_category

    adapter, _ = mock_transport_adapter(json_handler(body, status_code=status_code))
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    exc = info.value
    assert exc.category == expected
    assert exc.provider == PROVIDER_ANTHROPIC
    assert exc.model == DEFAULT_ANTHROPIC_MODEL
    assert exc.operation == "PLAN"
    assert exc.http_status == status_code
    assert exc.retryable is is_retryable_category(expected)


def test_transport_exceptions_are_refined_and_stamped():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("offline timeout")

    adapter, _ = mock_transport_adapter(handler)
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())

    exc = info.value
    assert exc.category == "timeout"
    assert exc.retryable is True
    assert exc.provider == PROVIDER_ANTHROPIC
    assert exc.operation == "REVIEW"
    assert exc.model == DEFAULT_ANTHROPIC_MODEL
    # Never a self-referential cause.
    assert exc.__cause__ is not exc


def test_classify_anthropic_error_returns_none_for_a_plain_baseline():
    from supervisor import SupervisorRunError as Err

    plain = Err("boom", category="provider_unavailable", http_status=503)
    assert classify_anthropic_error(plain) is None  # baseline kept
    billing = Err(
        "credit balance is too low",
        category="invalid_request",
        http_status=400,
        provider_error_code="invalid_request_error",
    )
    assert classify_anthropic_error(billing) == CATEGORY_PAYMENT_REQUIRED


# --------------------------------------------------------------------------- #
# 31-35  SupervisorPort conformance, isolation and the M7.1 milestone boundary
# --------------------------------------------------------------------------- #


def test_adapter_satisfies_the_supervisor_port():
    adapter = adapter_with(FakeHttpTransport())
    assert isinstance(adapter, SupervisorPort)
    assert isinstance(adapter, AnthropicSupervisorAdapter)
    assert callable(adapter.plan)
    assert callable(adapter.review)
    assert callable(adapter.health_check)
    # The transport seam is only the injected port: the adapter never builds one.
    assert adapter._transport.__class__ is FakeHttpTransport


def test_no_live_network_is_touched_during_a_full_offline_cycle(monkeypatch):
    guard = LiveNetworkGuard(monkeypatch)
    transport = FakeHttpTransport(
        [
            json_response(messages_envelope(json.dumps(fake_plan_payload()))),
            json_response(messages_envelope(json.dumps(fake_review_payload()))),
        ],
        default=json_response({"data": [{"id": "claude-a"}]}),
    )
    adapter = adapter_with(transport, clock=ticking_clock(0.0, 0.05))

    assert adapter.plan(plan_request()).payload == fake_plan_payload()
    assert adapter.review(review_request()).payload == fake_review_payload()
    assert adapter.health_check().status == HEALTH_READY
    assert len(transport.requests) == 3  # two completions + one cheap probe

    # Even the failing (no key) cycle stays local.
    offline = adapter_with(FakeHttpTransport(), env={})
    with pytest.raises(SupervisorRunError):
        offline.plan(plan_request())
    with pytest.raises(SupervisorRunError):
        offline.review(review_request())
    assert offline.health_check().status == HEALTH_KEY_MISSING

    guard.assert_silent()


def test_module_imports_no_provider_sdk_and_no_http_client():
    tree = ast.parse(ADAPTER_SRC.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])

    forbidden = {"anthropic", "openai", "requests", "urllib", "urllib3", "httpx", "http"}
    assert not (modules & forbidden), modules & forbidden
    # Only the standard library plus supervisor internals, and only the
    # protocol-independent shared helpers (never another provider adapter).
    assert modules <= {
        "__future__",
        "json",
        "os",
        "time",
        "typing",
        "base",
        "errors",
        "models",
        "openai_compatible",
        "schemas",
        "transport",
    }, modules
    assert "deepseek_adapter" not in modules
    assert "openai_adapter" not in modules


def test_anthropic_is_registered_without_changing_the_default(tmp_path):
    """M7.4 boundary: the adapter is buildable, the M4 default is untouched."""
    wt = _load_worker_tandem()

    assert wt.TandemController.supervisor_provider_available(PROVIDER_ANTHROPIC) is True
    assert PROVIDER_ANTHROPIC in wt.SUPERVISOR_BUILDABLE_PROVIDERS
    assert wt.ANTHROPIC_API_KEY_ENV == ANTHROPIC_API_KEY_ENV

    controller = wt.TandemController(tmp_path)
    try:
        # No saved policy: still MANUAL + codex_legacy only. Registration must
        # never change the DEFAULT provider just because Claude now exists.
        assert controller.supervisor_router.registered_providers() == (
            DEFAULT_LEGACY_PROVIDER,
        )
        assert controller.supervisor_policy.plan_provider == DEFAULT_LEGACY_PROVIDER
        assert controller.supervisor_policy.review_provider == DEFAULT_LEGACY_PROVIDER
        # Buildable on demand with the Claude default model (no I/O).
        provider = controller.supervisor_provider(PROVIDER_ANTHROPIC)
        assert isinstance(provider, AnthropicSupervisorAdapter)
        assert provider.model == DEFAULT_ANTHROPIC_MODEL
    finally:
        controller.db.close()


def test_defaults_and_auto_policy_are_unchanged_by_m71():
    # The default policy is still the M4 one.
    policy = SupervisorPolicy()
    assert policy.mode == "MANUAL"
    assert policy.plan_provider == DEFAULT_LEGACY_PROVIDER
    assert policy.review_provider == DEFAULT_LEGACY_PROVIDER
    assert policy.fallback_chain == ()
    assert policy.max_provider_retries == 0

    # AUTO still DECLARES Anthropic for HIGH/REVIEW (M8) while the app cannot
    # build it yet (M7.1 leaves that gap open, exactly as documented).
    from supervisor import default_auto_policy

    auto = default_auto_policy()
    assert auto.candidates_for("REVIEW", "HIGH")[0] == PROVIDER_ANTHROPIC
    assert auto.is_auto is True


# --------------------------------------------------------------------------- #
# M7.2 - PLAN execution + canonical normalization (no Anthropic field escapes)
# --------------------------------------------------------------------------- #


def plan_adapter(content: str):
    """Adapter whose next PLAN response carries ``content`` as its one text block."""
    return fake_transport_adapter(messages_envelope(content))


def test_m72_plan_success_is_canonical_and_identity_free():
    plan = fake_plan_payload()
    adapter, transport = plan_adapter(json.dumps(plan))

    result = adapter.plan(plan_request())

    # Exactly the canonical contract - no Anthropic-specific field escapes.
    assert result.payload == plan
    assert set(result.payload) == set(fake_plan_payload())
    assert "step_no" not in result.payload
    assert "attempt" not in result.payload
    assert "state" not in result.payload
    assert result.provider == PROVIDER_ANTHROPIC
    assert result.model == DEFAULT_ANTHROPIC_MODEL
    assert result.fallback_from is None and result.fallback_reason is None
    # The provider only ever saw system + user prompt.
    body = transport.requests[0]["json"]
    assert "step_no" not in body and "attempt" not in body and "state" not in body


def test_m72_plan_accepts_plain_json():
    adapter, _ = plan_adapter("  " + json.dumps(fake_plan_payload()) + "  ")
    assert adapter.plan(plan_request()).payload == fake_plan_payload()


def test_m72_plan_accepts_a_single_fenced_json_object():
    adapter, _ = plan_adapter("```json\n" + json.dumps(fake_plan_payload()) + "\n```")
    assert adapter.plan(plan_request()).payload == fake_plan_payload()


def test_m72_fenced_block_is_delimited_by_its_fence():
    """Shared-helper semantics: a closed fence delimits the JSON object."""
    adapter, _ = plan_adapter(
        "```json\n" + json.dumps(fake_plan_payload()) + "\n```\nHope that helps!"
    )
    assert adapter.plan(plan_request()).payload == fake_plan_payload()


@pytest.mark.parametrize(
    "content",
    [
        '{"decision": "PLAN_READY",}',  # malformed JSON (trailing comma)
        '{"decision": }',  # malformed JSON
        '[{"decision": "PLAN_READY"}]',  # array root
        '"PLAN_READY"',  # scalar root
        "42",  # numeric root
        '{"decision": "PLAN_READY"}{"decision": "BLOCKED"}',  # two objects
    ],
)
def test_m72_non_object_or_malformed_json_is_rejected(content):
    adapter, _ = plan_adapter(content)
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False
    assert "exactly one JSON object" in info.value.sanitized_message


@pytest.mark.parametrize(
    "content",
    [
        'Here is the plan: {"decision": "PLAN_READY"}',  # prose + JSON
        "```json\n" + json.dumps(fake_plan_payload()) + "\nnot json",  # fence + junk
    ],
)
def test_m72_prose_around_json_is_rejected(content):
    adapter, _ = plan_adapter(content)
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT


def test_m72_json_followed_by_prose_is_rejected():
    adapter, _ = plan_adapter(json.dumps(fake_plan_payload()) + " That is my plan.")
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT


@pytest.mark.parametrize(
    ("field", "value"),
    [("step_no", 1), ("attempt", 2), ("state", "PLAN_READY")],
)
def test_m72_controller_owned_identity_is_rejected(field, value):
    """The provider must never own step_no / attempt / state."""
    polluted = {**fake_plan_payload(), field: value}
    adapter, _ = plan_adapter(json.dumps(polluted))

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    exc = info.value
    assert exc.category == CATEGORY_CONTRACT
    assert exc.retryable is False
    assert field in exc.sanitized_message  # the offending key is named
    assert exc.operation == "PLAN"


@pytest.mark.parametrize(
    "extra",
    [
        {"stop_reason": "end_turn"},
        {"model": DEFAULT_ANTHROPIC_MODEL},
        {"usage": {"input_tokens": 1}},
        {"served_by": "anthropic"},
    ],
)
def test_m72_provider_specific_fields_never_escape(extra):
    adapter, _ = plan_adapter(json.dumps({**fake_plan_payload(), **extra}))
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert next(iter(extra)) in info.value.sanitized_message


# --------------------------------------------------------------------------- #
# M7.2 - REVIEW execution: only the four canonical verdicts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("verdict", list(REVIEW_VERDICTS))
def test_m72_review_accepts_every_canonical_verdict(verdict):
    payload = {**fake_review_payload(), "verdict": verdict}
    adapter, _ = fake_transport_adapter(messages_envelope(json.dumps(payload)))

    result = adapter.review(review_request())

    assert result.payload == payload
    assert result.payload["verdict"] == verdict
    assert set(result.payload) == set(fake_review_payload())
    assert result.provider == PROVIDER_ANTHROPIC


@pytest.mark.parametrize(
    "verdict",
    ["MAYBE", "APPROVED", "approve", "Approve", "REVISION", "", 5, None, True],
)
def test_m72_review_rejects_every_other_verdict(verdict):
    payload = {**fake_review_payload(), "verdict": verdict}
    adapter, _ = fake_transport_adapter(messages_envelope(json.dumps(payload)))

    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())

    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False
    assert info.value.operation == "REVIEW"


@pytest.mark.parametrize(
    "payload",
    [
        {"verdict": "APPROVE"},  # missing required fields
        {k: v for k, v in fake_review_payload().items() if k != "verdict"},
        {**fake_review_payload(), "issues": "none"},  # wrong type
        {**fake_review_payload(), "required_changes": "nothing"},  # wrong type
    ],
)
def test_m72_malformed_review_is_rejected(payload):
    adapter, _ = fake_transport_adapter(messages_envelope(json.dumps(payload)))
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_CONTRACT


@pytest.mark.parametrize(
    ("field", "value"),
    [("step_no", 1), ("attempt", 2), ("state", "REVIEW_APPROVED")],
)
def test_m72_review_identity_is_rejected(field, value):
    polluted = {**fake_review_payload(), field: value}
    adapter, _ = fake_transport_adapter(messages_envelope(json.dumps(polluted)))

    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())

    assert info.value.category == CATEGORY_CONTRACT
    assert field in info.value.sanitized_message


def test_m72_review_succeeds_through_the_real_transport():
    payload = {**fake_review_payload(), "verdict": "REVISE"}
    adapter, _ = mock_transport_adapter(
        json_handler(
            messages_envelope(
                json.dumps(payload), usage={"input_tokens": 9, "output_tokens": 4}
            )
        )
    )

    result = adapter.review(review_request())

    assert result.payload["verdict"] == "REVISE"
    assert result.usage.input_tokens == 9 and result.usage.output_tokens == 4


def test_m72_prompt_requires_json_only_and_names_every_identity_key():
    for operation in ("PLAN", "REVIEW"):
        prompt = messages_system_prompt(operation)
        assert "ONE JSON object" in prompt
        assert "nothing else" in prompt
        assert "no prose before or after" in prompt
        for key in ("step_no", "attempt", "state"):
            assert key in prompt
        assert "Worker_tandem owns them" in prompt


# --------------------------------------------------------------------------- #
# M7.2 - JSON strictness: shared helper, no brace scanning, no behaviour drift
# --------------------------------------------------------------------------- #


def test_m72_embedded_object_brace_scanning_is_never_used():
    """A JSON object hidden inside prose is NOT scavenged out of the text."""
    adapter, _ = plan_adapter(
        "Sure! Here is the plan you asked for: "
        + json.dumps(fake_plan_payload())
        + " - tell me if you want changes."
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert "exactly one JSON object" in info.value.sanitized_message


def test_m72_uses_the_shared_strict_helper_verbatim():
    """M7.2 mandates the existing strict helper: no re-forked implementation."""
    from supervisor import strict_json_object

    samples = [
        '{"a": 1}',
        '  {"a": 1}  ',
        "```json\n{\"a\": 1}\n```",
        "```\n{\"a\": 1}\n```",
        'prose {"a": 1}',
        '{"a": 1} prose',
        '{"a": 1}{"b": 2}',
        '[{"a": 1}]',
        '"scalar"',
        "42",
        "",
        None,
    ]
    for sample in samples:
        assert anthropic_strict_json_object(sample) == strict_json_object(sample)


def test_m72_does_not_weaken_the_other_providers():
    """Lenient (DeepSeek/Codex) and strict (OpenAI/Anthropic) both stay intact."""
    from supervisor import strict_json_object
    from supervisor.schemas import extract_json_object

    prose_wrapped = 'sure: {"a": 1}'
    assert extract_json_object(prose_wrapped) == {"a": 1}  # lenient path intact
    assert strict_json_object(prose_wrapped) is None  # strict path intact
    # The OpenAI adapter still uses the same shared strict helper.
    from supervisor import openai_adapter

    assert openai_adapter.strict_json_object is strict_json_object







