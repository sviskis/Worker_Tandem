"""DeepSeek supervisor adapter tests (milestone M5).

Mirrors the M3 approach: the adapter is exercised entirely through injected
transport doubles, so no test ever opens a socket or calls a paid API.

Two harnesses are used, exactly as in M1/M2:

* ``FakeHttpTransport`` (M2) - success paths, request-shape assertions and the
  no-network proof;
* ``HttpTransport`` + ``httpx.MockTransport`` (M1/M2) - because only the REAL
  transport maps non-2xx statuses and httpx exceptions into
  ``SupervisorRunError`` categories.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import socket
import sys

import httpx
import pytest

from supervisor import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_DEEPSEEK_API_BASE,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_LEGACY_PROVIDER,
    DEEPSEEK_API_KEY_ENV,
    MODELS_PATH,
    PROVIDER_DEEPSEEK,
    DeepSeekSupervisorAdapter,
    FakeHttpResponse,
    FakeHttpTransport,
    HttpTransport,
    ProviderConfig,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorPort,
    SupervisorReviewRequest,
    SupervisorRunError,
    TransportPort,
    default_deepseek_config,
    deepseek_endpoint,
    fake_plan_payload,
    fake_review_payload,
    supervisor_plan_schema,
    supervisor_review_schema,
)
from supervisor.errors import (
    CATEGORY_AUTH,
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_API_KEY,
    CATEGORY_INVALID_MODEL,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_NOT_FOUND,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_RATE_LIMIT,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
)
from supervisor.models import (
    HEALTH_AUTH_ERROR,
    HEALTH_KEY_MISSING,
    HEALTH_NETWORK_ERROR,
    HEALTH_PROVIDER_ERROR,
    HEALTH_QUOTA_LIMIT,
    HEALTH_RATE_LIMIT,
    HEALTH_READY,
)

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"

TEST_KEY = "sk-deepseek-testkey0000000000"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def plan_request(prompt: str = "PLAN prompt"):
    return SupervisorPlanRequest(
        step_no=1,
        attempt=1,
        risk="LOW",
        requires_human=False,
        prompt=prompt,
        context={},
        schema=supervisor_plan_schema(),
    )


def review_request(prompt: str = "REVIEW prompt"):
    return SupervisorReviewRequest(
        step_no=1,
        attempt=1,
        risk="LOW",
        requires_human=False,
        prompt=prompt,
        context={},
        schema=supervisor_review_schema(),
    )


def chat_body(content: str, *, usage=None, model: str = DEFAULT_DEEPSEEK_MODEL) -> dict:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def json_response(body, *, status_code: int = 200) -> FakeHttpResponse:
    return FakeHttpResponse(
        status_code=status_code, payload=body, text=json.dumps(body)
    )


def config(**overrides) -> ProviderConfig:
    return default_deepseek_config(**overrides)


def fake_transport_adapter(
    body,
    *,
    cfg: ProviderConfig | None = None,
    env=None,
    clock=None,
):
    """Adapter + FakeHttpTransport for success-path tests."""
    transport = FakeHttpTransport([json_response(body)])
    adapter = DeepSeekSupervisorAdapter(
        cfg or config(),
        transport,
        env={"DEEPSEEK_API_KEY": TEST_KEY} if env is None else env,
        clock=clock,
    )
    return adapter, transport


def mock_transport_adapter(
    handler,
    *,
    cfg: ProviderConfig | None = None,
    env=None,
):
    """Adapter + REAL HttpTransport over httpx.MockTransport (status mapping)."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(client=client)
    adapter = DeepSeekSupervisorAdapter(
        cfg or config(),
        transport,
        env={"DEEPSEEK_API_KEY": TEST_KEY} if env is None else env,
    )
    return adapter, transport


def raising_handler(exc_factory):
    def handler(request):
        raise exc_factory()

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


# --------------------------------------------------------------------------- #
# Port conformance + PLAN/REVIEW success paths
# --------------------------------------------------------------------------- #


def test_adapter_satisfies_supervisor_port():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload()))
    )
    assert isinstance(adapter, SupervisorPort)
    assert isinstance(transport, TransportPort)
    assert adapter.provider_id == PROVIDER_DEEPSEEK


def test_plan_success_normalized():
    expected = fake_plan_payload(goal="ship it")
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(expected)))

    result = adapter.plan(plan_request())

    assert result.provider == PROVIDER_DEEPSEEK
    assert result.model == DEFAULT_DEEPSEEK_MODEL
    assert result.payload == expected
    assert "step_no" not in result.payload
    assert result.fallback_from is None and result.fallback_reason is None


def test_plan_fenced_json_is_extracted():
    fenced = "```json\n" + json.dumps(fake_plan_payload()) + "\n```"
    adapter, _ = fake_transport_adapter(chat_body(fenced))
    assert adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"


def test_plan_surrounding_whitespace_is_tolerated():
    content = "\n\n   " + json.dumps(fake_plan_payload()) + "  \n"
    adapter, _ = fake_transport_adapter(chat_body(content))
    assert adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"


@pytest.mark.parametrize(
    "verdict", ["APPROVE", "REVISE", "BLOCKED", "ARCHITECTURE_DECISION_REQUIRED"]
)
def test_review_verdicts_normalized(verdict):
    expected = fake_review_payload(verdict=verdict)
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(expected)))

    result = adapter.review(review_request())

    assert result.payload == expected
    assert result.payload["verdict"] == verdict
    assert result.provider == PROVIDER_DEEPSEEK
    assert "step_no" not in result.payload


# --------------------------------------------------------------------------- #
# Request shape / endpoint / model / api_base
# --------------------------------------------------------------------------- #


def test_request_shape_and_endpoint():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())),
        cfg=config(max_output_tokens=512, temperature=0.0),
    )

    adapter.plan(plan_request(prompt="DO THE THING"))

    recorded = transport.requests[-1]
    assert recorded["method"] == "POST"
    assert recorded["url"] == deepseek_endpoint(
        DEFAULT_DEEPSEEK_API_BASE, CHAT_COMPLETIONS_PATH
    )
    assert recorded["headers"]["Authorization"] == f"Bearer {TEST_KEY}"
    body = recorded["json"]
    assert body["model"] == DEFAULT_DEEPSEEK_MODEL
    assert body["stream"] is False
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "DO THE THING"
    # the system prompt states the identity-ownership rule
    assert "step_no" in body["messages"][0]["content"]
    assert recorded["timeout_sec"] == default_deepseek_config().timeout_sec


def test_model_is_configurable():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())), cfg=config(model="deepseek-reasoner")
    )
    assert adapter.model == "deepseek-reasoner"
    assert adapter.plan(plan_request()).model == "deepseek-reasoner"
    assert transport.requests[-1]["json"]["model"] == "deepseek-reasoner"


def test_api_base_is_configurable_and_not_hardcoded():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())),
        cfg=config(api_base="https://deepseek.internal/proxy/"),
    )
    adapter.plan(plan_request())
    assert (
        transport.requests[-1]["url"]
        == "https://deepseek.internal/proxy/chat/completions"
    )


def test_default_config_is_centralized():
    cfg = default_deepseek_config()
    assert cfg.provider_id == PROVIDER_DEEPSEEK
    assert cfg.model == DEFAULT_DEEPSEEK_MODEL
    assert cfg.api_base == DEFAULT_DEEPSEEK_API_BASE
    assert cfg.api_key_source == f"env:{DEEPSEEK_API_KEY_ENV}"
    assert cfg.supports_structured_output is True


# --------------------------------------------------------------------------- #
# Local preconditions: API key and model (no request attempted)
# --------------------------------------------------------------------------- #


def test_missing_api_key_raises_invalid_api_key_without_a_request():
    transport = FakeHttpTransport()
    adapter = DeepSeekSupervisorAdapter(config(), transport, env={})

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_API_KEY
    assert info.value.retryable is False
    assert info.value.operation == "PLAN"
    assert DEEPSEEK_API_KEY_ENV in info.value.sanitized_message
    assert transport.requests == []  # never called


def test_blank_api_key_is_treated_as_missing():
    transport = FakeHttpTransport()
    adapter = DeepSeekSupervisorAdapter(
        config(), transport, env={DEEPSEEK_API_KEY_ENV: "   "}
    )

    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())

    assert info.value.category == CATEGORY_INVALID_API_KEY
    assert transport.requests == []


def test_empty_model_fails_before_network():
    transport = FakeHttpTransport()
    adapter = DeepSeekSupervisorAdapter(
        config(model=""), transport, env={DEEPSEEK_API_KEY_ENV: TEST_KEY}
    )

    assert adapter.model is None
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False
    assert transport.requests == []


# --------------------------------------------------------------------------- #
# Malformed / untrusted provider output fails closed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "content", ["", "   ", "I cannot help with that.", "```\nnot json\n```", "[1, 2, 3]"]
)
def test_non_json_content_is_contract_error(content):
    adapter, _ = fake_transport_adapter(chat_body(content))
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False


def test_missing_content_is_contract_error():
    adapter, _ = fake_transport_adapter(
        {"choices": [{"message": {"role": "assistant"}}]}
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT


def test_schema_invalid_payload_is_contract_error():
    bad = json.dumps({"decision": "PLAN_READY"})  # missing required keys
    adapter, _ = fake_transport_adapter(chat_body(bad))
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False


def test_invalid_review_verdict_is_contract_error():
    bad = json.dumps({**fake_review_payload(), "verdict": "SHIP_IT"})
    adapter, _ = fake_transport_adapter(chat_body(bad))
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_CONTRACT


@pytest.mark.parametrize("field", ["step_no", "attempt", "state"])
def test_identity_fields_from_provider_are_rejected(field):
    payload = {**fake_plan_payload(), field: 999}
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(payload)))

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_CONTRACT
    assert field in info.value.sanitized_message


# --------------------------------------------------------------------------- #
# Usage / latency telemetry
# --------------------------------------------------------------------------- #


def test_usage_extracted_when_supplied():
    body = chat_body(
        json.dumps(fake_plan_payload()),
        usage={"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    )
    adapter, _ = fake_transport_adapter(body, clock=ticking_clock(10.0, 10.25))

    usage = adapter.plan(plan_request()).usage

    assert usage.input_tokens == 120
    assert usage.output_tokens == 45
    assert usage.total_tokens == 165
    assert usage.latency_ms == 250


def test_unknown_usage_stays_null_and_cost_is_never_guessed():
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(fake_plan_payload())))

    usage = adapter.plan(plan_request()).usage

    assert usage is not None
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.estimated_cost is None


# --------------------------------------------------------------------------- #
# HTTP / transport failure classification
# --------------------------------------------------------------------------- #


def test_http_401_is_auth_non_retryable():
    handler = lambda request: httpx.Response(
        401,
        json={"error": {"message": "Authentication Fails", "code": "invalid_api_key"}},
    )
    adapter, _ = mock_transport_adapter(handler)

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_AUTH
    assert info.value.retryable is False
    assert info.value.http_status == 401
    assert info.value.operation == "PLAN"


def test_http_402_is_payment_required():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            402, json={"error": {"message": "Insufficient Balance"}}
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_PAYMENT_REQUIRED
    assert info.value.retryable is False


def test_http_429_is_rate_limit_retryable():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(429, json={"error": {"message": "Rate limit reached"}})
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_RATE_LIMIT
    assert info.value.retryable is True


def test_http_503_is_provider_unavailable():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            503, json={"error": {"message": "Service Unavailable"}}
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_PROVIDER_UNAVAILABLE
    assert info.value.retryable is True


def test_http_400_is_invalid_request():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(400, json={"error": {"message": "bad params"}})
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False


def test_http_404_generic_stays_not_found():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(404, json={"error": {"message": "no such route"}})
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_NOT_FOUND
    assert info.value.category != CATEGORY_INVALID_MODEL


def test_http_404_model_message_is_refined_to_invalid_model():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            404, json={"error": {"message": "Model Not Exist", "code": "model_not_found"}}
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_INVALID_MODEL
    assert info.value.retryable is False
    assert info.value.http_status == 404


def test_timeout_classification():
    adapter, _ = mock_transport_adapter(
        raising_handler(lambda: httpx.ConnectTimeout("connect timed out"))
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_TIMEOUT
    assert info.value.retryable is True


def test_connect_error_classification():
    adapter, _ = mock_transport_adapter(
        raising_handler(lambda: httpx.ConnectError("refused"))
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_NETWORK
    assert info.value.retryable is True


def test_transport_error_classification():
    adapter, _ = mock_transport_adapter(
        raising_handler(lambda: httpx.ReadError("read failed"))
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_TRANSPORT
    assert info.value.operation == "REVIEW"


def test_refined_error_keeps_the_original_as_cause():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(404, json={"error": {"message": "Model Not Exist"}})
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    cause = info.value.__cause__
    assert isinstance(cause, SupervisorRunError)
    assert cause.category == CATEGORY_NOT_FOUND
    assert cause is not info.value  # never a self-referential chain


# --------------------------------------------------------------------------- #
# Health checks (cheap; never a completion request)
# --------------------------------------------------------------------------- #


def test_health_key_missing_without_a_request():
    transport = FakeHttpTransport()
    adapter = DeepSeekSupervisorAdapter(config(), transport, env={})

    health = adapter.health_check()

    assert health.status == HEALTH_KEY_MISSING
    assert health.ready is False
    assert health.provider_id == PROVIDER_DEEPSEEK
    assert DEEPSEEK_API_KEY_ENV in health.detail
    assert transport.requests == []


def test_health_ready_via_models_endpoint():
    adapter, transport = fake_transport_adapter(
        {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]}
    )

    health = adapter.health_check()

    assert health.status == HEALTH_READY
    assert health.ready is True
    assert health.detail == "2 models available"
    recorded = transport.requests[-1]
    assert recorded["method"] == "GET"
    assert recorded["url"] == deepseek_endpoint(DEFAULT_DEEPSEEK_API_BASE, MODELS_PATH)
    assert recorded["headers"]["Authorization"] == f"Bearer {TEST_KEY}"


@pytest.mark.parametrize(
    "handler,expected",
    [
        (
            lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}),
            HEALTH_AUTH_ERROR,
        ),
        (
            lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}),
            HEALTH_RATE_LIMIT,
        ),
        (
            lambda r: httpx.Response(402, json={"error": {"message": "no balance"}}),
            HEALTH_QUOTA_LIMIT,
        ),
        (lambda r: httpx.Response(503, text="down"), HEALTH_PROVIDER_ERROR),
        (lambda r: httpx.Response(500, text="boom"), HEALTH_PROVIDER_ERROR),
    ],
)
def test_health_error_statuses(handler, expected):
    adapter, _ = mock_transport_adapter(handler)
    assert adapter.health_check().status == expected


def test_health_network_error_status():
    adapter, _ = mock_transport_adapter(
        raising_handler(lambda: httpx.ConnectError("refused"))
    )
    assert adapter.health_check().status == HEALTH_NETWORK_ERROR


def test_health_never_runs_a_completion():
    seen: list[str] = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(200, json={"data": []})

    adapter, _ = mock_transport_adapter(handler)
    adapter.health_check()

    assert seen == ["GET"]  # never POST /chat/completions


# --------------------------------------------------------------------------- #
# Secret safety
# --------------------------------------------------------------------------- #


def test_key_never_leaks_into_diagnostics():
    secret = "sk-deepseek-LIVESECRET1234567890"

    def handler(request):
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": f"Authentication Fails, your key {secret} is invalid",
                    "code": "invalid_api_key",
                }
            },
        )

    adapter, _ = mock_transport_adapter(handler, env={DEEPSEEK_API_KEY_ENV: secret})

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    exc = info.value
    assert secret not in str(exc)
    assert secret not in exc.sanitized_message
    assert secret not in json.dumps(exc.to_dict())
    assert "***" in exc.sanitized_message

    # the health probe must not leak it either
    assert secret not in adapter.health_check().detail


def test_config_stores_only_a_reference_not_the_secret():
    adapter, _ = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())),
        env={DEEPSEEK_API_KEY_ENV: TEST_KEY},
    )
    assert adapter.config.api_key_source == f"env:{DEEPSEEK_API_KEY_ENV}"
    assert TEST_KEY not in repr(adapter.config)
    assert not hasattr(adapter, "api_key")  # only the source, never the value


def test_successful_result_never_contains_the_key():
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(fake_plan_payload())))

    result = adapter.plan(plan_request())

    assert TEST_KEY not in json.dumps(result.payload)
    assert TEST_KEY not in json.dumps(result.usage.__dict__)


# --------------------------------------------------------------------------- #
# No live network
# --------------------------------------------------------------------------- #


def test_no_live_network_calls(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", boom)

    fake_adapter, _ = fake_transport_adapter(chat_body(json.dumps(fake_plan_payload())))
    assert fake_adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"

    mock_adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(200, json=chat_body(json.dumps(fake_review_payload())))
    )
    assert mock_adapter.review(review_request()).payload["verdict"] == "APPROVE"


# --------------------------------------------------------------------------- #
# M5 must not change any routing default
# --------------------------------------------------------------------------- #


def test_deepseek_is_not_the_default_provider():
    policy = SupervisorPolicy()
    assert policy.plan_provider == DEFAULT_LEGACY_PROVIDER
    assert policy.review_provider == DEFAULT_LEGACY_PROVIDER
    assert PROVIDER_DEEPSEEK != DEFAULT_LEGACY_PROVIDER
    assert PROVIDER_DEEPSEEK not in policy.fallback_chain


def test_controller_still_routes_only_codex_legacy(tmp_path):
    spec = importlib.util.spec_from_file_location("worker_tandem_m5", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_m5"] = module
    spec.loader.exec_module(module)

    controller = module.TandemController(tmp_path)
    try:
        assert controller.supervisor_router.registered_providers() == (
            DEFAULT_LEGACY_PROVIDER,
        )
        assert controller.supervisor_policy.plan_provider == DEFAULT_LEGACY_PROVIDER
        assert controller.supervisor_policy.review_provider == DEFAULT_LEGACY_PROVIDER
    finally:
        controller.db.close()





