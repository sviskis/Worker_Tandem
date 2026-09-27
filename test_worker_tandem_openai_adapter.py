"""OpenAI supervisor adapter tests (milestone M6).

Mirrors the M3/M5 approach: the adapter is driven entirely through injected
transport doubles, so no test ever opens a socket or calls a paid API.

Two harnesses, exactly as in M1/M2:

* ``FakeHttpTransport`` - success paths, request-shape assertions, no-network;
* ``HttpTransport`` + ``httpx.MockTransport`` - because only the REAL transport
  maps non-2xx statuses and httpx exceptions into ``SupervisorRunError``.

The API contract under test is the documented one: chat completions with
``response_format={"type":"json_object"}`` + ``GET /models`` health probe.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import socket
import subprocess
import sys

import httpx
import pytest

from supervisor import (
    CHAT_COMPLETIONS_PATH,
    DEFAULT_LEGACY_PROVIDER,
    DEFAULT_OPENAI_API_BASE,
    DEFAULT_OPENAI_MODEL,
    DEEPSEEK_API_KEY_ENV,
    OPENAI_API_KEY_ENV,
    PROVIDER_OPENAI,
    DeepSeekSupervisorAdapter,
    FakeHttpResponse,
    FakeHttpTransport,
    HttpTransport,
    LegacyCodexSupervisorAdapter,
    OpenAISupervisorAdapter,
    ProviderConfig,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorPort,
    SupervisorReviewRequest,
    SupervisorRunError,
    TransportPort,
    classify_openai_error,
    default_deepseek_config,
    default_openai_config,
    fake_plan_payload,
    fake_review_payload,
    openai_endpoint,
    strict_json_object,
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
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_RESPONSE_TOO_LARGE,
    CATEGORY_TEMPORARY_5XX,
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

TEST_KEY = "sk-openai-testkey0000000000"

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


def chat_body(content: str, *, usage=None, model: str = DEFAULT_OPENAI_MODEL) -> dict:
    """OpenAI chat-completions envelope (identical in shape to DeepSeek's)."""
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
    return default_openai_config(**overrides)


def fake_transport_adapter(
    body, *, cfg: ProviderConfig | None = None, env=None, clock=None, transport=None
):
    """Adapter + FakeHttpTransport for success-path tests."""
    transport = transport or FakeHttpTransport([json_response(body)])
    adapter = OpenAISupervisorAdapter(
        cfg or config(),
        transport,
        env={"OPENAI_API_KEY": TEST_KEY} if env is None else env,
        clock=clock,
    )
    return adapter, transport


def mock_transport_adapter(handler, *, cfg: ProviderConfig | None = None, env=None):
    """Adapter + REAL HttpTransport over httpx.MockTransport (status mapping)."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = HttpTransport(client=client)
    adapter = OpenAISupervisorAdapter(
        cfg or config(),
        transport,
        env={"OPENAI_API_KEY": TEST_KEY} if env is None else env,
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


class _CountingStream(httpx.SyncByteStream):
    """Records how many chunks were actually pulled from the wire."""

    def __init__(self, chunk: bytes, count: int) -> None:
        self.chunk = chunk
        self.count = count
        self.pulled = 0

    def __iter__(self):
        for _ in range(self.count):
            self.pulled += 1
            yield self.chunk


class _StubCodexRunner:
    """Minimal StructuredRunnerPort double for the Codex Legacy parity check."""

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


# --------------------------------------------------------------------------- #
# 1-5  PLAN / REVIEW success + port conformance
# --------------------------------------------------------------------------- #


def test_adapter_satisfies_supervisor_port():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload()))
    )
    assert isinstance(adapter, SupervisorPort)
    assert isinstance(transport, TransportPort)
    assert adapter.provider_id == PROVIDER_OPENAI


def test_plan_success_normalized():
    expected = fake_plan_payload(goal="ship it")
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(expected)))

    result = adapter.plan(plan_request())

    assert result.provider == PROVIDER_OPENAI
    assert result.model == DEFAULT_OPENAI_MODEL
    assert result.payload == expected
    assert "step_no" not in result.payload
    assert result.fallback_from is None and result.fallback_reason is None


@pytest.mark.parametrize(
    "verdict", ["APPROVE", "REVISE", "BLOCKED", "ARCHITECTURE_DECISION_REQUIRED"]
)
def test_review_verdicts_normalized(verdict):
    expected = fake_review_payload(verdict=verdict)
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(expected)))

    result = adapter.review(review_request())

    assert result.payload == expected
    assert result.provider == PROVIDER_OPENAI
    assert "step_no" not in result.payload


# --------------------------------------------------------------------------- #
# 6-7  Usage
# --------------------------------------------------------------------------- #


def test_usage_extracted_when_supplied():
    body = chat_body(
        json.dumps(fake_plan_payload()),
        usage={"prompt_tokens": 321, "completion_tokens": 12, "total_tokens": 333},
    )
    adapter, _ = fake_transport_adapter(body, clock=ticking_clock(4.0, 4.5))

    usage = adapter.plan(plan_request()).usage

    assert usage.input_tokens == 321
    assert usage.output_tokens == 12
    assert usage.total_tokens == 333
    assert usage.latency_ms == 500
    assert usage.estimated_cost is None


def test_unknown_usage_stays_null():
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(fake_plan_payload())))

    usage = adapter.plan(plan_request()).usage

    assert usage is not None
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.estimated_cost is None


# --------------------------------------------------------------------------- #
# 36-38  Configured timeout / api_base / model
# --------------------------------------------------------------------------- #


def test_request_shape_and_endpoint():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())),
        cfg=config(max_output_tokens=256, temperature=0.0, timeout_sec=90),
    )

    adapter.plan(plan_request(prompt="DO THE THING"))

    recorded = transport.requests[-1]
    assert recorded["method"] == "POST"
    assert recorded["url"] == openai_endpoint(
        DEFAULT_OPENAI_API_BASE, CHAT_COMPLETIONS_PATH
    )
    assert recorded["url"] == "https://api.openai.com/v1/chat/completions"
    assert recorded["headers"]["Authorization"] == f"Bearer {TEST_KEY}"
    assert recorded["timeout_sec"] == 90
    body = recorded["json"]
    assert body["model"] == DEFAULT_OPENAI_MODEL
    assert body["stream"] is False
    assert body["max_tokens"] == 256
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "DO THE THING"
    assert "step_no" in body["messages"][0]["content"]


def test_configured_api_base_is_used():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())),
        cfg=config(api_base="https://openai.internal/gateway/v1/"),
    )
    adapter.plan(plan_request())
    assert (
        transport.requests[-1]["url"]
        == "https://openai.internal/gateway/v1/chat/completions"
    )


def test_configured_model_is_used():
    adapter, transport = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())), cfg=config(model="gpt-4o")
    )
    assert adapter.model == "gpt-4o"
    assert adapter.plan(plan_request()).model == "gpt-4o"
    assert transport.requests[-1]["json"]["model"] == "gpt-4o"


def test_default_config_is_centralized():
    cfg = default_openai_config()
    assert cfg.provider_id == PROVIDER_OPENAI
    assert cfg.display_name == "OpenAI"
    assert cfg.model == DEFAULT_OPENAI_MODEL
    assert cfg.api_base == DEFAULT_OPENAI_API_BASE
    assert cfg.api_key_source == f"env:{OPENAI_API_KEY_ENV}"
    assert cfg.timeout_sec == 120
    assert cfg.supports_structured_output is True


# --------------------------------------------------------------------------- #
# 8-10  Local preconditions (no request attempted)
# --------------------------------------------------------------------------- #


def test_missing_api_key():
    transport = FakeHttpTransport()
    adapter = OpenAISupervisorAdapter(config(), transport, env={})

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_API_KEY
    assert info.value.retryable is False
    assert OPENAI_API_KEY_ENV in info.value.sanitized_message
    assert transport.requests == []


def test_blank_api_key():
    transport = FakeHttpTransport()
    adapter = OpenAISupervisorAdapter(
        config(), transport, env={OPENAI_API_KEY_ENV: "  "}
    )

    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())

    assert info.value.category == CATEGORY_INVALID_API_KEY
    assert transport.requests == []


def test_missing_model():
    transport = FakeHttpTransport()
    adapter = OpenAISupervisorAdapter(
        config(model=""), transport, env={OPENAI_API_KEY_ENV: TEST_KEY}
    )

    assert adapter.model is None
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False
    assert transport.requests == []


# --------------------------------------------------------------------------- #
# 11-16  HTTP / transport error mapping (strongest evidence first)
# --------------------------------------------------------------------------- #


def test_auth_401():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_AUTH
    assert info.value.retryable is False
    assert info.value.http_status == 401
    assert info.value.operation == "PLAN"
    assert info.value.provider_error_code == "invalid_api_key"


def test_forbidden_403():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            403, json={"error": {"message": "Country not supported"}}
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_AUTH
    assert info.value.retryable is False
    assert info.value.http_status == 403


def test_billing_payment_402():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(402, json={"error": {"message": "Payment required"}})
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_PAYMENT_REQUIRED
    assert info.value.retryable is False


def test_hard_quota_on_429_is_not_retryable():
    """A structured ``insufficient_quota`` code is billing, even on HTTP 429."""
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            429,
            json={
                "error": {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                }
            },
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_PAYMENT_REQUIRED
    assert info.value.retryable is False


def test_rate_limit_429_without_reset_evidence():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            429,
            json={
                "error": {
                    "message": "Rate limit reached for gpt-4o-mini on requests per min",
                    "type": "requests",
                    "code": "rate_limit_exceeded",
                }
            },
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_RATE_LIMIT
    assert info.value.retryable is True


def test_quota_with_reset_is_retryable():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            429,
            json={
                "error": {
                    "message": "You exceeded your current quota; try again at "
                    "Sep 27th 2026 1:29 AM"
                }
            },
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_QUOTA_WITH_RESET
    assert info.value.retryable is True


# --------------------------------------------------------------------------- #
# 17-22  Remaining transport / HTTP mappings
# --------------------------------------------------------------------------- #


def test_provider_unavailable_503():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            503, json={"error": {"message": "The server is overloaded"}}
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_PROVIDER_UNAVAILABLE
    assert info.value.retryable is True


def test_other_5xx_is_temporary():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            505, json={"error": {"message": "HTTP version not supported"}}
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_TEMPORARY_5XX
    assert info.value.retryable is True


def test_timeout():
    adapter, _ = mock_transport_adapter(
        raising_handler(lambda: httpx.ConnectTimeout("connect timed out"))
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_TIMEOUT
    assert info.value.retryable is True


def test_network_error():
    adapter, _ = mock_transport_adapter(
        raising_handler(lambda: httpx.ConnectError("refused"))
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_NETWORK
    assert info.value.retryable is True


def test_transport_error():
    adapter, _ = mock_transport_adapter(
        raising_handler(lambda: httpx.ReadError("read failed"))
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_TRANSPORT
    assert info.value.operation == "REVIEW"


def test_generic_400_is_invalid_request():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            400,
            json={"error": {"message": "bad params", "type": "invalid_request_error"}},
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False


def test_404_generic_stays_not_found():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(404, json={"error": {"message": "no such route"}})
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_NOT_FOUND
    assert info.value.category != CATEGORY_INVALID_MODEL


def test_404_structured_model_error_becomes_invalid_model():
    adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(
            404,
            json={
                "error": {
                    "message": "The model `gpt-nope` does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_INVALID_MODEL
    assert info.value.retryable is False
    assert info.value.http_status == 404


# --------------------------------------------------------------------------- #
# 23-26  Strict JSON acceptance envelope
# --------------------------------------------------------------------------- #


def test_plain_json_accepted():
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(fake_plan_payload())))
    assert adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"


def test_surrounding_whitespace_accepted():
    content = "\n\n  " + json.dumps(fake_plan_payload()) + "  \n"
    adapter, _ = fake_transport_adapter(chat_body(content))
    assert adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"


def test_fenced_json_accepted():
    fenced = "```json\n" + json.dumps(fake_plan_payload()) + "\n```"
    adapter, _ = fake_transport_adapter(chat_body(fenced))
    assert adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"


def test_fenced_without_language_tag_accepted():
    fenced = "```\n" + json.dumps(fake_plan_payload()) + "\n```"
    adapter, _ = fake_transport_adapter(chat_body(fenced))
    assert adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"


def test_missing_message_content_is_contract_error():
    adapter, _ = fake_transport_adapter(
        {"choices": [{"message": {"role": "assistant"}}]}
    )
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT


def test_prose_and_ambiguous_content_are_rejected():
    plan = json.dumps(fake_plan_payload())
    for content in (
        "Here is the plan:\n" + plan,  # prose before
        plan + "\nHope that helps!",  # prose after
        plan + " \n " + plan,  # two concatenated objects
        "[1, 2, 3]",
        "",
        "   ",
        "I cannot help with that.",
        "```\nnot json\n```",
    ):
        adapter, _ = fake_transport_adapter(chat_body(content))
        with pytest.raises(SupervisorRunError) as info:
            adapter.plan(plan_request())
        assert info.value.category == CATEGORY_CONTRACT, content
        assert info.value.retryable is False


# --------------------------------------------------------------------------- #
# 27-29  Canonical validation + identity ownership
# --------------------------------------------------------------------------- #


def test_invalid_verdict_rejected():
    bad = json.dumps({**fake_review_payload(), "verdict": "SHIP_IT"})
    adapter, _ = fake_transport_adapter(chat_body(bad))
    with pytest.raises(SupervisorRunError) as info:
        adapter.review(review_request())
    assert info.value.category == CATEGORY_CONTRACT


def test_schema_invalid_payload_rejected():
    bad = json.dumps({"decision": "PLAN_READY"})  # missing required keys
    adapter, _ = fake_transport_adapter(chat_body(bad))
    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False


@pytest.mark.parametrize("field", ["step_no", "attempt", "state"])
def test_identity_fields_from_provider_are_rejected(field):
    payload = {**fake_plan_payload(), field: 999}
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(payload)))

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_CONTRACT
    assert field in info.value.sanitized_message


# --------------------------------------------------------------------------- #
# 30  Response size safety (transport cap, never bypassed)
# --------------------------------------------------------------------------- #


def test_oversized_response_rejected_by_transport():
    chunk_size = 65536
    stream = _CountingStream(b"x" * chunk_size, 100)  # 6.4 MB total

    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))
    )
    transport = HttpTransport(client=client, max_response_bytes=chunk_size)
    adapter = OpenAISupervisorAdapter(
        config(), transport, env={OPENAI_API_KEY_ENV: TEST_KEY}
    )

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    assert info.value.category == CATEGORY_RESPONSE_TOO_LARGE
    assert info.value.retryable is False
    assert stream.pulled <= 3  # aborted, never fully buffered


# --------------------------------------------------------------------------- #
# 31  Secret safety
# --------------------------------------------------------------------------- #


def test_key_redacted_from_diagnostics():
    secret = "sk-openai-LIVESECRET1234567890"

    def handler(request):
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": f"Incorrect API key provided: {secret}",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    adapter, _ = mock_transport_adapter(handler, env={OPENAI_API_KEY_ENV: secret})

    with pytest.raises(SupervisorRunError) as info:
        adapter.plan(plan_request())

    exc = info.value
    assert secret not in str(exc)
    assert secret not in exc.sanitized_message
    assert secret not in json.dumps(exc.to_dict())
    assert "***" in exc.sanitized_message

    health = adapter.health_check()
    assert secret not in health.detail
    assert secret not in str(health)


def test_key_is_never_stored_on_the_adapter_or_config():
    adapter, _ = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload())), env={OPENAI_API_KEY_ENV: TEST_KEY}
    )
    assert adapter.config.api_key_source == f"env:{OPENAI_API_KEY_ENV}"
    assert TEST_KEY not in repr(adapter.config)
    assert not hasattr(adapter, "api_key")
    # The value only ever appears in the outbound Authorization header.
    assert adapter._headers(TEST_KEY)["Authorization"] == f"Bearer {TEST_KEY}"


def test_successful_result_never_contains_the_key():
    adapter, _ = fake_transport_adapter(chat_body(json.dumps(fake_plan_payload())))
    result = adapter.plan(plan_request())
    assert TEST_KEY not in json.dumps(result.payload)
    assert TEST_KEY not in json.dumps(result.usage.__dict__)


# --------------------------------------------------------------------------- #
# 32-35  Health
# --------------------------------------------------------------------------- #


def test_health_ready():
    adapter, transport = fake_transport_adapter(
        {"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]}
    )

    health = adapter.health_check()

    assert health.status == HEALTH_READY
    assert health.ready is True
    assert health.provider_id == PROVIDER_OPENAI
    assert health.detail == "2 models available"
    recorded = transport.requests[-1]
    assert recorded["method"] == "GET"
    assert recorded["url"] == "https://api.openai.com/v1/models"
    assert recorded["headers"]["Authorization"] == f"Bearer {TEST_KEY}"


def test_health_key_missing():
    transport = FakeHttpTransport()
    adapter = OpenAISupervisorAdapter(config(), transport, env={})

    health = adapter.health_check()

    assert health.status == HEALTH_KEY_MISSING
    assert health.ready is False
    assert OPENAI_API_KEY_ENV in health.detail
    assert transport.requests == []


@pytest.mark.parametrize(
    "handler,expected",
    [
        (
            lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}),
            HEALTH_AUTH_ERROR,
        ),
        (
            lambda r: httpx.Response(429, json={"error": {"message": "too many"}}),
            HEALTH_RATE_LIMIT,
        ),
        (
            lambda r: httpx.Response(
                429,
                json={
                    "error": {
                        "message": "quota exceeded",
                        "code": "insufficient_quota",
                    }
                },
            ),
            HEALTH_QUOTA_LIMIT,
        ),
        (lambda r: httpx.Response(503, text="down"), HEALTH_PROVIDER_ERROR),
    ],
)
def test_health_error_statuses(handler, expected):
    adapter, _ = mock_transport_adapter(handler)
    assert adapter.health_check().status == expected


def test_health_network_error():
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

    assert seen == ["GET"]


# --------------------------------------------------------------------------- #
# 40  No live network
# --------------------------------------------------------------------------- #


def test_no_live_network_calls(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)

    fake_adapter, _ = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload()))
    )
    assert fake_adapter.plan(plan_request()).payload["decision"] == "PLAN_READY"
    assert fake_adapter.health_check().status == HEALTH_READY

    mock_adapter, _ = mock_transport_adapter(
        lambda r: httpx.Response(200, json=chat_body(json.dumps(fake_review_payload())))
    )
    assert mock_adapter.review(review_request()).payload["verdict"] == "APPROVE"


# --------------------------------------------------------------------------- #
# 41-42  No SQLite / FSM mutation
# --------------------------------------------------------------------------- #


def _load_worker_tandem(name: str):
    spec = importlib.util.spec_from_file_location(name, SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_alone_creates_no_sqlite(tmp_path):
    transport = FakeHttpTransport(
        [json_response(chat_body(json.dumps(fake_plan_payload())))]
    )
    adapter = OpenAISupervisorAdapter(
        config(), transport, env={OPENAI_API_KEY_ENV: TEST_KEY}
    )

    adapter.plan(plan_request())

    assert list(tmp_path.rglob("*.db")) == []


def test_adapter_does_not_mutate_fsm_or_sqlite(tmp_path):
    module = _load_worker_tandem("worker_tandem_m6")
    plan_def = {
        "plan_version": "m6-test-1",
        "steps": [
            {
                "step_no": 1,
                "phase": "FOUNDATION",
                "title": "ONE",
                "description": "m6 integration step",
                "risk": "LOW",
            }
        ],
    }
    controller = module.TandemController(tmp_path)
    try:
        controller.db.import_plan(plan_def)
        before = controller.db.get_step(1)

        def count(table: str) -> int:
            row = controller.db.db.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()
            return int(row["n"])

        events_before = count("events")
        plans_before = count("codex_plans")
        reviews_before = count("codex_reviews")

        transport = FakeHttpTransport(
            [
                json_response(chat_body(json.dumps(fake_plan_payload()))),
                json_response(chat_body(json.dumps(fake_review_payload()))),
            ]
        )
        adapter = OpenAISupervisorAdapter(
            config(), transport, env={OPENAI_API_KEY_ENV: TEST_KEY}
        )
        # M6: the adapter is exercised directly; the controller never calls it.
        adapter.plan(plan_request())
        adapter.review(review_request())

        after = controller.db.get_step(1)
        assert after.state == before.state
        assert after.attempt == before.attempt
        assert count("events") == events_before
        assert count("codex_plans") == plans_before
        assert count("codex_reviews") == reviews_before
        assert controller.db.latest_codex_plan(1) is None
    finally:
        controller.db.close()


# --------------------------------------------------------------------------- #
# 18  Router: OpenAI is an available CAPABILITY, never the default
# --------------------------------------------------------------------------- #


def test_openai_is_not_the_default_provider():
    policy = SupervisorPolicy()
    assert policy.plan_provider == DEFAULT_LEGACY_PROVIDER
    assert policy.review_provider == DEFAULT_LEGACY_PROVIDER
    assert PROVIDER_OPENAI != DEFAULT_LEGACY_PROVIDER
    assert PROVIDER_OPENAI not in policy.fallback_chain


def test_openai_is_selectable_without_changing_the_default(tmp_path):
    from supervisor import SupervisorRouter

    openai_adapter, _ = fake_transport_adapter(
        chat_body(json.dumps(fake_plan_payload()))
    )
    codex_adapter = LegacyCodexSupervisorAdapter(
        _StubCodexRunner(fake_plan_payload()), workdir=tmp_path / ".wt"
    )
    router = SupervisorRouter(
        {DEFAULT_LEGACY_PROVIDER: codex_adapter, PROVIDER_OPENAI: openai_adapter},
        SupervisorPolicy(
            plan_provider=PROVIDER_OPENAI, review_provider=PROVIDER_OPENAI
        ),
    )

    result = router.plan(plan_request())

    assert result.provider == PROVIDER_OPENAI
    assert set(router.registered_providers()) == {
        DEFAULT_LEGACY_PROVIDER,
        PROVIDER_OPENAI,
    }
    # Registering a capability never mutates the default policy.
    assert SupervisorPolicy().plan_provider == DEFAULT_LEGACY_PROVIDER


def test_controller_still_routes_only_codex_legacy(tmp_path):
    module = _load_worker_tandem("worker_tandem_m6_controller")
    controller = module.TandemController(tmp_path)
    try:
        assert controller.supervisor_router.registered_providers() == (
            DEFAULT_LEGACY_PROVIDER,
        )
        assert controller.supervisor_policy.plan_provider == DEFAULT_LEGACY_PROVIDER
        assert controller.supervisor_policy.review_provider == DEFAULT_LEGACY_PROVIDER
    finally:
        controller.db.close()


# --------------------------------------------------------------------------- #
# 23  CANONICAL PARITY: OpenAI == DeepSeek == Codex Legacy
# --------------------------------------------------------------------------- #


def _deepseek_adapter(body):
    transport = FakeHttpTransport([json_response(body)])
    return DeepSeekSupervisorAdapter(
        default_deepseek_config(), transport, env={DEEPSEEK_API_KEY_ENV: TEST_KEY}
    )


def _codex_adapter(payload, workdir):
    return LegacyCodexSupervisorAdapter(_StubCodexRunner(payload), workdir=workdir)


def test_parity_plan_payload_across_all_providers(tmp_path):
    fixture = fake_plan_payload(goal="parity")
    content = json.dumps(fixture)

    openai_payload = (
        fake_transport_adapter(chat_body(content))[0].plan(plan_request()).payload
    )
    deepseek_payload = _deepseek_adapter(chat_body(content)).plan(
        plan_request()
    ).payload
    codex_payload = _codex_adapter(fixture, tmp_path / ".wt").plan(
        plan_request()
    ).payload

    assert openai_payload == deepseek_payload == codex_payload == fixture
    assert "step_no" not in openai_payload
    assert "attempt" not in openai_payload


def test_parity_review_payload_across_all_providers(tmp_path):
    fixture = fake_review_payload(verdict="REVISE", summary="parity")
    content = json.dumps(fixture)

    openai_payload = (
        fake_transport_adapter(chat_body(content))[0].review(review_request()).payload
    )
    deepseek_payload = _deepseek_adapter(chat_body(content)).review(
        review_request()
    ).payload
    codex_payload = _codex_adapter(fixture, tmp_path / ".wt").review(
        review_request()
    ).payload

    assert openai_payload == deepseek_payload == codex_payload == fixture


def test_parity_results_carry_only_provider_metadata(tmp_path):
    fixture = fake_plan_payload()
    content = json.dumps(fixture)

    openai_result = fake_transport_adapter(chat_body(content))[0].plan(plan_request())
    deepseek_result = _deepseek_adapter(chat_body(content)).plan(plan_request())
    codex_result = _codex_adapter(fixture, tmp_path / ".wt").plan(plan_request())

    for result in (openai_result, deepseek_result, codex_result):
        assert result.payload == fixture
        assert result.fallback_from is None and result.fallback_reason is None
    assert openai_result.provider == PROVIDER_OPENAI
    assert deepseek_result.provider == default_deepseek_config().provider_id
    assert codex_result.provider == DEFAULT_LEGACY_PROVIDER


# --------------------------------------------------------------------------- #
# Shared strict extractor + the documented DeepSeek divergence
# --------------------------------------------------------------------------- #


def test_strict_json_object_accepts_plain_and_fenced_objects():
    assert strict_json_object('{"a": 1}') == {"a": 1}
    assert strict_json_object('\n  {"a": 1}  \n') == {"a": 1}
    assert strict_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert strict_json_object('```\n{"a": 1}\n```') == {"a": 1}


@pytest.mark.parametrize(
    "content",
    [
        'Here: {"a": 1}',
        '{"a": 1} done',
        '{"a": 1} {"b": 2}',
        "[1, 2]",
        "not json",
        "",
        None,
        123,
    ],
)
def test_strict_json_object_rejects_ambiguous_input(content):
    assert strict_json_object(content) is None


def test_deepseek_remains_lenient_in_m6():
    """M6 must not change M5 behaviour: DeepSeek still tolerates prose.

    The divergence is deliberate and documented (``TODO(M7)`` in
    ``deepseek_adapter._parse_content``); this test pins it so the tightening
    cannot happen by accident.
    """
    prose = "Here is the plan:\n" + json.dumps(fake_plan_payload())

    assert (
        _deepseek_adapter(chat_body(prose)).plan(plan_request()).payload["decision"]
        == "PLAN_READY"
    )

    openai_adapter, _ = fake_transport_adapter(chat_body(prose))
    with pytest.raises(SupervisorRunError) as info:
        openai_adapter.plan(plan_request())
    assert info.value.category == CATEGORY_CONTRACT







