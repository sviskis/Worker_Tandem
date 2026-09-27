"""Worker Tandem supervisor foundation tests (milestones M1-M2).

These tests pin the provider-neutral supervisor abstraction:

* the canonical PLAN/REVIEW schema contract (and controller-owned identity);
* the fail-closed error taxonomy;
* the httpx transport's timeout/network/transport/HTTP classification, its
  body-size cap and its secret redaction;
* the deterministic, offline fake adapter / fake transport.

No test performs a live or paid API call: the transport is always driven by
``httpx.MockTransport`` or by ``FakeHttpTransport``, and ``socket`` is
neutralized where a no-network proof is claimed.
"""

from __future__ import annotations

import json
import socket

import httpx
import pytest

from supervisor import (
    ALL_CATEGORIES,
    NON_RETRYABLE_CATEGORIES,
    RETRYABLE_CATEGORIES,
    FakeHttpResponse,
    FakeHttpTransport,
    FakeSupervisorAdapter,
    HttpTransport,
    ProviderHealth,
    SupervisorPort,
    SupervisorPlanRequest,
    SupervisorReviewRequest,
    SupervisorRunError,
    TransportPort,
    controller_owned_identity_problems,
    fake_plan_payload,
    fake_review_payload,
    normalize_plan_payload,
    normalize_review_payload,
    sanitize_diagnostic,
    supervisor_plan_schema,
    supervisor_review_schema,
    validate_plan_payload,
    validate_review_payload,
)
from supervisor.errors import (
    CATEGORY_AUTH,
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_RATE_LIMIT,
    CATEGORY_RESPONSE_TOO_LARGE,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
)
from supervisor.models import HEALTH_KEY_MISSING, HEALTH_READY

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def plan_request(step_no: int = 1, attempt: int = 1, risk: str = "LOW"):
    return SupervisorPlanRequest(
        step_no=step_no,
        attempt=attempt,
        risk=risk,
        requires_human=False,
        prompt="plan this step",
        context={},
        schema=supervisor_plan_schema(),
    )


def review_request(step_no: int = 1, attempt: int = 1, risk: str = "LOW"):
    return SupervisorReviewRequest(
        step_no=step_no,
        attempt=attempt,
        risk=risk,
        requires_human=False,
        prompt="review this step",
        context={},
        schema=supervisor_review_schema(),
    )


def mock_client(handler) -> httpx.Client:
    """An httpx client whose transport is a MockTransport (no sockets)."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def transport_with(handler, **kwargs) -> HttpTransport:
    return HttpTransport(client=mock_client(handler), **kwargs)


def raising_handler(exc_factory):
    def handler(request):
        raise exc_factory()

    return handler


# --------------------------------------------------------------------------- #
# 1-5  FakeSupervisorAdapter scenarios
# --------------------------------------------------------------------------- #


def test_fake_adapter_is_supervisor_port_and_plans():
    fake = FakeSupervisorAdapter.healthy()
    assert isinstance(fake, SupervisorPort)
    result = fake.plan(plan_request())
    assert result.provider == "fake"
    assert result.payload["decision"] == "PLAN_READY"
    assert validate_plan_payload(result.payload) == []
    assert "step_no" not in result.payload


def test_fake_review_approve():
    fake = FakeSupervisorAdapter.with_review_verdict("APPROVE")
    result = fake.review(review_request())
    assert result.payload["verdict"] == "APPROVE"
    assert validate_review_payload(result.payload) == []


def test_fake_review_revise():
    fake = FakeSupervisorAdapter.with_review_verdict("REVISE")
    assert fake.review(review_request()).payload["verdict"] == "REVISE"


def test_fake_review_blocked():
    fake = FakeSupervisorAdapter.with_review_verdict("BLOCKED")
    assert fake.review(review_request()).payload["verdict"] == "BLOCKED"


def test_fake_retryable_error():
    fake = FakeSupervisorAdapter.with_retryable_plan_error(category=CATEGORY_RATE_LIMIT)
    with pytest.raises(SupervisorRunError) as info:
        fake.plan(plan_request())
    assert info.value.category == CATEGORY_RATE_LIMIT
    assert info.value.retryable is True


def test_fake_non_retryable_error():
    fake = FakeSupervisorAdapter.with_non_retryable_plan_error(category=CATEGORY_AUTH)
    with pytest.raises(SupervisorRunError) as info:
        fake.plan(plan_request())
    assert info.value.category == CATEGORY_AUTH
    assert info.value.retryable is False


def test_fake_malformed_plan_is_rejected_by_validation():
    fake = FakeSupervisorAdapter.with_malformed_plan()
    result = fake.plan(plan_request())
    assert validate_plan_payload(result.payload)  # non-empty problems


def test_fake_unknown_category_error_fails_closed():
    fake = FakeSupervisorAdapter.with_unknown_category_error()
    with pytest.raises(SupervisorRunError) as info:
        fake.review(review_request())
    assert info.value.retryable is False


# --------------------------------------------------------------------------- #
# 6-9  Canonical schema validation
# --------------------------------------------------------------------------- #


def test_plan_validation_success():
    assert validate_plan_payload(fake_plan_payload()) == []
    assert validate_plan_payload(fake_plan_payload(decision="BLOCKED")) == []


def test_plan_validation_failure():
    missing = fake_plan_payload()
    del missing["summary"]
    assert any("summary" in p for p in validate_plan_payload(missing))

    wrong_type = fake_plan_payload()
    wrong_type["files"] = "a.py"  # must be an array
    assert any("files" in p for p in validate_plan_payload(wrong_type))

    bad_decision = fake_plan_payload()
    bad_decision["decision"] = "MAYBE"
    assert any("decision" in p for p in validate_plan_payload(bad_decision))


def test_review_validation_success():
    assert validate_review_payload(fake_review_payload()) == []
    for verdict in (
        "APPROVE",
        "REVISE",
        "BLOCKED",
        "ARCHITECTURE_DECISION_REQUIRED",
    ):
        assert validate_review_payload(fake_review_payload(verdict=verdict)) == []


def test_invalid_verdict_rejected():
    payload = fake_review_payload()
    payload["verdict"] = "SHIP_IT"
    problems = validate_review_payload(payload)
    assert any("verdict" in p for p in problems)


def test_malformed_payload_raises_contract_error():
    with pytest.raises(SupervisorRunError) as info:
        normalize_plan_payload({"decision": "PLAN_READY"}, provider="fake")
    assert info.value.category == CATEGORY_CONTRACT
    assert info.value.retryable is False

    with pytest.raises(SupervisorRunError) as info:
        normalize_review_payload({"verdict": "APPROVE"}, provider="fake")
    assert info.value.category == CATEGORY_CONTRACT


def test_normalize_returns_valid_payload_unchanged():
    assert normalize_plan_payload(fake_plan_payload())["decision"] == "PLAN_READY"
    assert normalize_review_payload(fake_review_payload())["verdict"] == "APPROVE"


# --------------------------------------------------------------------------- #
# 10  Controller-owned identity is never provider authority
# --------------------------------------------------------------------------- #


def test_provider_identity_is_not_authoritative():
    for schema in (supervisor_plan_schema(), supervisor_review_schema()):
        assert "step_no" not in schema["properties"]
        assert "step_no" not in schema["required"]
        assert "attempt" not in schema["properties"]

    payload = fake_plan_payload()
    payload["step_no"] = 999
    problems = validate_plan_payload(payload)
    assert problems
    assert any("step_no" in p for p in problems)
    assert controller_owned_identity_problems(payload)

    review_payload = fake_review_payload()
    review_payload["attempt"] = 7
    assert controller_owned_identity_problems(review_payload)
    assert validate_review_payload(review_payload)


# --------------------------------------------------------------------------- #
# Error taxonomy
# --------------------------------------------------------------------------- #


def test_taxonomy_retryable_and_non_retryable_categories():
    assert RETRYABLE_CATEGORIES and NON_RETRYABLE_CATEGORIES
    assert not (RETRYABLE_CATEGORIES & NON_RETRYABLE_CATEGORIES)
    assert ALL_CATEGORIES == RETRYABLE_CATEGORIES | NON_RETRYABLE_CATEGORIES
    for category in RETRYABLE_CATEGORIES:
        assert SupervisorRunError("x", category=category).retryable is True
    for category in NON_RETRYABLE_CATEGORIES:
        assert SupervisorRunError("x", category=category).retryable is False


def test_unknown_category_fails_closed():
    err = SupervisorRunError("something odd", category="unknown")
    assert err.retryable is False
    assert err.category == "unknown"


# --------------------------------------------------------------------------- #
# 11-18  Transport failure classification
# --------------------------------------------------------------------------- #


def test_transport_response_and_port():
    transport = HttpTransport(client=mock_client(lambda r: httpx.Response(200, json={"ok": 1})))
    assert isinstance(transport, TransportPort)
    response = transport.post_json("https://api.example/v1", json_body={"a": 1})
    assert response.status_code == 200
    assert response.payload == {"ok": 1}
    transport.close()


def test_timeout_classification():
    transport = transport_with(raising_handler(lambda: httpx.ConnectTimeout("connect timed out")))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_TIMEOUT
    assert info.value.retryable is True


def test_read_timeout_classification():
    transport = transport_with(raising_handler(lambda: httpx.ReadTimeout("read timed out")))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_TIMEOUT
    assert info.value.retryable is True


def test_connect_error_classification():
    transport = transport_with(raising_handler(lambda: httpx.ConnectError("connection refused")))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_NETWORK
    assert info.value.retryable is True


def test_transport_error_classification():
    transport = transport_with(raising_handler(lambda: httpx.ReadError("read failed")))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_TRANSPORT
    assert info.value.retryable is True


def test_http_401_is_auth_non_retryable():
    handler = lambda r: httpx.Response(401, json={"error": {"message": "bad key", "code": "invalid_api_key"}})
    transport = transport_with(handler)
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_AUTH
    assert info.value.retryable is False
    assert info.value.http_status == 401
    assert info.value.provider_error_code == "invalid_api_key"


def test_http_402_is_payment_required():
    transport = transport_with(lambda r: httpx.Response(402, json={"error": "billing disabled"}))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_PAYMENT_REQUIRED
    assert info.value.retryable is False


def test_http_429_is_rate_limit_retryable():
    transport = transport_with(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_RATE_LIMIT
    assert info.value.retryable is True
    assert info.value.http_status == 429


def test_http_503_is_provider_unavailable():
    transport = transport_with(lambda r: httpx.Response(503, text="upstream down"))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_PROVIDER_UNAVAILABLE
    assert info.value.retryable is True


def test_http_400_is_invalid_request():
    transport = transport_with(lambda r: httpx.Response(400, json={"error": {"message": "bad params"}}))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False


def test_http_404_is_not_globally_invalid_model():
    transport = transport_with(lambda r: httpx.Response(404, json={"error": {"message": "no route"}}))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})
    assert info.value.category == "not_found"
    assert info.value.category != "invalid_model"


def test_invalid_url_is_invalid_request():
    transport = HttpTransport(client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("not-a-valid-url", json_body={})
    assert info.value.category == CATEGORY_INVALID_REQUEST
    assert info.value.retryable is False


# --------------------------------------------------------------------------- #
# 19  Response size safety (abort before unbounded buffering)
# --------------------------------------------------------------------------- #


class _CountingStream(httpx.SyncByteStream):
    """A byte stream that records how many chunks were actually pulled."""

    def __init__(self, chunk: bytes, count: int) -> None:
        self.chunk = chunk
        self.count = count
        self.pulled = 0

    def __iter__(self):
        for _ in range(self.count):
            self.pulled += 1
            yield self.chunk


def test_oversized_response_rejected_and_read_aborted_early():
    chunk_size = 65536
    stream = _CountingStream(b"x" * chunk_size, 100)  # 6.4 MB total

    def handler(request):
        return httpx.Response(200, stream=stream)

    transport = HttpTransport(
        client=mock_client(handler), max_response_bytes=chunk_size
    )
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://api.example/v1", json_body={})

    assert info.value.category == CATEGORY_RESPONSE_TOO_LARGE
    assert info.value.retryable is False
    # The read was aborted the moment the cap was exceeded: the full body was
    # never pulled into memory.
    assert stream.pulled < 100
    assert stream.pulled <= 3


# --------------------------------------------------------------------------- #
# 20  Secret redaction
# --------------------------------------------------------------------------- #


def test_secrets_absent_from_raised_and_loggable_diagnostics():
    secret = "sk-live-SUPERSECRET1234567890"

    def handler(request):
        # The header really is on the wire...
        assert "authorization" in {k.lower() for k in request.headers.keys()}
        return httpx.Response(
            401,
            json={"error": {"message": f"Invalid API key {secret}", "code": "invalid_api_key"}},
        )

    transport = transport_with(handler)
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json(
            "https://api.example/v1",
            headers={"Authorization": f"Bearer {secret}"},
            json_body={"api_key": secret},
        )
    exc = info.value
    assert secret not in str(exc)
    assert secret not in exc.sanitized_message
    assert secret not in json.dumps(exc.to_dict())
    assert "***" in exc.sanitized_message


def test_sanitize_diagnostic_redacts_known_secret_shapes():
    assert "sk-abcDEFGH12345678" not in sanitize_diagnostic("key sk-abcDEFGH12345678 here")
    assert "sk-ant-abcDEFGH12345678" not in sanitize_diagnostic("k sk-ant-abcDEFGH12345678")
    assert "abcDEF123456" not in sanitize_diagnostic("Authorization: Bearer abcDEF123456")
    assert "xyz123456" not in sanitize_diagnostic('api_key="xyz123456"')
    assert sanitize_diagnostic("plain message") == "plain message"


# --------------------------------------------------------------------------- #
# Shared client + injectable transport seam
# --------------------------------------------------------------------------- #


def test_transport_uses_one_shared_client(monkeypatch):
    client = mock_client(lambda r: httpx.Response(200, json={"ok": True}))
    transport = HttpTransport(client=client)
    constructed = {"n": 0}

    def explode(*args, **kwargs):
        constructed["n"] += 1
        raise AssertionError("a new httpx.Client was constructed per request")

    monkeypatch.setattr(httpx, "Client", explode)
    for _ in range(2):
        response = transport.post_json("https://api.example/v1", json_body={})
        assert response.status_code == 200
    assert constructed["n"] == 0


def test_fake_http_transport_satisfies_transport_port_and_records():
    transport = FakeHttpTransport([FakeHttpResponse(200, {"ok": True}, "{}")])
    assert isinstance(transport, TransportPort)
    response = transport.post_json(
        "https://x/v1", headers={"H": "1"}, json_body={"a": 1}
    )
    assert response.status_code == 200
    assert response.payload == {"ok": True}
    assert transport.requests[-1]["json"] == {"a": 1}
    assert transport.requests[-1]["headers"] == {"H": "1"}
    assert transport.get_json("https://x/v1").status_code == 200


def test_fake_http_transport_can_raise_scripted_error():
    error = SupervisorRunError("down", category=CATEGORY_PROVIDER_UNAVAILABLE)
    transport = FakeHttpTransport([FakeHttpResponse(error=error)])
    with pytest.raises(SupervisorRunError) as info:
        transport.post_json("https://x/v1", json_body={})
    assert info.value.retryable is True


# --------------------------------------------------------------------------- #
# 21-22  Provider health
# --------------------------------------------------------------------------- #


def test_fake_adapter_health_ready():
    health = FakeSupervisorAdapter.healthy().health_check()
    assert health.status == HEALTH_READY
    assert health.ready is True


def test_fake_adapter_health_key_missing_like():
    fake = FakeSupervisorAdapter.healthy(
        health=ProviderHealth(
            provider_id="openai",
            status=HEALTH_KEY_MISSING,
            detail="OPENAI_API_KEY missing",
        )
    )
    health = fake.health_check()
    assert health.status == HEALTH_KEY_MISSING
    assert health.ready is False


# --------------------------------------------------------------------------- #
# 23  No live network calls
# --------------------------------------------------------------------------- #


def test_no_live_network_calls(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", boom)

    transport = HttpTransport(
        client=mock_client(lambda r: httpx.Response(200, json={"ok": True}))
    )
    assert transport.post_json("https://api.example/v1", json_body={}).payload == {
        "ok": True
    }

    fake = FakeHttpTransport()  # never touches a socket either
    assert fake.post_json("https://x/v1", json_body={}).status_code == 200




