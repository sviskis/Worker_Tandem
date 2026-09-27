"""Anthropic (Claude) provider adapter - M7.1 foundation.

Anthropic's **Messages API is not OpenAI-shaped**, so this module deliberately
does NOT reuse the OpenAI-compatible request/response skeleton
(``openai_compatible.OpenAICompatibleSupervisorAdapter``). It reuses only the
protocol-independent helpers named in the M6 hand-off:

* ``api_endpoint``                    - base URL + path joining
* ``MODELS_PATH``                     - the shared cheap health-probe path
* ``stamp_run_error``                 - supervisor-context error stamping
* ``error_haystack`` / ``matches_hints`` - text-evidence matching
* ``health_status_for``               - canonical category -> health status
* ``sanitize_key_detail``             - KEY_MISSING detail naming only the env var
* ``base.resolve_api_key`` / ``api_key_env_name`` / ``key_missing_run_error``
* ``base.normalize_plan_payload`` / ``normalize_review_payload`` (fail-closed
  canonical validation) and the shared models/transport ports.

Protocol differences owned HERE:

* endpoint: ``POST {api_base}/messages`` (not ``/chat/completions``)
* auth: the ``x-api-key`` header (never ``Authorization: Bearer``)
* version: the ``anthropic-version`` header (constant + configurable)
* body: ``system`` is a TOP-LEVEL string, ``messages`` carries the user turn
  only, and ``max_tokens`` is MANDATORY
* no ``response_format``: JSON is requested in the system prompt and enforced by
  this adapter's own strict text envelope
* response text lives in ``content`` (a list of typed blocks) and usage is
  ``input_tokens``/``output_tokens``
* error type names: ``authentication_error``, ``permission_error``,
  ``not_found_error``, ``rate_limit_error``, ``overloaded_error``, ``api_error``

M7.1 scope: the adapter is COMPLETE and fully offline-testable, but it is NOT
registered with the router's AUTO policy, NOT wired into ``worker_tandem``, and
NOT the default provider - runtime behaviour is unchanged.

API keys: ``ANTHROPIC_API_KEY`` is resolved at call time through the shared
``env:`` reference. The plaintext key is never stored on the adapter, cached,
persisted, logged or displayed. A missing/blank key fails PLAN/REVIEW locally
with a non-retryable ``invalid_api_key`` error and performs NO HTTP call, and
reports ``KEY_MISSING`` for health.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Mapping, Optional

from .base import (
    api_key_env_name,
    key_missing_run_error,
    normalize_plan_payload,
    normalize_review_payload,
    resolve_api_key,
)
from .errors import (
    CATEGORY_AUTH,
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_MODEL,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NOT_FOUND,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_RESPONSE_TOO_LARGE,
    CATEGORY_TEMPORARY_5XX,
    CATEGORY_UNKNOWN,
    SupervisorRunError,
)
from .models import (
    HEALTH_KEY_MISSING,
    HEALTH_READY,
    PLAN_OPERATION,
    REVIEW_OPERATION,
    ProviderConfig,
    ProviderHealth,
    ProviderUsage,
    SupervisorPlanRequest,
    SupervisorPlanResult,
    SupervisorReviewRequest,
    SupervisorReviewResult,
)
from .openai_compatible import (
    MODELS_PATH,
    api_endpoint,
    error_haystack,
    health_status_for,
    matches_hints,
    sanitize_key_detail,
    stamp_run_error,
    strict_json_object,
)
from .schemas import supervisor_plan_schema, supervisor_review_schema
from .transport import TransportPort

# --------------------------------------------------------------------------- #
# Provider identity + centralized defaults
# --------------------------------------------------------------------------- #

PROVIDER_ANTHROPIC = "anthropic"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Optional operator override for the Messages API version (env, never stored).
ANTHROPIC_API_VERSION_ENV = "ANTHROPIC_API_VERSION"

DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_ANTHROPIC_TIMEOUT_SEC = 120

#: The Messages API requires an ``anthropic-version`` header.
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"

#: ``max_tokens`` is MANDATORY for Anthropic (the OpenAI body may omit it).
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096

#: Anthropic accepts temperature in [0, 1] only.
ANTHROPIC_TEMPERATURE_RANGE = (0.0, 1.0)

#: ``POST {api_base}/messages``.
MESSAGES_PATH = "/messages"


def default_anthropic_config(**overrides: Any) -> ProviderConfig:
    """The ONE place Anthropic's default base URL, model and version live.

    ``api_key_source`` is a reference (``env:ANTHROPIC_API_KEY``), never a
    secret. ``supports_structured_output`` stays ``False``: the Messages API has
    no ``response_format`` field, so JSON is prompt-enforced by this adapter.
    """
    values: dict[str, Any] = {
        "provider_id": PROVIDER_ANTHROPIC,
        "display_name": "Claude",
        "model": DEFAULT_ANTHROPIC_MODEL,
        "enabled": True,
        "api_base": DEFAULT_ANTHROPIC_API_BASE,
        "api_key_source": f"env:{ANTHROPIC_API_KEY_ENV}",
        "timeout_sec": DEFAULT_ANTHROPIC_TIMEOUT_SEC,
        "max_output_tokens": DEFAULT_ANTHROPIC_MAX_TOKENS,
        "temperature": None,
        "supports_structured_output": False,
        "cost_policy": "premium",
        "priority": 3,
    }
    values.update(overrides)
    return ProviderConfig(**values)


def anthropic_endpoint(api_base: Optional[str], path: str) -> str:
    """``POST/GET {api_base}{path}`` with Anthropic's default base."""
    return api_endpoint(api_base, path, default_base=DEFAULT_ANTHROPIC_API_BASE)


# --------------------------------------------------------------------------- #
# Error classification (provider-specific refinement)
# --------------------------------------------------------------------------- #

#: Anthropic ``error.type`` -> canonical category.
_ANTHROPIC_TYPE_CATEGORIES = {
    "authentication_error": CATEGORY_AUTH,
    "permission_error": CATEGORY_AUTH,
    "invalid_request_error": CATEGORY_INVALID_REQUEST,
    "not_found_error": CATEGORY_NOT_FOUND,
    "rate_limit_error": CATEGORY_RATE_LIMIT,
    "overloaded_error": CATEGORY_PROVIDER_UNAVAILABLE,
    "api_error": CATEGORY_PROVIDER_UNAVAILABLE,
    "request_too_large": CATEGORY_RESPONSE_TOO_LARGE,
}

_PAYMENT_TYPES = frozenset({"billing_error", "insufficient_credit", "payment_required"})
_PAYMENT_HINTS = (
    "credit balance",
    "insufficient",
    "billing",
    "payment",
    "quota",
    "upgrade",
)
_MODEL_TYPES = frozenset({"model_not_found", "invalid_model"})
_MODEL_HINTS = ("model", "claude", "not_found_error")
_RESET_HINTS = ("try again", "retry", "resets", "reset at", "temporarily")


def classify_anthropic_error(exc: SupervisorRunError) -> Optional[str]:
    """Anthropic-specific refinement, strongest evidence first.

    Returns ``None`` to keep the transport's conservative baseline (401/403 ->
    ``auth``, 402 -> ``payment_required``, 429 -> ``rate_limit``, 400 ->
    ``invalid_request``, 404 -> ``not_found``, 500/502/503/504 ->
    ``provider_unavailable``, timeout/network/transport).
    """
    error_type = (exc.provider_error_code or "").strip().lower()
    text = error_haystack(exc)

    if error_type in _PAYMENT_TYPES:  # 1. explicit billing type wins outright
        return CATEGORY_PAYMENT_REQUIRED
    if error_type in _MODEL_TYPES:  # 2. explicit model type
        return CATEGORY_INVALID_MODEL
    if error_type == "rate_limit_error":  # 3. throttled - reset evidence?
        return (
            CATEGORY_QUOTA_WITH_RESET
            if matches_hints(text, _RESET_HINTS)
            else CATEGORY_RATE_LIMIT
        )
    if matches_hints(text, _PAYMENT_HINTS):  # 4. billing wording without a type
        return CATEGORY_PAYMENT_REQUIRED
    if error_type == "not_found_error" and matches_hints(text, _MODEL_HINTS):
        return CATEGORY_INVALID_MODEL  # 5. a 404 that names the model
    if error_type == "invalid_request_error" and matches_hints(text, _MODEL_HINTS):
        return CATEGORY_INVALID_MODEL  # 6. a 400 that names the model
    if error_type in _ANTHROPIC_TYPE_CATEGORIES:  # 7. declared type, no refinement
        return _ANTHROPIC_TYPE_CATEGORIES[error_type]
    if (
        exc.category == CATEGORY_UNKNOWN
        and isinstance(exc.http_status, int)
        and 500 <= exc.http_status < 600
    ):
        return CATEGORY_TEMPORARY_5XX  # 8. any other 5xx is temporary
    return None


def refine_provider_error(
    exc: SupervisorRunError,
    *,
    provider: str,
    model: Optional[str],
    operation: str,
) -> SupervisorRunError:
    """Anthropic-specific refinement plus shared supervisor-context stamping."""
    return stamp_run_error(
        exc,
        provider=provider,
        model=model,
        operation=operation,
        category=classify_anthropic_error(exc),
    )


# --------------------------------------------------------------------------- #
# Request construction (Anthropic Messages body)
# --------------------------------------------------------------------------- #


def messages_system_prompt(operation: str) -> str:
    """The Anthropic JSON-envelope instruction.

    The Messages API has no ``response_format`` field, so the canonical schema
    and the identity-ownership rule are stated in the TOP-LEVEL ``system`` field
    instead of being enforced by the API.
    """
    if str(operation).upper() == PLAN_OPERATION:
        role = "Planner/Architect"
        schema = supervisor_plan_schema()
    else:
        role = "independent Reviewer/Supervisor"
        schema = supervisor_review_schema()
    compact = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (
        f"You are the {role} in Worker_tandem.\n"
        "Reply with ONE JSON object and nothing else - no markdown fences, "
        "no commentary, no prose before or after the JSON object.\n"
        "It must satisfy exactly this JSON Schema:\n"
        f"{compact}\n"
        "Never include step_no, attempt or state: Worker_tandem owns them."
    )


def build_messages_body(
    *,
    model: str,
    system_prompt: str,
    prompt: str,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    """The Anthropic Messages request body (NOT the OpenAI chat body).

    Differences this encodes:

    * ``system`` is a TOP-LEVEL field - the system turn is not a message;
    * ``messages`` carries the user turn only (Anthropic requires at least one);
    * ``max_tokens`` is REQUIRED and is therefore always present;
    * there is no ``response_format``: :func:`messages_system_prompt` asks for
      JSON and :func:`anthropic_strict_json_object` enforces the envelope;
    * no ``stream`` field: the injected transport is synchronous.
    """
    tokens = (
        int(max_output_tokens)
        if isinstance(max_output_tokens, int) and max_output_tokens > 0
        else DEFAULT_ANTHROPIC_MAX_TOKENS
    )
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        body["temperature"] = float(temperature)
    return body


# --------------------------------------------------------------------------- #
# Response extraction (Anthropic content blocks + usage)
# --------------------------------------------------------------------------- #


def content_text(payload: Any) -> tuple[Optional[str], Optional[str]]:
    """``(text, problem)`` for an Anthropic Messages response - deterministic only.

    M7.2 content contract:

    * exactly **one** text block -> its text becomes the candidate payload;
    * **zero** text blocks -> problem (nothing usable was returned);
    * **two or more** text blocks -> problem (**ambiguous**): unrelated blocks are
      never concatenated, because stitching arbitrary text together could
      manufacture a payload that the model never emitted as one object;
    * non-text blocks (``thinking``, ``tool_use``, ...) are ignored, but every
      entry must be a well-formed block: a non-object entry, a missing/blank
      ``type`` or a text block without a string ``text`` is a problem, so a
      malformed response fails closed instead of being partially scavenged.

    ``problem`` is a short, provider-neutral explanation; it never contains the
    provider's text.
    """
    if not isinstance(payload, dict):
        return None, "response payload is not a JSON object"
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return None, "response has no content block list"
    if not blocks:
        return None, "content block list is empty"

    texts: list[str] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            return None, f"content block {index} is not an object"
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type.strip():
            return None, f"content block {index} has no type"
        if block_type.strip() != "text":
            continue  # non-text block: carries no payload text
        text = block.get("text")
        if not isinstance(text, str):
            return None, f"text block {index} has no string text"
        texts.append(text)

    if not texts:
        return None, "content has no text block"
    if len(texts) > 1:
        return None, f"content has {len(texts)} text blocks (ambiguous)"
    text = texts[0].strip()
    if not text:
        return None, "text block is empty"
    return text, None


def message_text(payload: Any) -> Optional[str]:
    """Non-raising convenience view of :func:`content_text`.

    Returns the single deterministic text block, or ``None`` for every problem
    case (missing/empty/ambiguous content). The adapter itself always uses
    :func:`content_text`, so the reason for a rejection is never lost.
    """
    text, _problem = content_text(payload)
    return text


def anthropic_strict_json_object(content: Any) -> Optional[dict]:
    """The strict text envelope applied to Anthropic content.

    M7.2 de-duplication: this now delegates to the **shared** strict helper
    (:func:`supervisor.openai_compatible.strict_json_object`), as the M7.2 spec
    requires reusing the existing strict primary helper. The public name is kept
    so the M7.1 surface stays stable, but there is exactly ONE strict envelope
    implementation in the tree and the DeepSeek/OpenAI behaviour is untouched
    (DeepSeek deliberately keeps the lenient extractor).

    Accepted: a plain JSON object, or one fenced object, with surrounding
    whitespace. Rejected: prose before/after the object, two concatenated
    objects, arrays, scalars, non-JSON - and there is no brace scanning.
    """
    return strict_json_object(content)


def _as_int(value: Any) -> Optional[int]:
    """Best-effort integer, or ``None`` - never an invented value.

    Mirrors the shared converter the OpenAI-compatible providers use, so all
    three API adapters treat provider numeric quirks identically.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def anthropic_usage_from_payload(
    payload: Any, *, latency_ms: Optional[int] = None
) -> ProviderUsage:
    """Anthropic usage -> :class:`ProviderUsage`.

    Anthropic reports ``input_tokens``/``output_tokens`` instead of the
    ``prompt_tokens``/``completion_tokens`` pair the chat-completions providers
    use, so this is the small provider-specific variant of the shared mapper.
    The total is summed when the API omits it, missing values stay ``None``, cost
    is never guessed, and - exactly like the other providers - a
    ``ProviderUsage`` is always returned so ``latency_ms`` is never lost.
    """
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return ProviderUsage(latency_ms=latency_ms)
    input_tokens = _as_int(usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    total = _as_int(usage.get("total_tokens"))
    if total is None and (input_tokens is not None or output_tokens is not None):
        total = (input_tokens or 0) + (output_tokens or 0)
    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        estimated_cost=None,
        latency_ms=latency_ms,
    )


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class AnthropicSupervisorAdapter:
    """``SupervisorPort`` implementation for Anthropic's Messages API.

    M7.1 scope: the adapter is complete and offline-testable, but it is NOT
    registered with the router and NOT wired into the app, so the runtime
    behaviour of Worker_tandem is unchanged.
    """

    provider_id = PROVIDER_ANTHROPIC
    default_api_base = DEFAULT_ANTHROPIC_API_BASE
    default_model = DEFAULT_ANTHROPIC_MODEL
    default_timeout_sec = DEFAULT_ANTHROPIC_TIMEOUT_SEC
    default_key_env = ANTHROPIC_API_KEY_ENV
    default_api_version = DEFAULT_ANTHROPIC_API_VERSION

    def __init__(
        self,
        config: ProviderConfig,
        transport: TransportPort,
        *,
        env: Optional[Mapping[str, str]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        cls = type(self)
        self.config = config
        self.provider_id = config.provider_id or cls.provider_id
        self.model = (config.model or "").strip() or None
        self.api_base = (
            str(config.api_base or "").strip().rstrip("/") or cls.default_api_base
        )
        self.api_key_source = config.api_key_source or f"env:{cls.default_key_env}"
        self._transport = transport
        #: Optional injected environment (tests); ``os.environ`` is the default.
        self._env = env
        self._clock = clock or time.perf_counter
        #: An explicit ``ProviderConfig.api_version`` wins, then the
        #: ``ANTHROPIC_API_VERSION`` env override, then the default. ``getattr``
        #: keeps the shared frozen dataclass untouched: a future field would be
        #: honoured automatically.
        self._api_version_override = str(
            getattr(config, "api_version", "") or ""
        ).strip()

    # -- SupervisorPort ----------------------------------------------------- #

    def plan(self, request: SupervisorPlanRequest) -> SupervisorPlanResult:
        return self._invoke(PLAN_OPERATION, request, normalize_plan_payload)

    def review(self, request: SupervisorReviewRequest) -> SupervisorReviewResult:
        return self._invoke(REVIEW_OPERATION, request, normalize_review_payload)

    def health_check(self) -> ProviderHealth:
        """Cheap probe: ``GET {api_base}/models`` - never a completion.

        A missing/blank key reports ``KEY_MISSING`` and performs no HTTP call.
        """
        key = self._api_key()
        if not key:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=HEALTH_KEY_MISSING,
                detail=sanitize_key_detail(self.api_key_source),
                model=self.model,
            )
        try:
            response = self._transport.get_json(
                self.health_endpoint(),
                headers=self._headers(key),
                timeout_sec=self._timeout_sec(),
            )
        except SupervisorRunError as exc:
            refined = self._refine_error(exc, operation="HEALTH")
            return ProviderHealth(
                provider_id=self.provider_id,
                status=health_status_for(refined.category),
                detail=refined.sanitized_message,
                model=self.model,
            )
        return ProviderHealth(
            provider_id=self.provider_id,
            status=HEALTH_READY,
            detail=self._health_detail(response),
            model=self.model,
        )

    # -- provider-owned pieces --------------------------------------------- #

    def messages_endpoint(self) -> str:
        """``POST {api_base}/messages``."""
        return anthropic_endpoint(self.api_base, MESSAGES_PATH)

    def health_endpoint(self) -> str:
        """``GET {api_base}/models`` (the shared cheap probe path)."""
        return anthropic_endpoint(self.api_base, MODELS_PATH)

    def api_version(self) -> str:
        """The ``anthropic-version`` header value."""
        environ: Mapping[str, str] = os.environ if self._env is None else self._env
        override = str(environ.get(ANTHROPIC_API_VERSION_ENV) or "").strip()
        return self._api_version_override or override or self.default_api_version

    def _api_key(self) -> Optional[str]:
        """Resolved at call time; never cached on the instance."""
        return resolve_api_key(self.api_key_source, env=self._env)

    def api_key_env(self) -> str:
        """The environment variable NAME this adapter reads (never a value).

        The GUI/ops layer can display this instead of hard-coding the name.
        """
        return api_key_env_name(self.api_key_source) or self.default_key_env

    def _headers(self, key: str) -> dict:
        """Anthropic authenticates with ``x-api-key`` (not a bearer token)."""
        return {
            "x-api-key": key,
            "anthropic-version": self.api_version(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _timeout_sec(self) -> Optional[int]:
        value = self.config.timeout_sec
        return int(value) if isinstance(value, int) and value > 0 else None

    def _health_detail(self, response: Any) -> str:
        payload = getattr(response, "payload", None)
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            names = [
                entry.get("id")
                for entry in data
                if isinstance(entry, dict) and entry.get("id")
            ]
            return f"{len(names)} models available" if names else "reachable"
        return "reachable"

    def _refine_error(self, exc: SupervisorRunError, *, operation: str) -> SupervisorRunError:
        return refine_provider_error(
            exc, provider=self.provider_id, model=self.model, operation=operation
        )

    # -- the single execution path ------------------------------------------ #

    def _invoke(self, operation: str, request: Any, normalizer):
        # 1. local preconditions - no request is attempted without them.
        key = self._api_key()
        if not key:
            raise key_missing_run_error(
                self.provider_id,
                model=self.model,
                operation=operation,
                api_key_source=self.api_key_source,
            )
        self._require_model(operation)
        self._require_temperature(operation)

        # 2. execution through the injected transport (Messages API).
        body = self._build_body(operation, request)
        started = self._clock()
        try:
            response = self._transport.post_json(
                self.messages_endpoint(),
                headers=self._headers(key),
                json_body=body,
                timeout_sec=self._timeout_sec(),
            )
        except SupervisorRunError as exc:
            refined = self._refine_error(exc, operation=operation)
            # Never ``raise x from x``: a self-referential ``__cause__`` would
            # break traceback rendering.
            if refined is exc:
                raise
            raise refined from exc

        latency_ms = max(0, int(round((self._clock() - started) * 1000)))

        # 3. untrusted output -> text -> strict JSON -> canonical validation.
        payload = dict(
            normalizer(
                self._parse_content(response.payload, operation),
                provider=self.provider_id,
                model=self.model,
                operation=operation,
            )
        )
        usage = anthropic_usage_from_payload(response.payload, latency_ms=latency_ms)
        if operation == PLAN_OPERATION:
            return SupervisorPlanResult(
                payload=payload,
                provider=self.provider_id,
                model=self.model,
                usage=usage,
            )
        return SupervisorReviewResult(
            payload=payload,
            provider=self.provider_id,
            model=self.model,
            usage=usage,
        )

    # -- local preconditions (fail closed, no HTTP) ------------------------- #

    def _require_model(self, operation: str) -> None:
        if self.model:
            return
        raise SupervisorRunError(
            "no provider model configured",
            provider=self.provider_id,
            model=None,
            operation=operation,
            category=CATEGORY_INVALID_REQUEST,
            retryable=False,
            sanitized_message=(
                "ProviderConfig.model is empty; set a model before calling "
                "the provider."
            ),
        )

    def _require_temperature(self, operation: str) -> None:
        """Anthropic accepts ``temperature`` in ``[0, 1]`` only."""
        value = self.config.temperature
        if value is None:
            return
        low, high = ANTHROPIC_TEMPERATURE_RANGE
        try:
            number: Optional[float] = float(value)
        except (TypeError, ValueError):
            number = None
        if number is not None and low <= number <= high:
            return
        raise SupervisorRunError(
            "provider temperature is out of range",
            provider=self.provider_id,
            model=self.model,
            operation=operation,
            category=CATEGORY_INVALID_REQUEST,
            retryable=False,
            sanitized_message=(
                f"Anthropic accepts temperature in [{low:g}, {high:g}]; "
                f"ProviderConfig.temperature={value!r} was rejected locally."
            ),
        )

    # -- request / response construction ------------------------------------ #

    def _build_body(self, operation: str, request: Any) -> dict:
        """The Messages body for one operation (never the OpenAI chat body)."""
        return build_messages_body(
            model=self.model or "",
            system_prompt=self._system_prompt(operation),
            prompt=getattr(request, "prompt", "") or "",
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
        )

    def _system_prompt(self, operation: str) -> str:
        return messages_system_prompt(operation)

    def _parse_content(self, payload: Any, operation: str) -> dict:
        """Response content -> ONE deterministic JSON object (fail closed).

        ``content_text`` enforces the M7.2 deterministic content contract and
        the strict envelope enforces the JSON contract; canonical validation
        (which also rejects controller-owned ``step_no``/``attempt``/``state``)
        happens in ``_invoke`` via the shared normalizer.
        """
        text, problem = content_text(payload)
        if problem is not None:
            raise SupervisorRunError(
                "provider returned unusable content",
                provider=self.provider_id,
                model=self.model,
                operation=operation,
                category=CATEGORY_CONTRACT,
                retryable=False,
                sanitized_message=f"Anthropic content rejected: {problem}.",
            )
        parsed = anthropic_strict_json_object(text)
        if parsed is None:
            raise SupervisorRunError(
                "provider content was not a single JSON object",
                provider=self.provider_id,
                model=self.model,
                operation=operation,
                category=CATEGORY_CONTRACT,
                retryable=False,
                sanitized_message=(
                    "Anthropic content must be exactly one JSON object: no prose "
                    "before or after it, no multiple objects, no arrays or "
                    "scalars, no embedded-object scanning."
                ),
            )
        return parsed





