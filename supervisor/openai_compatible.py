"""Shared OpenAI-compatible provider helpers (Milestone M6).

DeepSeek (M5) and OpenAI (M6) speak the same REST contract, so the code that is
PROVEN common now lives here exactly once:

* endpoint construction;
* content extraction from a chat-completions response;
* token-usage normalization;
* error "stamping" (supervisor context + fail-closed retryability);
* canonical error category -> health status mapping;
* request-body and system-prompt construction;
* a STRICT JSON extractor for API providers.

Deliberately NOT here (these stay provider-specific, as M6 requires):
provider id, ``api_base`` defaults, auth/key source, default model, default
timeout, provider-specific error refinements, health endpoint choice and any
provider-specific request options.

Nothing in this module performs I/O: it is pure, dependency-free and therefore
trivially testable.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Mapping, Optional

from .base import (
    api_key_env_name,
    key_missing_run_error,
    normalize_plan_payload,
    normalize_review_payload,
    resolve_api_key,
)
from .errors import (
    CATEGORY_AUTH,
    CATEGORY_INVALID_API_KEY,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_PAYMENT_REQUIRED,
    CATEGORY_PROVIDER_UNAVAILABLE,
    CATEGORY_QUOTA_WITH_RESET,
    CATEGORY_RATE_LIMIT,
    CATEGORY_TEMPORARY_5XX,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    CATEGORY_UNKNOWN,
    SupervisorRunError,
    is_retryable_category,
    sanitize_diagnostic,
)
from .models import (
    HEALTH_AUTH_ERROR,
    HEALTH_KEY_MISSING,
    HEALTH_NETWORK_ERROR,
    HEALTH_PROVIDER_ERROR,
    HEALTH_QUOTA_LIMIT,
    HEALTH_RATE_LIMIT,
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
from .schemas import supervisor_plan_schema, supervisor_review_schema
from .transport import TransportPort

# --------------------------------------------------------------------------- #
# Endpoints (construction centralized; provider supplies the default base)
# --------------------------------------------------------------------------- #

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"


def api_endpoint(api_base: Optional[str], path: str, *, default_base: str) -> str:
    """``{api_base or default_base}{path}`` with exactly one separator."""
    base = str(api_base or "").strip().rstrip("/") or default_base
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


# --------------------------------------------------------------------------- #
# Strict JSON extraction for API providers
# --------------------------------------------------------------------------- #

_FENCE = "```"


def strict_json_object(content: Any) -> Optional[dict]:
    """A JSON object, optionally inside a single fenced block. Nothing else.

    Deliberately STRICTER than :func:`supervisor.schemas.extract_json_object`
    (which Codex Legacy still relies on): there is no prose tolerance and no
    brace scanning, so an ambiguous provider response can never be silently
    scavenged into a valid-looking payload.

    Accepted:  ``{"a":1}``, ``\\n {"a":1} \\n``, ```` ```json {...} ``` ````.
    Rejected:  prose before/after, two concatenated objects, arrays, non-JSON.
    """
    if not isinstance(content, str):
        return None

    text = content.strip()
    if text.startswith(_FENCE):
        # drop the opening fence line (it may carry a language tag)
        _, _, remainder = text.partition("\n")
        if not remainder:
            return None
        text = remainder
        closing = text.rfind(_FENCE)
        if closing != -1:
            text = text[:closing]
        text = text.strip()

    if not text.startswith("{") or not text.endswith("}"):
        return None

    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# --------------------------------------------------------------------------- #
# Response interpretation (proven common to DeepSeek + OpenAI)
# --------------------------------------------------------------------------- #


def extract_message_content(payload: Any) -> Optional[str]:
    """Conservative extractor for an OpenAI-compatible chat response.

    Handles ``choices[0].message.content`` (string or part list),
    ``choices[0].text`` and a couple of plausible top-level fallbacks. Returns
    ``None`` when nothing usable is present.
    """
    if not isinstance(payload, dict):
        return None

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    parts = [
                        part.get("text")
                        for part in content
                        if isinstance(part, dict)
                    ]
                    joined = "".join(p for p in parts if isinstance(p, str))
                    if joined.strip():
                        return joined
            text = choice.get("text")
            if isinstance(text, str) and text.strip():
                return text

    for key in ("output_text", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _as_int(value: Any) -> Optional[int]:
    """Best-effort integer, or ``None`` - never an invented value."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def usage_from_payload(
    payload: Any, *, latency_ms: Optional[int] = None
) -> ProviderUsage:
    """Real token usage when the provider reported it; otherwise all ``None``.

    Both DeepSeek and OpenAI (chat completions) use ``prompt_tokens`` /
    ``completion_tokens`` / ``total_tokens``. ``estimated_cost`` is NEVER
    guessed: V1 models no pricing metadata.
    """
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return ProviderUsage(latency_ms=latency_ms)
    return ProviderUsage(
        input_tokens=_as_int(usage.get("prompt_tokens")),
        output_tokens=_as_int(usage.get("completion_tokens")),
        total_tokens=_as_int(usage.get("total_tokens")),
        estimated_cost=None,
        latency_ms=latency_ms,
    )


# --------------------------------------------------------------------------- #
# Error stamping (supervisor context) + health mapping
# --------------------------------------------------------------------------- #


def error_haystack(exc: SupervisorRunError) -> str:
    """Lower-cased structured error text: provider message + code/type.

    The transport fills ``sanitized_message`` from ``error.message`` and
    ``provider_error_code`` from ``error.code``/``error.type``, so this is the
    structured evidence an adapter classifies from - not vague prose.
    """
    return f"{exc.sanitized_message or ''} {exc.provider_error_code or ''}".lower()


def matches_hints(text: str, hints: Iterable[str]) -> bool:
    return any(hint in text for hint in hints)


def stamp_run_error(
    exc: SupervisorRunError,
    *,
    provider: str,
    model: Optional[str],
    operation: str,
    category: Optional[str] = None,
) -> SupervisorRunError:
    """Rebuild a transport error carrying supervisor context.

    The transport only knows the HTTP method, so ``operation``/``provider``/
    ``model`` and the retryability are (re)stamped here. Retry decisions always
    come from the canonical taxonomy (fail closed). Provider-specific category
    refinement is supplied by the adapter via ``category``.
    """
    final = category or exc.category
    return SupervisorRunError(
        exc.sanitized_message or "provider request failed",
        provider=provider,
        model=model,
        operation=operation,
        category=final,
        retryable=is_retryable_category(final),
        http_status=exc.http_status,
        provider_error_code=exc.provider_error_code,
        sanitized_message=exc.sanitized_message,
        raw_response_ref=exc.raw_response_ref,
        usage=exc.usage,
    )


#: Canonical error category -> health status.
HEALTH_BY_CATEGORY = {
    CATEGORY_AUTH: HEALTH_AUTH_ERROR,
    CATEGORY_INVALID_API_KEY: HEALTH_AUTH_ERROR,
    CATEGORY_RATE_LIMIT: HEALTH_RATE_LIMIT,
    CATEGORY_QUOTA_WITH_RESET: HEALTH_QUOTA_LIMIT,
    CATEGORY_PAYMENT_REQUIRED: HEALTH_QUOTA_LIMIT,
    CATEGORY_TIMEOUT: HEALTH_NETWORK_ERROR,
    CATEGORY_NETWORK: HEALTH_NETWORK_ERROR,
    CATEGORY_TRANSPORT: HEALTH_NETWORK_ERROR,
    CATEGORY_PROVIDER_UNAVAILABLE: HEALTH_PROVIDER_ERROR,
    CATEGORY_TEMPORARY_5XX: HEALTH_PROVIDER_ERROR,
    CATEGORY_UNKNOWN: HEALTH_PROVIDER_ERROR,
}


def health_status_for(category: str) -> str:
    return HEALTH_BY_CATEGORY.get(category, HEALTH_PROVIDER_ERROR)


def sanitize_key_detail(api_key_source: Optional[str]) -> str:
    """A ``KEY_MISSING`` detail naming only the environment VARIABLE."""
    name = api_key_env_name(api_key_source) or "(unset)"
    return sanitize_diagnostic(f"{name} is not set")


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def build_chat_completions_body(
    *,
    model: str,
    system_prompt: str,
    prompt: str,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_object: bool = False,
    stream: bool = False,
) -> dict:
    """The shared OpenAI-compatible chat-completions request body."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
    }
    if isinstance(max_output_tokens, int) and max_output_tokens > 0:
        body["max_tokens"] = max_output_tokens
    if temperature is not None:
        body["temperature"] = float(temperature)
    if json_object:
        body["response_format"] = {"type": "json_object"}
    return body


def json_object_system_prompt(
    operation: str,
    *,
    plan_role: str = "Planner/Architect",
    review_role: str = "independent Reviewer/Supervisor",
) -> str:
    """The shared structured-output system prompt.

    States the role, the exact canonical schema and the identity-ownership rule
    (``step_no``/``attempt``/``state`` belong to Worker_tandem, not the model).
    """
    if operation == PLAN_OPERATION:
        role = plan_role
        schema = supervisor_plan_schema()
    else:
        role = review_role
        schema = supervisor_review_schema()
    compact = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (
        f"You are the {role} in Worker_tandem.\n"
        "Reply with ONE JSON object and nothing else - no markdown fences, "
        "no commentary.\n"
        "It must satisfy exactly this JSON Schema:\n"
        f"{compact}\n"
        "Never include step_no, attempt or state: Worker_tandem owns them."
    )


# --------------------------------------------------------------------------- #
# Shared adapter skeleton (used by DeepSeek M5 and OpenAI M6)
# --------------------------------------------------------------------------- #


class OpenAICompatibleSupervisorAdapter:
    """Shared ``SupervisorPort`` skeleton for chat-completions providers.

    Subclasses supply ONLY the provider-specific pieces (exactly the list M6
    says must stay provider-owned):

    * ``provider_id`` / ``default_api_base`` / ``default_model`` /
      ``default_timeout_sec`` / ``default_key_env`` class attributes;
    * ``_parse_content(payload, operation)`` - strict vs. lenient JSON;
    * ``_refine_error(exc, ...)`` - provider-specific error classification;
    * ``_health_path()`` / ``_health_detail(response)`` when the probe differs.

    Everything else - key resolution, model preconditions, request body,
    transport call, usage extraction, health mapping, error stamping - exists
    exactly once, here.
    """

    provider_id = ""
    default_api_base = ""
    default_model = ""
    default_timeout_sec = 180
    default_key_env = ""

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

    # -- hooks the provider must own --------------------------------------- #

    def _parse_content(self, payload: Any, operation: str) -> dict:
        """content -> canonical JSON object. Provider-specific strictness."""
        raise NotImplementedError

    def _refine_error(
        self, exc: SupervisorRunError, *, provider: str, model: Optional[str], operation: str
    ) -> SupervisorRunError:
        """Provider-specific error classification over the transport baseline."""
        raise NotImplementedError

    def _health_path(self) -> str:
        return MODELS_PATH

    def _health_detail(self, response) -> str:
        data = (
            response.payload.get("data")
            if isinstance(response.payload, dict)
            else None
        )
        if isinstance(data, list):
            names = [
                entry.get("id")
                for entry in data
                if isinstance(entry, dict) and entry.get("id")
            ]
            return f"{len(names)} models available" if names else "reachable"
        return "reachable"

    # -- SupervisorPort ----------------------------------------------------- #

    def plan(self, request: SupervisorPlanRequest) -> SupervisorPlanResult:
        return self._invoke(PLAN_OPERATION, request, normalize_plan_payload)

    def review(self, request: SupervisorReviewRequest) -> SupervisorReviewResult:
        return self._invoke(REVIEW_OPERATION, request, normalize_review_payload)

    def health_check(self) -> ProviderHealth:
        """Cheap probe: ``GET {api_base}/models``. Never a completion request."""
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
                api_endpoint(
                    self.api_base,
                    self._health_path(),
                    default_base=self.default_api_base,
                ),
                headers=self._headers(key),
                timeout_sec=self._timeout_sec(),
            )
        except SupervisorRunError as exc:
            # Provider-specific classification also applies to the health probe:
            # a 429 carrying `insufficient_quota` is a QUOTA problem, not a
            # plain rate limit.
            refined = self._refine_error(
                exc,
                provider=self.provider_id,
                model=self.model,
                operation="HEALTH",
            )
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

    # -- internals ---------------------------------------------------------- #

    def _api_key(self) -> Optional[str]:
        """Resolved at call time; never cached on the instance."""
        return resolve_api_key(self.api_key_source, env=self._env)

    @staticmethod
    def _headers(key: str) -> dict:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _timeout_sec(self) -> Optional[int]:
        value = self.config.timeout_sec
        return int(value) if isinstance(value, int) and value > 0 else None

    def _chat_path(self) -> str:
        return CHAT_COMPLETIONS_PATH

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
        if not self.model:
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

        # 2. provider execution through the injected transport.
        body = self._build_body(operation, request.prompt)
        started = self._clock()
        try:
            response = self._transport.post_json(
                api_endpoint(
                    self.api_base,
                    self._chat_path(),
                    default_base=self.default_api_base,
                ),
                headers=self._headers(key),
                json_body=body,
                timeout_sec=self._timeout_sec(),
            )
        except SupervisorRunError as exc:
            refined = self._refine_error(
                exc, provider=self.provider_id, model=self.model, operation=operation
            )
            # Never ``raise x from x``: a self-referential ``__cause__`` would
            # break traceback rendering.
            if refined is exc:
                raise
            raise refined from exc

        latency_ms = max(0, int(round((self._clock() - started) * 1000)))

        # 3. untrusted output -> JSON -> canonical validation -> normalize.
        payload = dict(
            normalizer(
                self._parse_content(response.payload, operation),
                provider=self.provider_id,
                model=self.model,
                operation=operation,
            )
        )
        usage = usage_from_payload(response.payload, latency_ms=latency_ms)
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

    # -- request construction ---------------------------------------------- #

    def _build_body(self, operation: str, prompt: str) -> dict:
        """Provider request OPTIONS over the shared OpenAI-compatible body."""
        return build_chat_completions_body(
            model=self.model,
            system_prompt=self._system_prompt(operation),
            prompt=prompt,
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            json_object=self.config.supports_structured_output,
        )

    def _system_prompt(self, operation: str) -> str:
        return json_object_system_prompt(operation)




