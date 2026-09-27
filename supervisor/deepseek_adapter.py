"""DeepSeek API supervisor adapter (Milestone M5).

A real, provider-neutral ``SupervisorPort`` implementation backed by the
DeepSeek chat-completions REST API (OpenAI-compatible), using ONLY the shared
``supervisor.transport`` layer - no ``requests``, no ``urllib``, no DeepSeek
SDK, no second HTTP abstraction.

Invariants
----------
* The API key is read from the environment (``DEEPSEEK_API_KEY``) at CALL time;
  it is never persisted, logged, echoed in an error or returned.
* ``api_base`` and ``model`` come from :class:`~supervisor.models.ProviderConfig`.
  DeepSeek's defaults live in exactly one place (:func:`default_deepseek_config`)
  and endpoint construction is centralized (:func:`deepseek_endpoint`).
* Prompt/context ownership stays outside the provider: the adapter only
  serializes ``SupervisorPlanRequest``/``SupervisorReviewRequest``.
* The provider never owns ``step_no``/``attempt``/FSM state: a model that echoes
  them is rejected by the canonical schema validation.
* Provider output is untrusted: content -> JSON -> canonical validation ->
  normalization. Nothing malformed ever reaches the controller.
"""

from __future__ import annotations

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
    CATEGORY_CONTRACT,
    CATEGORY_INVALID_MODEL,
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NOT_FOUND,
    SupervisorRunError,
    sanitize_diagnostic,
)
from .models import (
    HEALTH_KEY_MISSING,
    HEALTH_PROVIDER_ERROR,
    HEALTH_READY,
    PLAN_OPERATION,
    REVIEW_OPERATION,
    ProviderConfig,
    ProviderHealth,
    SupervisorPlanRequest,
    SupervisorPlanResult,
    SupervisorReviewRequest,
    SupervisorReviewResult,
)
from .schemas import (
    extract_json_object,
    supervisor_plan_schema,
    supervisor_review_schema,
)
from .transport import TransportPort

#: Shared OpenAI-compatible helpers (M6). The path constants and the generic
#: helpers are re-exported from here so existing DeepSeek imports keep working
#: unchanged: ``CHAT_COMPLETIONS_PATH``, ``MODELS_PATH``,
#: ``extract_message_content``, ``usage_from_payload``, ``health_status_for``.
from .openai_compatible import (
    CHAT_COMPLETIONS_PATH,
    MODELS_PATH,
    OpenAICompatibleSupervisorAdapter,
    api_endpoint,
    build_chat_completions_body,
    error_haystack,
    extract_message_content,
    health_status_for,
    json_object_system_prompt,
    matches_hints,
    stamp_run_error,
    usage_from_payload,
)

# --------------------------------------------------------------------------- #
# Provider identity + centralized defaults
# --------------------------------------------------------------------------- #

PROVIDER_DEEPSEEK = "deepseek"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_TIMEOUT_SEC = 180

#: DeepSeek is OpenAI-compatible; the shared module owns the path constants
#: (imported above, so they stay importable from this module unchanged).


def default_deepseek_config(**overrides: Any) -> ProviderConfig:
    """The ONE place DeepSeek's default base URL and model are expressed."""
    values: dict[str, Any] = {
        "provider_id": PROVIDER_DEEPSEEK,
        "display_name": "DeepSeek",
        "model": DEFAULT_DEEPSEEK_MODEL,
        "enabled": True,
        "api_base": DEFAULT_DEEPSEEK_API_BASE,
        "api_key_source": f"env:{DEEPSEEK_API_KEY_ENV}",
        "timeout_sec": DEFAULT_DEEPSEEK_TIMEOUT_SEC,
        "max_output_tokens": None,
        "temperature": None,
        "supports_structured_output": True,
        "cost_policy": "cheap",
        "priority": 1,
    }
    values.update(overrides)
    return ProviderConfig(**values)


def deepseek_endpoint(api_base: Optional[str], path: str) -> str:
    """Centralized endpoint construction (delegates to the shared helper)."""
    return api_endpoint(api_base, path, default_base=DEFAULT_DEEPSEEK_API_BASE)


# --------------------------------------------------------------------------- #
# Provider-specific error refinement
# --------------------------------------------------------------------------- #


#: ``_as_int``, ``usage_from_payload`` and ``health_status_for`` moved to the
#: shared ``supervisor.openai_compatible`` module in M6 (imported above; still
#: importable from here for backward compatibility).


#: Substrings that let the adapter promote a generic 404 to ``invalid_model``.
_MODEL_ERROR_HINTS = (
    "model",
    "does not exist",
    "not exist",
    "unknown",
    "unsupported",
)


def refine_provider_error(
    exc: SupervisorRunError,
    *,
    provider: str,
    model: Optional[str],
    operation: str,
) -> SupervisorRunError:
    """DeepSeek-specific refinement plus the shared supervisor-context stamping.

    DeepSeek's only provider-specific rule: a generic 404 may become
    ``invalid_model`` when the provider's own error text points at the model.
    Everything else (operation/provider/model stamping and the fail-closed
    retry decision) comes from the shared helper.
    """
    category = None
    if exc.category == CATEGORY_NOT_FOUND and matches_hints(
        error_haystack(exc), _MODEL_ERROR_HINTS
    ):
        category = CATEGORY_INVALID_MODEL
    return stamp_run_error(
        exc, provider=provider, model=model, operation=operation, category=category
    )


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


class DeepSeekSupervisorAdapter(OpenAICompatibleSupervisorAdapter):
    """A real DeepSeek API ``SupervisorPort`` implementation.

    Provider-specific pieces only. The shared chat-completions machinery (key
    resolution, model preconditions, request body, transport call, usage
    extraction, health mapping, error stamping) lives in
    :class:`supervisor.openai_compatible.OpenAICompatibleSupervisorAdapter`.
    """

    provider_id = PROVIDER_DEEPSEEK
    default_api_base = DEFAULT_DEEPSEEK_API_BASE
    default_model = DEFAULT_DEEPSEEK_MODEL
    default_timeout_sec = DEFAULT_DEEPSEEK_TIMEOUT_SEC
    default_key_env = DEEPSEEK_API_KEY_ENV

    # -- provider-specific hook --------------------------------------------- #

    def _refine_error(
        self,
        exc: SupervisorRunError,
        *,
        provider: str,
        model: Optional[str],
        operation: str,
    ) -> SupervisorRunError:
        return refine_provider_error(
            exc, provider=provider, model=model, operation=operation
        )

    # -- inherited from the shared skeleton ---------------------------------- #
    # ``_api_key``, ``_headers``, ``_timeout_sec``, ``_health_path``,
    # ``_health_detail``, ``_invoke``, ``_build_body`` and ``_system_prompt``
    # all come from OpenAICompatibleSupervisorAdapter.

    # -- request construction ---------------------------------------------- #

    def _parse_content(self, payload: Any, operation: str) -> dict:
        """content -> JSON object, failing closed on anything malformed.

        M5 BEHAVIOUR PRESERVED: DeepSeek still uses the lenient canonical
        ``extract_json_object`` (which tolerates prose around the object).

        M6 introduced the stricter ``openai_compatible.strict_json_object`` for
        the OpenAI adapter. DeepSeek is deliberately NOT switched in M6 - the
        M6 spec forbids changing M5 behaviour - so the two API adapters differ
        here until a dedicated milestone unifies them.

        TODO(M7): adopt ``strict_json_object`` for DeepSeek once the tightening
        is approved, and delete this divergence.
        """
        content = extract_message_content(payload)
        if not content:
            raise SupervisorRunError(
                "provider returned no message content",
                provider=self.provider_id,
                model=self.model,
                operation=operation,
                category=CATEGORY_CONTRACT,
                retryable=False,
                sanitized_message="DeepSeek response contained no message content",
            )
        parsed = extract_json_object(content)
        if parsed is None:
            raise SupervisorRunError(
                "provider returned malformed JSON",
                provider=self.provider_id,
                model=self.model,
                operation=operation,
                category=CATEGORY_CONTRACT,
                retryable=False,
                sanitized_message=sanitize_diagnostic(
                    f"not a JSON object: {content[:200]}"
                ),
            )
        return parsed




