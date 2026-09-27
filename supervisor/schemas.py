"""Canonical, provider-neutral PLAN/REVIEW schemas and validation (M1).

Every provider response is normalized and validated against these schemas
before it can reach the controller. Validation FAILS CLOSED: malformed output
never mutates the FSM.

The schemas are identical in shape to the current Codex PLAN/REVIEW schemas
and, like them, contain NO ``step_no``: step identity is controller-owned.

The validator is a small, dependency-free JSON-Schema (draft-07 subset)
checker, byte-for-byte behaviorally compatible with the one already used by
Worker Tandem for Codex structured output.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from .errors import CATEGORY_CONTRACT, SupervisorRunError
from .models import PLAN_DECISIONS, REVIEW_VERDICTS

#: Keys owned by the controller. A provider must never treat them as its own
#: authority; validation rejects any payload that carries them.
CONTROLLER_OWNED_KEYS = ("step_no", "attempt", "state")


def supervisor_plan_schema() -> dict:
    """The one canonical PLAN schema every provider must satisfy."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": list(PLAN_DECISIONS)},
            "goal": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "changes": {"type": "array", "items": {"type": "string"}},
            "checks": {"type": "array", "items": {"type": "string"}},
            "architecture_risk": {"type": "string"},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "rollback": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": [
            "decision",
            "goal",
            "files",
            "changes",
            "checks",
            "architecture_risk",
            "open_questions",
            "rollback",
            "summary",
        ],
    }


def supervisor_review_schema() -> dict:
    """The one canonical REVIEW schema every provider must satisfy."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": list(REVIEW_VERDICTS)},
            "issues": {"type": "array", "items": {"type": "string"}},
            "required_changes": {"type": "array", "items": {"type": "string"}},
            "checks": {"type": "array", "items": {"type": "string"}},
            "architecture_risk": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": [
            "verdict",
            "issues",
            "required_changes",
            "checks",
            "architecture_risk",
            "summary",
        ],
    }


# --------------------------------------------------------------------------- #
# Dependency-free JSON-Schema subset validator
# --------------------------------------------------------------------------- #


def validate_against_schema(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Minimal, dependency-free JSON Schema check for the canonical schemas.

    Returns a list of human-readable problems; an empty list means valid.
    Supports: type, enum, object(required/properties/additionalProperties),
    array(items), string, integer, number, boolean.
    """
    problems: list[str] = []
    if not isinstance(schema, dict):
        return problems

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        problems.append(f"{path}: {value!r} is not one of {enum}")

    stype = schema.get("type")
    if stype == "object":
        if not isinstance(value, dict):
            problems.append(f"{path}: expected object, got {type(value).__name__}")
            return problems
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(f"{path}: missing required key '{key}'")
        for key, subschema in properties.items():
            if key in value:
                problems.extend(
                    validate_against_schema(value[key], subschema, f"{path}.{key}")
                )
        if schema.get("additionalProperties") is False:
            extra = [key for key in value if key not in properties]
            if extra:
                problems.append(f"{path}: unexpected keys {extra}")
    elif stype == "array":
        if not isinstance(value, list):
            problems.append(f"{path}: expected array, got {type(value).__name__}")
            return problems
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                problems.extend(
                    validate_against_schema(item, items, f"{path}[{index}]")
                )
    elif stype == "string":
        if not isinstance(value, str):
            problems.append(f"{path}: expected string, got {type(value).__name__}")
    elif stype == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected integer, got {type(value).__name__}")
    elif stype == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected number, got {type(value).__name__}")
    elif stype == "boolean":
        if not isinstance(value, bool):
            problems.append(f"{path}: expected boolean, got {type(value).__name__}")
    return problems


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from model text."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if "\n" in stripped:
            first, rest = stripped.split("\n", 1)
            if first.strip() and not first.strip().startswith("{"):
                stripped = rest.strip()
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        return obj

    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(stripped[start:index + 1])
                    except (ValueError, TypeError):
                        obj = None
                    if isinstance(obj, dict):
                        return obj
                    break
        start = stripped.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- #
# Payload validation helpers (pure; return problems)
# --------------------------------------------------------------------------- #


def controller_owned_identity_problems(payload: Any) -> list[str]:
    """Problems when a provider payload claims controller-owned identity."""
    if not isinstance(payload, dict):
        return []
    return [
        f"$: provider must not supply controller-owned key '{key}'"
        for key in CONTROLLER_OWNED_KEYS
        if key in payload
    ]


def validate_plan_payload(payload: Any) -> list[str]:
    """Schema problems for a PLAN payload (empty list == valid)."""
    problems = validate_against_schema(payload, supervisor_plan_schema())
    problems.extend(controller_owned_identity_problems(payload))
    return problems


def validate_review_payload(payload: Any) -> list[str]:
    """Schema problems for a REVIEW payload (empty list == valid)."""
    problems = validate_against_schema(payload, supervisor_review_schema())
    problems.extend(controller_owned_identity_problems(payload))
    return problems


# --------------------------------------------------------------------------- #
# Canonical serialization helpers (stable hashes for audit/telemetry)
# --------------------------------------------------------------------------- #


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


