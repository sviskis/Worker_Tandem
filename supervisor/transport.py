"""Synchronous, injectable HTTP transport for supervisor adapters (M2).

One dependency: ``httpx``. No provider SDKs.

Design guarantees:
* synchronous V1;
* ONE shared ``httpx.Client`` per ``HttpTransport`` instance (never one per
  request) unless a client is injected;
* explicit timeouts, automatic retries disabled (``httpx`` default);
* the response body is read with a byte counter and the read is ABORTED as
  soon as ``max_response_bytes`` is exceeded -- the body is never buffered
  unbounded first;
* Authorization / API-key values never appear in raised diagnostics or logs;
* transport exceptions and HTTP statuses are normalized into provider-neutral
  ``SupervisorRunError`` categories.

Adapters receive a transport (``HttpTransport`` or a test double) via
constructor injection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

from .errors import (
    CATEGORY_INVALID_REQUEST,
    CATEGORY_NETWORK,
    CATEGORY_RESPONSE_TOO_LARGE,
    CATEGORY_TIMEOUT,
    CATEGORY_TRANSPORT,
    SupervisorRunError,
    category_for_http_status,
    sanitize_diagnostic,
)

#: Default per-request timeout. API providers get 60-180s, never 1800s.
DEFAULT_TIMEOUT_SEC = 180

#: Hard cap on a single provider response body (1 MiB).
MAX_RESPONSE_BYTES = 1_048_576


@dataclass(frozen=True)
class TransportResponse:
    """A successful (2xx) provider response."""

    status_code: int
    payload: Any = None
    text: str = ""
    raw_bytes: int = 0


@runtime_checkable
class TransportPort(Protocol):
    """The injectable transport seam real adapters depend on."""

    def post_json(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        json_body: Any = None,
        timeout_sec: Optional[int] = None,
    ) -> TransportResponse:
        ...

    def get_json(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        timeout_sec: Optional[int] = None,
    ) -> TransportResponse:
        ...


class HttpTransport:
    """Thin, provider-neutral wrapper around one shared ``httpx.Client``."""

    def __init__(
        self,
        *,
        client: Optional[httpx.Client] = None,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.timeout_sec = int(timeout_sec)
        self.max_response_bytes = int(max_response_bytes)
        if client is not None:
            # Injected client (e.g. httpx.MockTransport backed) - we do not own it.
            self._client = client
            self._owns_client = False
        else:
            # One shared synchronous client per transport instance. httpx has
            # no automatic retries, which is exactly what we want: retry policy
            # belongs to the supervisor router, not the HTTP layer.
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout_sec),
                follow_redirects=False,
                transport=transport,
            )
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                pass

    def __enter__(self) -> "HttpTransport":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- error builders (all sanitized) ------------------------------------ #

    def _transport_error(
        self, category: str, exc: Exception, operation: str
    ) -> SupervisorRunError:
        return SupervisorRunError(
            f"provider transport failure: {type(exc).__name__}",
            operation=operation,
            category=category,
            sanitized_message=sanitize_diagnostic(f"{type(exc).__name__}: {exc}"),
        )

    def _invalid_url(self, exc: Exception, operation: str) -> SupervisorRunError:
        return SupervisorRunError(
            "invalid provider endpoint URL",
            operation=operation,
            category=CATEGORY_INVALID_REQUEST,
            sanitized_message=sanitize_diagnostic(f"{type(exc).__name__}: {exc}"),
        )

    @staticmethod
    def _require_http_url(url: str, operation: str) -> None:
        """Reject a non-absolute/non-http(s) endpoint deterministically.

        A missing scheme is a permanent configuration error (non-retryable),
        and is never reported as a retryable transport outage. The URL itself
        is never echoed: it could embed credentials.
        """
        scheme = ""
        try:
            scheme = urlsplit(str(url)).scheme.lower()
        except Exception:
            scheme = ""
        if scheme not in ("http", "https"):
            raise SupervisorRunError(
                "invalid provider endpoint URL",
                operation=operation,
                category=CATEGORY_INVALID_REQUEST,
                sanitized_message=(
                    f"endpoint URL must be absolute http(s); got scheme "
                    f"{scheme or '(none)'}"
                ),
            )


    # -- core send --------------------------------------------------------- #

    def _send(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict] = None,
        json_body: Any = None,
        timeout_sec: Optional[int] = None,
    ) -> TransportResponse:
        timeout = httpx.Timeout(
            timeout_sec if timeout_sec is not None else self.timeout_sec
        )
        self._require_http_url(url, method)
        try:
            request = self._client.build_request(
                method, url, headers=headers, json=json_body, timeout=timeout
            )
        except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
            raise self._invalid_url(exc, method) from exc

        try:
            response = self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            # ConnectTimeout / ReadTimeout / WriteTimeout / PoolTimeout.
            raise self._transport_error(CATEGORY_TIMEOUT, exc, method) from exc
        except httpx.ConnectError as exc:
            raise self._transport_error(CATEGORY_NETWORK, exc, method) from exc
        except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
            # A bad/missing scheme is a permanent configuration error, not a
            # retryable transport outage.
            raise self._invalid_url(exc, method) from exc
        except (httpx.ReadError, httpx.WriteError, httpx.ProtocolError) as exc:
            raise self._transport_error(CATEGORY_TRANSPORT, exc, method) from exc
        except httpx.TransportError as exc:
            raise self._transport_error(CATEGORY_TRANSPORT, exc, method) from exc

        try:
            raw = self._read_body(response)
        finally:
            response.close()

        status = response.status_code
        payload, text = self._decode(raw)
        if 200 <= status < 300:
            return TransportResponse(
                status_code=status, payload=payload, text=text, raw_bytes=len(raw)
            )
        raise self._http_error(status, payload, text, method)

    def _read_body(self, response: httpx.Response) -> bytes:
        """Streamed read with an aborting byte cap (never unbounded buffering).

        The streaming path is the production path: bytes are counted as they
        arrive and the read is ABORTED the moment the cap is exceeded, so a
        huge body is never pulled into memory.

        Some test transports (and any response the HTTP layer already buffered)
        have consumed their stream by construction; those are still explicitly
        size-checked so the cap is enforced on every path.
        """
        if response.is_stream_consumed:
            body = response.content
            if len(body) > self.max_response_bytes:
                raise SupervisorRunError(
                    "provider response exceeded the maximum allowed size",
                    category=CATEGORY_RESPONSE_TOO_LARGE,
                    http_status=response.status_code,
                    sanitized_message=(
                        f"response exceeded {self.max_response_bytes} bytes"
                    ),
                )
            return body

        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_raw():
                if not chunk:
                    continue
                total += len(chunk)
                if total > self.max_response_bytes:
                    raise SupervisorRunError(
                        "provider response exceeded the maximum allowed size",
                        category=CATEGORY_RESPONSE_TOO_LARGE,
                        http_status=response.status_code,
                        sanitized_message=(
                            f"response exceeded {self.max_response_bytes} bytes"
                        ),
                    )
                chunks.append(chunk)
        except SupervisorRunError:
            raise
        except httpx.TransportError as exc:
            raise self._transport_error(CATEGORY_TRANSPORT, exc, "READ") from exc
        return b"".join(chunks)

    # -- response helpers -------------------------------------------------- #

    @staticmethod
    def _decode(raw: bytes) -> tuple[Any, str]:
        if not raw:
            return None, ""
        text = raw.decode("utf-8", "replace")
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            payload = None
        return payload, text

    @staticmethod
    def _message_from_payload(payload: Any, text: str) -> tuple[str, Optional[str]]:
        """Extract a sanitized provider message + error code when present."""
        code: Optional[str] = None
        message = ""
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or error.get("detail") or "")
                code = error.get("code") or error.get("type") or error.get("status")
            elif isinstance(error, str):
                message = error
            if not message:
                message = str(payload.get("message") or "")
            if code is None:
                code = payload.get("code") or payload.get("type")
        if not message:
            message = text
        return sanitize_diagnostic(message), (str(code) if code else None)

    def _http_error(
        self, status: int, payload: Any, text: str, operation: str
    ) -> SupervisorRunError:
        message, code = self._message_from_payload(payload, text)
        return SupervisorRunError(
            f"provider returned HTTP {status}",
            operation=operation,
            category=category_for_http_status(status),
            http_status=status,
            provider_error_code=code,
            sanitized_message=message or f"HTTP {status}",
        )

    # -- public API -------------------------------------------------------- #

    def post_json(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        json_body: Any = None,
        timeout_sec: Optional[int] = None,
    ) -> TransportResponse:
        return self._send(
            "POST",
            url,
            headers=headers,
            json_body=json_body,
            timeout_sec=timeout_sec,
        )

    def get_json(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        timeout_sec: Optional[int] = None,
    ) -> TransportResponse:
        return self._send("GET", url, headers=headers, timeout_sec=timeout_sec)

