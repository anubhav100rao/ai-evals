"""Simulated internal microservices.

Every backend call made by a tool goes through `api_call`, which emits an OpenTelemetry CLIENT
span (HTTP method, URL, status code, service, latency) under the current TOOL span. Latency is
deterministic (hash of the request) so runs are reproducible, and a couple of carrier APIs are
deliberately flaky (first attempt times out, retry succeeds) so traces show retries/errors.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

SERVICES: dict[str, dict] = {
    "oms":        {"host": "oms.internal.shopnova",        "base_ms": 60,  "name": "order-management"},
    "payments":   {"host": "payments.internal.shopnova",   "base_ms": 110, "name": "payments"},
    "crm":        {"host": "crm.internal.shopnova",        "base_ms": 90,  "name": "crm"},
    "catalog":    {"host": "catalog.internal.shopnova",    "base_ms": 45,  "name": "catalog-search"},
    "inventory":  {"host": "inventory.internal.shopnova",  "base_ms": 70,  "name": "inventory"},
    "ticketing":  {"host": "helpdesk.internal.shopnova",   "base_ms": 150, "name": "helpdesk"},
    "address":    {"host": "address-validation.shopnova",  "base_ms": 180, "name": "address-validation"},
    "marketing":  {"host": "promo.internal.shopnova",      "base_ms": 40,  "name": "promotions"},
    "BlueDart":   {"host": "api.bluedart.example",         "base_ms": 320, "name": "carrier-bluedart"},
    "UPS":        {"host": "onlinetools.ups.example",      "base_ms": 260, "name": "carrier-ups"},
    "FedEx":      {"host": "apis.fedex.example",           "base_ms": 280, "name": "carrier-fedex"},
    "DHL":        {"host": "api-eu.dhl.example",           "base_ms": 410, "name": "carrier-dhl", "flaky": True},
    "Royal Mail": {"host": "api.royalmail.example",        "base_ms": 350, "name": "carrier-royalmail", "flaky": True},
}

_tracer = trace.get_tracer("shopnova.services")


class UpstreamError(Exception):
    pass


def _latency(service: str, path: str) -> float:
    base = SERVICES[service]["base_ms"]
    jitter = int(hashlib.md5(f"{service}{path}".encode()).hexdigest()[:4], 16) % base
    return (base + jitter) / 1000


@asynccontextmanager
async def api_call(service: str, method: str, path: str, *, attempt: int = 1, status: int = 200):
    """Emit an HTTP client span and simulate network latency for a backend call."""
    svc = SERVICES[service]
    url = f"https://{svc['host']}{path}"
    with _tracer.start_as_current_span(f"{method} {svc['name']}", kind=SpanKind.CLIENT) as span:
        span.set_attribute("http.request.method", method)
        span.set_attribute("url.full", url)
        span.set_attribute("server.address", svc["host"])
        span.set_attribute("peer.service", svc["name"])
        span.set_attribute("http.request.resend_count", attempt - 1)
        flaky_fail = svc.get("flaky") and attempt == 1
        await asyncio.sleep(_latency(service, path) * (3 if flaky_fail else 1))
        if flaky_fail:
            span.set_attribute("http.response.status_code", 504)
            span.set_attribute("error.type", "GatewayTimeout")
            err = UpstreamError(f"{svc['name']} timed out after {int(_latency(service, path) * 3000)}ms")
            span.record_exception(err)
            span.set_status(Status(StatusCode.ERROR, str(err)))
            raise err
        span.set_attribute("http.response.status_code", status)
        if status >= 400:
            span.set_status(Status(StatusCode.ERROR, f"HTTP {status}"))
        yield span


async def call_with_retry(service: str, method: str, path: str, *, status: int = 200, retries: int = 2):
    """api_call with retry + exponential backoff (visible as sibling spans in the trace)."""
    for attempt in range(1, retries + 2):
        try:
            async with api_call(service, method, path, attempt=attempt, status=status):
                return
        except UpstreamError:
            if attempt > retries:
                raise
            await asyncio.sleep(0.05 * 2 ** attempt)
