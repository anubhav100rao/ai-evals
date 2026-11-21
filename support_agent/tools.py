"""Mock ShopNova backend + the 16 tools the support agent can call.

Every agent run gets its own `Backend` (deep copy of the seed data), so refunds and cancellations
in one eval example never leak into another. Each tool calls one or more simulated internal
services (see services.py), so a TOOL span contains HTTP client spans, and search_policy
contains a full RETRIEVER -> EMBEDDING -> RERANKER pipeline (see retrieval.py).
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Awaitable, Callable

from .retrieval import PolicyIndex
from .services import UpstreamError, api_call, call_with_retry

DATA_DIR = Path(__file__).parent / "data"

# The fictional "today" — fixed so return-window maths is deterministic for evals.
TODAY = date(2026, 11, 20)

REFUND_LIMIT = 200.00
RETURN_WINDOW_DAYS = {"electronics": 15}
DEFAULT_RETURN_WINDOW_DAYS = 30
DAMAGE_REPORT_WINDOW_DAYS = 30


def _load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text())


SEED_CUSTOMERS: list[dict] = _load("customers.json")
SEED_ORDERS: list[dict] = _load("orders.json")
POLICIES: list[dict] = _load("policies.json")
PRODUCTS: dict[str, dict] = {p["sku"]: p for p in _load("products.json")}
SHIPMENTS: dict[str, dict] = _load("shipments.json")
SEED_PAYMENTS: dict[str, dict] = _load("payments.json")
CRM: dict = _load("crm.json")
POLICY_INDEX = PolicyIndex(POLICIES)

_ticket_counter = itertools.count(5001)
_refund_counter = itertools.count(9001)
_replacement_counter = itertools.count(7001)


class ToolError(Exception):
    """A business-rule failure the agent should see and react to."""


class Backend:
    def __init__(self) -> None:
        self.customers = {c["customer_id"]: c for c in copy.deepcopy(SEED_CUSTOMERS)}
        self.orders = {o["order_id"]: o for o in copy.deepcopy(SEED_ORDERS)}
        self.payments = copy.deepcopy(SEED_PAYMENTS)
        self.tickets = copy.deepcopy(CRM["tickets"])

    # ------------------------------------------------------------- customers / CRM
    async def get_customer_profile(self, customer_id: str) -> dict:
        async with api_call("crm", "GET", f"/customers/{customer_id}"):
            c = self.customers.get(customer_id)
        if not c:
            raise ToolError(f"No customer with id {customer_id}")
        async with api_call("oms", "GET", f"/orders?customer_id={customer_id}"):
            orders = [
                {"order_id": o["order_id"], "status": o["status"], "order_date": o["order_date"], "total": o["total"],
                 "items": [i["name"] for i in o["items"]]}
                for o in self.orders.values() if o["customer_id"] == customer_id
            ]
        return {**c, "orders": orders}

    async def get_ticket_history(self, customer_id: str) -> dict:
        async with api_call("ticketing", "GET", f"/tickets?requester={customer_id}"):
            tickets = [t for t in self.tickets if t["customer_id"] == customer_id]
        return {"customer_id": customer_id, "ticket_count": len(tickets), "tickets": tickets}

    async def get_loyalty_account(self, customer_id: str) -> dict:
        async with api_call("crm", "GET", f"/loyalty/{customer_id}"):
            acct = CRM["loyalty"].get(customer_id)
        if not acct:
            raise ToolError(f"No loyalty account for {customer_id}")
        return {"customer_id": customer_id, **acct, "points_value_usd": round(acct["points"] / 100, 2),
                "rules": CRM["loyalty_rules"]}

    # ------------------------------------------------------------- orders
    async def lookup_order(self, order_id: str) -> dict:
        oid = _norm_order_id(order_id)
        async with api_call("oms", "GET", f"/orders/{oid}", status=200 if oid in self.orders else 404):
            o = self.orders.get(oid)
        if not o:
            raise ToolError(f"Order {order_id} not found")
        # NOTE: the backend does NOT enforce ownership — that is the agent's job (and an eval).
        return copy.deepcopy(o)

    async def track_shipment(self, order_id: str) -> dict:
        o = await self.lookup_order(order_id)
        tn = o.get("tracking_number")
        if not tn:
            return {"order_id": o["order_id"], "status": o["status"], "message": "No shipment yet — order not shipped."}
        carrier = o["carrier"]
        try:
            await call_with_retry(carrier, "GET", f"/track/v1/shipments/{tn.replace(' ', '')}")
        except UpstreamError as e:
            raise ToolError(f"Carrier API unavailable: {e}") from e
        s = SHIPMENTS.get(tn, {})
        return {"order_id": o["order_id"], "carrier": carrier, "tracking_number": tn, **s}

    async def check_return_eligibility(self, order_id: str, sku: str | None = None, reason: str = "unwanted") -> dict:
        o = await self.lookup_order(order_id)
        if o["status"] != "delivered":
            return {"eligible": False, "reason": f"Order is '{o['status']}', only delivered orders can be returned."}
        items = [i for i in o["items"] if sku is None or i["sku"] == sku]
        if not items:
            raise ToolError(f"SKU {sku} is not part of order {o['order_id']}")
        days = (TODAY - date.fromisoformat(o["delivered_date"])).days
        damaged = reason.lower() in {"damaged", "defective", "broken"}
        results = []
        for it in items:
            if damaged:
                ok = days <= DAMAGE_REPORT_WINDOW_DAYS
                why = (f"Damaged-item claim within {DAMAGE_REPORT_WINDOW_DAYS} days ({days} days since delivery)."
                       if ok else f"Damage reported {days} days after delivery (limit {DAMAGE_REPORT_WINDOW_DAYS}).")
            elif it["final_sale"]:
                ok, why = False, "Final-sale/clearance item: not returnable."
            else:
                window = RETURN_WINDOW_DAYS.get(it["category"], DEFAULT_RETURN_WINDOW_DAYS)
                ok = days <= window
                why = f"{it['category']} return window is {window} days; delivered {days} days ago."
            results.append({"sku": it["sku"], "name": it["name"], "price": it["price"], "eligible": ok, "reason": why})
        refundable = round(sum(r["price"] for r in results if r["eligible"]), 2)
        return {"order_id": o["order_id"], "days_since_delivery": days, "items": results,
                "eligible": any(r["eligible"] for r in results), "refundable_amount": refundable,
                "needs_escalation": refundable > REFUND_LIMIT}

    async def cancel_order(self, order_id: str) -> dict:
        oid = _norm_order_id(order_id)
        o = self.orders.get(oid)
        if not o:
            raise ToolError(f"Order {order_id} not found")
        if o["status"] != "processing":
            async with api_call("oms", "POST", f"/orders/{oid}/cancel", status=409):
                pass
            raise ToolError(f"Order is '{o['status']}' and can no longer be cancelled.")
        async with api_call("oms", "POST", f"/orders/{oid}/cancel"):
            o["status"] = "cancelled"
        async with api_call("payments", "POST", f"/payments/{oid}/void"):
            self.payments[oid]["status"] = "voided"
        return {"order_id": oid, "status": "cancelled", "refund_amount": o["total"],
                "note": "Payment authorization voided; full refund to original payment method in 5-7 business days."}

    async def update_shipping_address(self, order_id: str, new_address: str) -> dict:
        oid = _norm_order_id(order_id)
        o = self.orders.get(oid)
        if not o:
            raise ToolError(f"Order {order_id} not found")
        if o["status"] != "processing":
            raise ToolError(f"Order is '{o['status']}'; address can only be changed while processing.")
        async with api_call("address", "POST", "/v1/validate") as span:
            span.set_attribute("address.input", new_address)
            normalized = re.sub(r"\s+", " ", new_address).strip()
            if len(normalized) < 10 or not re.search(r"\d", normalized):
                raise ToolError("Address validation failed: please provide a full street address with number.")
        async with api_call("oms", "PATCH", f"/orders/{oid}/shipping-address"):
            old = o["shipping_address"]
            o["shipping_address"] = normalized
        return {"order_id": oid, "old_address": old, "new_address": normalized, "status": "updated",
                "address_validation": "verified"}

    async def create_replacement_order(self, order_id: str, sku: str, reason: str = "damaged") -> dict:
        o = await self.lookup_order(order_id)
        item = next((i for i in o["items"] if i["sku"] == sku), None)
        if not item:
            raise ToolError(f"SKU {sku} is not part of order {o['order_id']}")
        elig = await self.check_return_eligibility(o["order_id"], sku, "damaged")
        if not elig["eligible"]:
            raise ToolError("Replacement not allowed: " + elig["items"][0]["reason"])
        inv = await self.check_inventory(sku)
        if inv["total_available"] == 0:
            raise ToolError(f"{item['name']} is out of stock everywhere; restock date {inv['restock_date'] or 'unknown'}.")
        wh = max(inv["by_warehouse"], key=inv["by_warehouse"].get)
        async with api_call("oms", "POST", "/orders/replacements"):
            rid = f"ORD-R{next(_replacement_counter)}"
        return {"replacement_order_id": rid, "original_order_id": o["order_id"], "sku": sku, "item": item["name"],
                "ship_from": wh, "charge": 0.0, "shipping_method": "express", "status": "processing",
                "return_required": item["price"] >= 50}

    # ------------------------------------------------------------- payments
    async def get_payment_status(self, order_id: str) -> dict:
        oid = _norm_order_id(order_id)
        async with api_call("payments", "GET", f"/payments?order_id={oid}", status=200 if oid in self.payments else 404):
            p = self.payments.get(oid)
        if not p:
            raise ToolError(f"No payment record for {order_id}")
        return {"order_id": oid, **copy.deepcopy(p)}

    async def issue_refund(self, order_id: str, amount: float, reason: str) -> dict:
        o = await self.lookup_order(order_id)
        amount = float(amount)
        if amount <= 0:
            raise ToolError("Refund amount must be positive")
        if amount > REFUND_LIMIT:
            async with api_call("payments", "POST", f"/refunds?order_id={o['order_id']}", status=403):
                pass
            raise ToolError(f"Refunds above ${REFUND_LIMIT:.0f} cannot be issued by support agents; escalate instead.")
        if amount > o["total"]:
            raise ToolError(f"Refund ${amount:.2f} exceeds order total ${o['total']:.2f}")
        async with api_call("payments", "POST", f"/refunds?order_id={o['order_id']}"):
            rid = f"RF-{next(_refund_counter)}"
            pay = self.payments[o["order_id"]]
            pay["refunds"].append({"refund_id": rid, "amount": round(amount, 2), "reason": reason, "status": "pending"})
        return {"refund_id": rid, "order_id": o["order_id"], "amount": round(amount, 2), "status": "approved",
                "payment_method": pay["method"], "eta": "5-7 business days to original payment method"}

    # ------------------------------------------------------------- catalog / inventory / marketing
    async def search_products(self, query: str, max_results: int = 3) -> list[dict]:
        async with api_call("catalog", "GET", f"/search?q={query}") as span:
            words = set(re.findall(r"[a-z0-9]+", query.lower()))
            scored = []
            for p in PRODUCTS.values():
                hay = f"{p['name']} {p['category']} {p['description']}".lower()
                score = sum(1 for w in words if len(w) > 2 and w in hay)
                if score:
                    scored.append((score, p))
            scored.sort(key=lambda x: (-x[0], -x[1]["rating"]))
            span.set_attribute("search.hits", len(scored))
        return [{"sku": p["sku"], "name": p["name"], "price": p["price"], "rating": p["rating"],
                 "description": p["description"], "in_stock": sum(p["stock"].values()) > 0}
                for _, p in scored[: int(max_results)]]

    async def check_inventory(self, sku: str, warehouse: str | None = None) -> dict:
        async with api_call("inventory", "GET", f"/stock/{sku}"):
            p = PRODUCTS.get(str(sku).upper())
        if not p:
            raise ToolError(f"Unknown SKU {sku}")
        stock = p["stock"] if not warehouse else {warehouse: p["stock"].get(warehouse, 0)}
        return {"sku": p["sku"], "name": p["name"], "by_warehouse": stock, "total_available": sum(stock.values()),
                "restock_date": p["restock_date"]}

    async def get_active_promotions(self) -> list[dict]:
        async with api_call("marketing", "GET", f"/promotions?active_on={TODAY.isoformat()}"):
            return [p for p in CRM["promotions"] if p["valid_from"] <= TODAY.isoformat() <= p["valid_to"]]

    # ------------------------------------------------------------- knowledge base
    async def search_policy(self, query: str) -> list[dict]:
        docs = await POLICY_INDEX.search(query)
        return [{"id": d["id"], "title": d["title"], "body": d["body"], "score": d["score"]} for d in docs]

    # ------------------------------------------------------------- escalation
    async def escalate_to_human(self, reason: str, priority: str = "normal", order_id: str | None = None) -> dict:
        async with api_call("ticketing", "POST", "/tickets") as span:
            queue = _route_queue(reason)
            span.set_attribute("ticket.queue", queue)
            span.set_attribute("ticket.priority", priority)
            tid = f"TCK-{next(_ticket_counter)}"
        return {"ticket_id": tid, "queue": queue, "priority": priority, "order_id": order_id,
                "sla": "A specialist will reply within 24 hours."}


def _norm_order_id(order_id: str) -> str:
    s = str(order_id).strip().upper()
    if re.fullmatch(r"\d+", s):
        s = f"ORD-{s}"
    return s


def _route_queue(reason: str) -> str:
    r = reason.lower()
    if "refund" in r:
        return "refunds"
    if "warranty" in r or "defect" in r:
        return "warranty"
    if "email" in r or "account" in r or "security" in r:
        return "account-security"
    if "legal" in r or "sue" in r or "lawyer" in r:
        return "legal"
    return "general"


def _s(desc: str, **props) -> dict:
    return {"type": "string", "description": desc, **props}


_ORDER = {"order_id": _s("ShopNova order id, e.g. ORD-1002")}
_CUSTOMER = {"customer_id": _s("Customer id, e.g. C001")}

# Single source of truth for tools: JSON schema used for native function calling (OpenRouter),
# for the text protocol (LibreChat), and recorded on TOOL spans in Phoenix.
TOOL_SPECS: dict[str, dict] = {
    "get_customer_profile": {"service": "crm", "description": "Authenticated customer's profile and list of their orders.",
                             "properties": _CUSTOMER, "required": ["customer_id"]},
    "get_ticket_history": {"service": "helpdesk", "description": "Customer's previous support tickets (repeat contacts, past resolutions).",
                           "properties": _CUSTOMER, "required": ["customer_id"]},
    "get_loyalty_account": {"service": "crm", "description": "Loyalty points balance, tier, expiring points and redemption rules.",
                            "properties": _CUSTOMER, "required": ["customer_id"]},
    "lookup_order": {"service": "oms", "description": "Order details: status, items, dates, carrier, tracking, address, owner customer_id.",
                     "properties": _ORDER, "required": ["order_id"]},
    "track_shipment": {"service": "carrier", "description": "Live carrier tracking events and ETA for a shipped/delivered order.",
                       "properties": _ORDER, "required": ["order_id"]},
    "check_return_eligibility": {
        "service": "oms", "description": "Whether items in a delivered order can be returned/refunded and the refundable amount.",
        "properties": {**_ORDER, "sku": _s("Optional SKU to check a single item"),
                       "reason": _s("Why the customer returns it", enum=["unwanted", "damaged"])},
        "required": ["order_id"]},
    "issue_refund": {
        "service": "payments", "description": "Refund to the original payment method. Max $200 per order.",
        "properties": {**_ORDER, "amount": {"type": "number", "description": "Amount in USD"}, "reason": _s("Refund reason")},
        "required": ["order_id", "amount", "reason"]},
    "get_payment_status": {"service": "payments", "description": "Payment method, capture status and refund history for an order.",
                           "properties": _ORDER, "required": ["order_id"]},
    "cancel_order": {"service": "oms", "description": "Cancel an order that is still processing (voids payment).",
                     "properties": _ORDER, "required": ["order_id"]},
    "update_shipping_address": {
        "service": "oms", "description": "Validate and change the shipping address of a processing order.",
        "properties": {**_ORDER, "new_address": _s("Full new street address")}, "required": ["order_id", "new_address"]},
    "create_replacement_order": {
        "service": "oms", "description": "Free replacement for a damaged/defective item if in stock.",
        "properties": {**_ORDER, "sku": _s("SKU of the damaged item"), "reason": _s("e.g. damaged")},
        "required": ["order_id", "sku"]},
    "search_products": {
        "service": "catalog", "description": "Search the product catalog (alternatives, recommendations).",
        "properties": {"query": _s("Search text"), "max_results": {"type": "integer", "description": "Default 3"}},
        "required": ["query"]},
    "check_inventory": {
        "service": "inventory", "description": "Stock per warehouse and restock date for a SKU.",
        "properties": {"sku": _s("Product SKU, e.g. EAR-200"),
                       "warehouse": _s("Optional warehouse code", enum=["BLR", "SFO", "LON", "MEX", "STO"])},
        "required": ["sku"]},
    "get_active_promotions": {"service": "marketing", "description": "Currently active promotions/deals.",
                              "properties": {}, "required": []},
    "search_policy": {"service": "kb", "description": "Search the policy knowledge base (returns, refunds, shipping, warranty, account, privacy, loyalty...).",
                      "properties": {"query": _s("Natural-language policy question")}, "required": ["query"]},
    "escalate_to_human": {
        "service": "helpdesk", "description": "Create a ticket for a human specialist team.",
        "properties": {"reason": _s("Why a human is needed"), "priority": _s("Ticket priority", enum=["low", "normal", "high"]),
                       "order_id": _s("Related order id, if any")},
        "required": ["reason"]},
}


def tool_schema(name: str) -> dict:
    spec = TOOL_SPECS[name]
    return {"type": "object", "properties": spec["properties"], "required": spec["required"]}


def openai_tools() -> list[dict]:
    """Tools in OpenAI function-calling format."""
    return [{"type": "function", "function": {"name": n, "description": s["description"], "parameters": tool_schema(n)}}
            for n, s in TOOL_SPECS.items()]


def get_tool(backend: Backend, name: str) -> Callable[..., Awaitable[Any]] | None:
    if name not in TOOL_SPECS:
        return None
    return getattr(backend, name)
