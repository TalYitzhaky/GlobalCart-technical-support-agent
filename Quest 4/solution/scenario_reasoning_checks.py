from __future__ import annotations

import json
from typing import Any


MAX_REASONING_ITEM_CHARS = 900
REFUND_SUCCESS_CLAIMS = [
    "refund approved",
    "approved a refund",
    "refund has been issued",
    "refund was issued",
    "returned approved",
    "auto_refund_approved",
]


def validate_reasoning_chain(scenario: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Validate audit-style reasoning without requiring exact wording."""
    failures: list[str] = []
    action = result.get("action_taken", {})
    chain = result.get("reasoning_chain")
    if not isinstance(chain, list) or not chain:
        return [_failure(scenario, "reasoning_chain must be a non-empty list", chain, action)]

    for index, item in enumerate(chain):
        if not isinstance(item, str) or not item.strip():
            failures.append(_failure(scenario, f"reasoning_chain[{index}] must be a non-empty string", chain, action))
        elif len(item) > MAX_REASONING_ITEM_CHARS:
            failures.append(_failure(scenario, f"reasoning_chain[{index}] is too long", chain, action))

    reasoning_text = "\n".join(str(item) for item in chain).lower()
    order_id = action.get("order_id")
    if order_id and str(order_id).lower() not in reasoning_text:
        failures.append(_failure(scenario, f"reasoning missing order id {order_id}", chain, action))

    for token in scenario.get("reasoning_all", []):
        if token.lower() not in reasoning_text:
            failures.append(_failure(scenario, f"reasoning missing token {token}", chain, action))

    any_tokens = scenario.get("reasoning_any", [])
    if any_tokens and not any(token.lower() in reasoning_text for token in any_tokens):
        failures.append(_failure(scenario, f"reasoning missing one of {any_tokens}", chain, action))

    for token in scenario.get("reasoning_absent", []):
        if token.lower() in reasoning_text:
            failures.append(_failure(scenario, f"reasoning contains forbidden token {token}", chain, action))

    refund_status = action.get("refund_status")
    if refund_status and refund_status != "APPROVED":
        for phrase in REFUND_SUCCESS_CLAIMS:
            if phrase in reasoning_text:
                failures.append(_failure(scenario, f"reasoning claims refund success via {phrase}", chain, action))

    if action.get("decision") == "AUTO_REFUND_APPROVED":
        if "approved" not in reasoning_text or "refund" not in reasoning_text:
            failures.append(_failure(scenario, "approved case reasoning must mention approved refund evidence", chain, action))

    if action.get("decision") == "ESCALATED_TO_HUMAN" and "escalat" not in reasoning_text:
        failures.append(_failure(scenario, "escalated case reasoning must mention escalation", chain, action))

    if action.get("error") and str(action["error"]).lower() not in reasoning_text:
        failures.append(_failure(scenario, f"reasoning missing error {action['error']}", chain, action))

    return failures


def _failure(scenario: dict[str, Any], label: str, chain: Any, action: dict[str, Any]) -> str:
    return (
        f"{scenario['name']}: {label}; "
        f"reasoning_chain={json.dumps(chain, ensure_ascii=False)}; "
        f"action_taken={json.dumps(action, ensure_ascii=False)}"
    )
