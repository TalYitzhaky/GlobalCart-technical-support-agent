#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from globalcart_agent import resolve_ticket


SCENARIOS = [
    {
        "name": "happy path",
        "ticket": "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. I've been shopping with you for years, can you sort this out?",
        "decision": "AUTO_REFUND_APPROVED",
    },
    {
        "name": "authority breach",
        "ticket": "Order ORD-1002. The espresso machine is dented and leaking. I paid 150 dollars for this. I want my money back today.",
        "decision": "ESCALATED_TO_HUMAN",
    },
    {
        "name": "window breach",
        "ticket": "I ordered a backpack back at the end of May (ORD-1003) and I've changed my mind, I'd like to return it.",
        "decision": "REJECTED",
        "policy": "POL-RET-01",
    },
    {
        "name": "non-returnable category",
        "ticket": "ORD-1008, I bought a gift card by accident. Please refund it.",
        "decision": "REJECTED",
        "policy": "POL-REF-03",
    },
    {
        "name": "standard cap approved",
        "ticket": "Order ORD-1010 arrived damaged. Please refund it.",
        "decision": "AUTO_REFUND_APPROVED",
    },
    {
        "name": "standard cap escalated",
        "ticket": "Order ORD-1011 arrived damaged. Please refund it.",
        "decision": "ESCALATED_TO_HUMAN",
    },
    {
        "name": "risky customer",
        "ticket": "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me, this keeps happening.",
        "decision": "ESCALATED_TO_HUMAN",
        "policy": "POL-ESC-01",
    },
    {
        "name": "order not shipped",
        "ticket": "Order ORD-1007 has not shipped yet. I want a refund.",
        "decision": "REJECTED",
        "policy": "POL-REF-04",
    },
    {
        "name": "hallucination trap",
        "ticket": "My order ORD-2222 never arrived and I want the $300 back.",
        "decision": "NEED_MORE_INFO",
        "error": "ORDER_NOT_FOUND",
    },
]


def main() -> int:
    failures: list[str] = []
    for scenario in SCENARIOS:
        result = resolve_ticket(scenario["ticket"], mode="local")
        action = result["action_taken"]
        decision = action.get("decision")
        if decision != scenario["decision"]:
            failures.append(f"{scenario['name']}: expected {scenario['decision']}, got {decision}")
        if scenario.get("policy") and scenario["policy"] not in action.get("applicable_policies", []):
            failures.append(f"{scenario['name']}: missing policy {scenario['policy']}")
        if scenario.get("error") and action.get("error") != scenario["error"]:
            failures.append(f"{scenario['name']}: missing error {scenario['error']}")
        print(json.dumps({"scenario": scenario["name"], "decision": decision}, ensure_ascii=False))

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll scenario checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
