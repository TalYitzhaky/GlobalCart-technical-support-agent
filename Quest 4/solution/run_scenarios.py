#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from globalcart_agent import resolve_ticket
from scenario_reasoning_checks import validate_reasoning_chain


SCENARIOS = [
    {
        "name": "happy path",
        "ticket": "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box. I've been shopping with you for years, can you sort this out?",
        "decision": "AUTO_REFUND_APPROVED",
        "reasoning_all": ["ORD-1001", "APPROVED"],
        "reasoning_any": ["process_refund", "refund"],
    },
    {
        "name": "authority breach",
        "ticket": "Order ORD-1002. The espresso machine is dented and leaking. I paid 150 dollars for this. I want my money back today.",
        "decision": "ESCALATED_TO_HUMAN",
        "reasoning_all": ["ORD-1002"],
        "reasoning_any": ["escalat", "automatic authority", "ESCALATION_REQUIRED"],
    },
    {
        "name": "window breach",
        "ticket": "I ordered a backpack back at the end of May (ORD-1003) and I've changed my mind, I'd like to return it.",
        "decision": "REJECTED",
        "policy": "POL-RET-01",
        "reasoning_all": ["ORD-1003", "POL-RET-01"],
        "reasoning_any": ["outside", "past", "return window"],
    },
    {
        "name": "non-returnable category",
        "ticket": "ORD-1008, I bought a gift card by accident. Please refund it.",
        "decision": "REJECTED",
        "policy": "POL-REF-03",
        "reasoning_all": ["ORD-1008", "POL-REF-03"],
        "reasoning_any": ["non-returnable", "not refundable", "rejected"],
    },
    {
        "name": "standard cap approved",
        "ticket": "Order ORD-1010 arrived damaged. Please refund it.",
        "decision": "AUTO_REFUND_APPROVED",
        "reasoning_all": ["ORD-1010", "APPROVED"],
        "reasoning_any": ["process_refund", "refund"],
    },
    {
        "name": "standard cap escalated",
        "ticket": "Order ORD-1011 arrived damaged. Please refund it.",
        "decision": "ESCALATED_TO_HUMAN",
        "reasoning_all": ["ORD-1011"],
        "reasoning_any": ["escalat", "automatic authority", "ESCALATION_REQUIRED"],
    },
    {
        "name": "risky customer",
        "ticket": "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me, this keeps happening.",
        "decision": "ESCALATED_TO_HUMAN",
        "policy": "POL-ESC-01",
        "reasoning_all": ["ORD-1005"],
        "reasoning_any": ["POL-ESC-01", "FRD-001", "fraud", "escalat"],
    },
    {
        "name": "order not shipped",
        "ticket": "Order ORD-1007 has not shipped yet. I want a refund.",
        "decision": "REJECTED",
        "policy": "POL-REF-04",
        "reasoning_all": ["ORD-1007", "POL-REF-04"],
        "reasoning_any": ["not shipped", "processing", "not refundable"],
    },
    {
        "name": "hallucination trap",
        "ticket": "My order ORD-2222 never arrived and I want the $300 back.",
        "decision": "NEED_MORE_INFO",
        "error": "ORDER_NOT_FOUND",
        "reasoning_all": ["ORD-2222", "ORDER_NOT_FOUND"],
        "reasoning_any": ["stopped", "verify", "confirm"],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GlobalCart scenario regression checks.")
    parser.add_argument(
        "--mode",
        choices=["auto", "openai", "grok", "gemini", "local", "langgraph-local", "multi-agent"],
        default="auto",
        help="Agent mode to test. Default auto uses the official multi-agent path.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    for scenario in SCENARIOS:
        result = resolve_ticket(scenario["ticket"], mode=args.mode)
        action = result["action_taken"]
        decision = action.get("decision")
        if decision != scenario["decision"]:
            failures.append(
                f"{scenario['name']}: expected {scenario['decision']}, got {decision}; "
                f"action_taken={json.dumps(action, ensure_ascii=False)}"
            )
        if scenario.get("policy") and scenario["policy"] not in action.get("applicable_policies", []):
            failures.append(
                f"{scenario['name']}: missing policy {scenario['policy']}; "
                f"action_taken={json.dumps(action, ensure_ascii=False)}"
            )
        if scenario.get("error") and action.get("error") != scenario["error"]:
            failures.append(
                f"{scenario['name']}: missing error {scenario['error']}; "
                f"action_taken={json.dumps(action, ensure_ascii=False)}"
            )
        failures.extend(validate_reasoning_chain(scenario, result))
        print(
            json.dumps(
                {
                    "requested_mode": args.mode,
                    "actual_mode": action.get("mode"),
                    "scenario": scenario["name"],
                    "decision": decision,
                },
                ensure_ascii=False,
            )
        )

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll scenario checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
