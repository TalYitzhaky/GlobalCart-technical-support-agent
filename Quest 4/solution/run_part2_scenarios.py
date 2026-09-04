#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from globalcart_agent import resolve_ticket
from globalcart_agent.multi_agent_provider import MULTI_AGENT_LLM_ENTRYPOINT
from globalcart_agent.multi_agent_workflow import _customer_response_is_safe


SCENARIOS = [
    {
        "name": "low-risk approved refund",
        "ticket": "Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked right out of the box.",
        "decision": "AUTO_REFUND_APPROVED",
        "must_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy", "process_refund"],
        "must_not_call": ["send_slack_alert"],
        "fraud_rules": [],
        "policies": ["POL-RET-02", "POL-REF-02"],
        "refund_status": "APPROVED",
        "escalation": False,
        "sentiment": "neutral",
        "tone_absent": ["urgent"],
    },
    {
        "name": "high-risk repeat-claim customer",
        "ticket": "This is Ronen, order ORD-1005. The tablet screen was smashed on arrival. Refund me, this keeps happening.",
        "decision": "ESCALATED_TO_HUMAN",
        "must_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy", "send_slack_alert"],
        "must_not_call": ["process_refund"],
        "fraud_rules": ["FRD-001", "FRD-002", "FRD-003", "FRD-004"],
        "policies": ["POL-RET-01", "POL-REF-01", "POL-ESC-01", "POL-ESC-02"],
        "refund_status": "NOT_ATTEMPTED",
        "escalation": True,
        "sentiment": "urgent",
        "tone_any": ["urgent", "direct"],
    },
    {
        "name": "refund above automatic authority",
        "ticket": "Order ORD-1002. The espresso machine is dented and leaking. I paid 150 dollars for this.",
        "decision": "ESCALATED_TO_HUMAN",
        "must_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy", "send_slack_alert"],
        "must_not_call": ["process_refund"],
        "fraud_rules": [],
        "policies": ["POL-RET-01", "POL-REF-01"],
        "refund_status": "NOT_ATTEMPTED",
        "escalation": True,
        "sentiment": "neutral",
        "tone_absent": ["urgent"],
    },
    {
        "name": "return-window rejection",
        "ticket": "I ordered a backpack back at the end of May (ORD-1003) and I've changed my mind, I'd like to return it.",
        "decision": "REJECTED",
        "must_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy"],
        "must_not_call": ["process_refund", "send_slack_alert"],
        "fraud_rules": [],
        "policies": ["POL-RET-01"],
        "refund_status": "NOT_ATTEMPTED",
        "escalation": False,
        "sentiment": "neutral",
        "tone_absent": ["urgent"],
    },
    {
        "name": "non-returnable category",
        "ticket": "ORD-1008, I bought a gift card by accident. Please refund it.",
        "decision": "REJECTED",
        "must_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy"],
        "must_not_call": ["process_refund", "send_slack_alert"],
        "fraud_rules": [],
        "policies": ["POL-REF-03"],
        "refund_status": "NOT_ATTEMPTED",
        "escalation": False,
        "sentiment": "concerned",
        "tone_any": ["sorry", "calm", "thanks"],
    },
    {
        "name": "missing order id",
        "ticket": "My package arrived broken and I need a refund.",
        "decision": "NEED_MORE_INFO",
        "must_call": [],
        "must_not_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy", "process_refund", "send_slack_alert"],
        "error": "ORDER_ID_MISSING",
        "escalation": False,
        "sentiment": "neutral",
    },
    {
        "name": "nonexistent order hallucination trap",
        "ticket": "My order ORD-2222 never arrived and I want the $300 back.",
        "decision": "NEED_MORE_INFO",
        "must_call": ["get_order_details"],
        "must_not_call": ["get_user_profile", "audit_fraud_risk", "check_return_policy", "process_refund", "send_slack_alert"],
        "error": "ORDER_NOT_FOUND",
        "escalation": False,
        "sentiment": "neutral",
    },
    {
        "name": "order not shipped policy error path",
        "ticket": "Order ORD-1007 has not shipped yet. I want a refund.",
        "decision": "REJECTED",
        "must_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy"],
        "must_not_call": ["process_refund", "send_slack_alert"],
        "policies": ["POL-REF-04"],
        "refund_status": "NOT_ATTEMPTED",
        "escalation": False,
        "sentiment": "neutral",
        "tone_absent": ["urgent"],
    },
    {
        "name": "escalation alert verification",
        "ticket": "Order ORD-1011 arrived damaged. Please refund it.",
        "decision": "ESCALATED_TO_HUMAN",
        "must_call": ["get_order_details", "get_user_profile", "audit_fraud_risk", "check_return_policy", "send_slack_alert"],
        "must_not_call": ["process_refund"],
        "fraud_rules": [],
        "policies": ["POL-RET-01", "POL-REF-01"],
        "refund_status": "NOT_ATTEMPTED",
        "escalation": True,
        "sentiment": "concerned",
        "tone_any": ["sorry", "calm", "thanks"],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Quest 4 Part 2 multi-agent scenario checks.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force MULTI_AGENT_LLM_PROVIDER=deterministic for a stable no-LLM baseline.",
    )
    args = parser.parse_args()

    original_provider = os.environ.get("MULTI_AGENT_LLM_PROVIDER")
    if args.deterministic:
        os.environ["MULTI_AGENT_LLM_PROVIDER"] = "deterministic"
    failures: list[str] = []
    try:
        for scenario in SCENARIOS:
            result = resolve_ticket(scenario["ticket"], mode="multi-agent")
            action = result["action_taken"]
            tools_called = action.get("tools_called", [])
            if action.get("decision") != scenario["decision"]:
                failures.append(_failure(scenario, "decision", action))
            for tool in scenario.get("must_call", []):
                if tool not in tools_called:
                    failures.append(_failure(scenario, f"missing tool {tool}", action))
            for tool in scenario.get("must_not_call", []):
                if tool in tools_called:
                    failures.append(_failure(scenario, f"unexpected tool {tool}", action))
            for rule_id in scenario.get("fraud_rules", []):
                if rule_id not in action.get("fraud_rule_ids", []):
                    failures.append(_failure(scenario, f"missing fraud rule {rule_id}", action))
            for policy_id in scenario.get("policies", []):
                if policy_id not in action.get("applicable_policies", []):
                    failures.append(_failure(scenario, f"missing policy {policy_id}", action))
            if scenario.get("refund_status") and action.get("refund_status") != scenario["refund_status"]:
                failures.append(_failure(scenario, f"refund_status expected {scenario['refund_status']}", action))
            if action.get("escalation_triggered") != scenario.get("escalation"):
                failures.append(_failure(scenario, f"escalation expected {scenario.get('escalation')}", action))
            if scenario.get("escalation") and not action.get("escalation_packet", {}).get("alert_id"):
                failures.append(_failure(scenario, "missing escalation alert id", action))
            if scenario.get("error") and action.get("error") != scenario["error"]:
                failures.append(_failure(scenario, f"error expected {scenario['error']}", action))
            if scenario.get("sentiment") and action.get("sentiment") != scenario["sentiment"]:
                failures.append(_failure(scenario, f"sentiment expected {scenario['sentiment']}", action))
            if action.get("agent3_response_mode") == "deterministic":
                response_text = result.get("customer_response", "").lower()
                if scenario.get("tone_any") and not any(phrase in response_text for phrase in scenario["tone_any"]):
                    failures.append(_failure(scenario, f"missing sentiment tone marker {scenario['tone_any']}", action))
                if scenario.get("tone_absent") and any(phrase in response_text for phrase in scenario["tone_absent"]):
                    failures.append(_failure(scenario, f"unexpected sentiment tone marker {scenario['tone_absent']}", action))
            execution = action.get("agent_execution", {})
            expected_agents = ["researcher", "decision_maker", "communications"]
            for agent in expected_agents:
                if agent not in execution:
                    failures.append(_failure(scenario, f"missing agent execution {agent}", action))
                elif execution[agent].get("entrypoint") != MULTI_AGENT_LLM_ENTRYPOINT:
                    failures.append(_failure(scenario, f"{agent} did not use multi_agent_provider entrypoint", action))
            print(
                json.dumps(
                    {
                        "scenario": scenario["name"],
                        "decision": action.get("decision"),
                        "sentiment": action.get("sentiment"),
                        "agent_execution": execution,
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        if original_provider is None:
            os.environ.pop("MULTI_AGENT_LLM_PROVIDER", None)
        else:
            os.environ["MULTI_AGENT_LLM_PROVIDER"] = original_provider

    if _unsafe_response_is_accepted():
        failures.append("unsafe customer response validator accepted a fabricated refund claim")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nAll Part 2 multi-agent scenario checks passed.")
    return 0


def _unsafe_response_is_accepted() -> bool:
    action = {
        "decision": "ESCALATED_TO_HUMAN",
        "refund_status": "NOT_ATTEMPTED",
        "refund_amount": 0.0,
        "escalation_triggered": True,
    }
    state = {"order_id": "ORD-1005", "refund": None}
    decision_packet = {"decision": "ESCALATED_TO_HUMAN", "reason": "Fraud audit requires human review."}
    return _customer_response_is_safe(
        "I approved your refund of 999.00 USD and it has already been issued.",
        action,
        state,
        decision_packet,
    )


def _failure(scenario, label, action):
    return f"{scenario['name']}: {label}; action_taken={json.dumps(action, ensure_ascii=False)}"


if __name__ == "__main__":
    raise SystemExit(main())
