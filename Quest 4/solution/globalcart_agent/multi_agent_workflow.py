from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import multi_agent_tools as mat

from .multi_agent_provider import MULTI_AGENT_LLM_ENTRYPOINT, call_multi_agent_llm
from .resolver import (
    DECISION_AUTO_REFUND_APPROVED,
    DECISION_ESCALATED,
    DECISION_NEED_MORE_INFO,
    DECISION_REJECTED,
    _base_output,
    _business_error_output,
    _call_tool,
    _canonical_decision,
    _has_error,
    _interaction_text,
    _response_contradicts_refund,
    _refund_reasoning,
    extract_order_id,
    infer_reason,
    infer_sentiment,
)


class MultiAgentState(TypedDict, total=False):
    ticket: str
    order_id: str | None
    reason: str
    sentiment: str
    order: dict[str, Any]
    user: dict[str, Any]
    fraud_report: dict[str, Any]
    investigation_report: dict[str, Any]
    policy: dict[str, Any]
    refund: dict[str, Any] | None
    decision_packet: dict[str, Any]
    escalation_packet: dict[str, Any]
    tools_called: list[str]
    agent_messages: list[dict[str, Any]]
    agent_execution: dict[str, Any]
    stop_reason: str
    error: dict[str, Any]
    reasoning_chain: list[str]
    action_taken: dict[str, Any]
    customer_response: str
    result: dict[str, Any]


def resolve_ticket_multi_agent(ticket: str) -> dict[str, Any]:
    """Resolve a ticket through the Quest #4 Part 2 multi-agent workflow."""
    graph = _build_graph()
    final_state = graph.invoke({"ticket": ticket, "tools_called": [], "agent_messages": []}, {"recursion_limit": 8})
    return final_state["result"]


def _build_graph():
    # Part 2 graph: a fixed three-agent handoff, no dynamic supervisor yet.
    builder = StateGraph(MultiAgentState)
    builder.add_node("researcher_fraud_auditor", _researcher_fraud_auditor)
    builder.add_node("decision_maker", _decision_maker)
    builder.add_node("communications_escalation_manager", _communications_escalation_manager)
    builder.add_edge(START, "researcher_fraud_auditor")
    builder.add_edge("researcher_fraud_auditor", "decision_maker")
    builder.add_edge("decision_maker", "communications_escalation_manager")
    builder.add_edge("communications_escalation_manager", END)
    return builder.compile()


AGENT1_PROMPT = """You are GlobalCart's Researcher & Fraud Auditor.

Investigate the ticket using only the verified order, customer, and fraud data
provided to you. Never invent order, customer, or fraud facts. Your job is to
produce a structured investigation report for the Operations Lead. You do not
make refund, policy, or escalation decisions.
"""

AGENT2_PROMPT = """You are GlobalCart's Decision Maker / Operations Lead.

You receive a verified investigation report and policy output. You may reason
about the operational outcome, but deterministic code will enforce authority,
canonical decision names, and refund execution. Never invent policy, fraud, or
refund facts. Do not claim a refund happened unless a verified refund result is
APPROVED.
"""

AGENT3_PROMPT = """You are GlobalCart's Communications & Escalation Manager.

You receive a verified operational decision. You may not change or reinterpret
that decision. Explain the result clearly and professionally to the customer.
Adapt the tone to the provided sentiment and sentiment_style_instruction.
Never claim a refund was issued unless the verified refund status is APPROVED.
Never invent policy, fraud, refund, order, customer, or escalation facts.
"""


def _agent1_llm_report(
    ticket: str,
    reason: str,
    sentiment: str,
    order: dict[str, Any],
    user: dict[str, Any],
    fraud_report: dict[str, Any],
) -> dict[str, Any]:
    return call_multi_agent_llm(
        agent_name="researcher",
        system_prompt=AGENT1_PROMPT,
        payload={
            "ticket": ticket,
            "deterministic_ticket_parse": {"order_id": order["order_id"], "reason": reason, "sentiment": sentiment},
            "verified_order": _compact(order),
            "verified_customer": _compact(user),
            "verified_fraud_report": _compact(fraud_report),
            "allowed_tools": ["get_order_details", "get_user_profile", "audit_fraud_risk"],
            "disallowed_decisions": ["refund_approval", "policy_decision", "escalation_execution"],
        },
        output_schema={
            "order_id": "ORD-####",
            "inferred_reason": "damaged_on_arrival | wrong_item | item_missing | late_delivery | changed_mind",
            "sentiment": "neutral | concerned | urgent",
            "investigation_status": "completed",
            "investigation_summary": "short summary based only on verified data",
            "requires_escalation": True,
            "fraud_risk_level": "low | medium | high",
            "matched_fraud_rule_ids": ["FRD-001"],
            "prior_fraud_flags": 0,
            "repeat_claims_in_window": 0,
        },
    )


def _validated_investigation_report(
    *,
    parsed: dict[str, Any] | None,
    ticket: str,
    order_id: str,
    reason: str,
    sentiment: str,
    order: dict[str, Any],
    user: dict[str, Any],
    fraud_report: dict[str, Any],
) -> dict[str, Any]:
    summary = ""
    if isinstance(parsed, dict) and isinstance(parsed.get("investigation_summary"), str):
        summary = parsed["investigation_summary"].strip()
    if not summary:
        summary = (
            f"Ticket reason appears to be {reason}; order {order_id} belongs to customer {user['user_id']} "
            f"and fraud audit risk is {fraud_report['risk_level']}."
        )
    return {
        "order_id": order_id,
        "inferred_reason": reason,
        "sentiment": sentiment,
        "order_facts": {
            "order_id": order["order_id"],
            "status": order.get("status"),
            "total_amount": order.get("total_amount"),
            "currency": order.get("currency"),
            "category": order.get("category"),
            "delivery_date": order.get("delivery_date"),
        },
        "customer_facts": {
            "user_id": user["user_id"],
            "tier": user.get("tier"),
            "initial_fraud_score": user.get("initial_fraud_score"),
            "prior_fraud_flags": user.get("prior_fraud_flags"),
        },
        "fraud_risk_level": fraud_report["risk_level"],
        "fraud_score": fraud_report["fraud_score"],
        "matched_fraud_rule_ids": fraud_report["matched_rule_ids"],
        "prior_fraud_flags": fraud_report["prior_fraud_flags"],
        "repeat_claims_in_window": fraud_report["repeat_claims_in_window"],
        "requires_escalation": fraud_report["requires_escalation"],
        "investigation_status": fraud_report.get("status", "completed"),
        "investigation_summary": summary,
        "ticket_excerpt": ticket[:240],
    }


def _agent2_llm_proposal(state: MultiAgentState, policy: dict[str, Any]) -> dict[str, Any]:
    return call_multi_agent_llm(
        agent_name="decision_maker",
        system_prompt=AGENT2_PROMPT,
        payload={
            "ticket": state["ticket"],
            "investigation_report": _compact(state.get("investigation_report")),
            "verified_order": _compact(state.get("order")),
            "verified_fraud_report": _compact(state.get("fraud_report")),
            "verified_policy": _compact(policy),
            "deterministic_authority_rules": [
                "If policy is not eligible, reject and do not execute refund.",
                "If fraud requires escalation, escalate and do not execute refund.",
                "If amount exceeds automatic cap, escalate and do not execute refund.",
                "Only process_refund may create an approved refund result.",
            ],
            "allowed_tools": ["check_return_policy", "process_refund"],
        },
        output_schema={
            "proposed_decision": "AUTO_REFUND_APPROVED | ESCALATED_TO_HUMAN | REJECTED | NEED_MORE_INFO",
            "refund_execution_requested": False,
            "decision_summary": "short operational rationale",
            "guardrail_notes": ["short note"],
        },
    )


def _proposed_decision(llm_packet: dict[str, Any]) -> str | None:
    parsed = llm_packet.get("parsed")
    if not isinstance(parsed, dict) or "proposed_decision" not in parsed:
        return None
    return _canonical_decision(parsed["proposed_decision"])


def _proposed_summary(llm_packet: dict[str, Any]) -> str | None:
    parsed = llm_packet.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("decision_summary"), str):
        return parsed["decision_summary"].strip() or None
    return None


def _execution_packet(llm_packet: dict[str, Any] | None, skipped: bool = False) -> dict[str, Any]:
    if skipped:
        return {"mode": "skipped", "provider": None, "model": None, "entrypoint": MULTI_AGENT_LLM_ENTRYPOINT}
    if not llm_packet:
        return {
            "mode": "deterministic_fallback",
            "provider": None,
            "model": None,
            "entrypoint": MULTI_AGENT_LLM_ENTRYPOINT,
        }
    packet = {
        "mode": llm_packet.get("mode", "deterministic_fallback"),
        "provider": llm_packet.get("provider"),
        "model": llm_packet.get("model"),
        "entrypoint": llm_packet.get("entrypoint", MULTI_AGENT_LLM_ENTRYPOINT),
    }
    if llm_packet.get("error"):
        packet["error"] = llm_packet["error"]
    return packet


# Agent 1: Researcher & Fraud Auditor.
# Gathers order/customer facts and produces the fraud-risk handoff report.
def _researcher_fraud_auditor(state: MultiAgentState) -> MultiAgentState:
    ticket = state["ticket"]
    tools_called = list(state.get("tools_called", []))
    agent_messages = list(state.get("agent_messages", []))
    agent_execution = dict(state.get("agent_execution", {}))
    order_id = extract_order_id(ticket)
    base_update: MultiAgentState = {
        "order_id": order_id,
        "reason": infer_reason(ticket),
        "sentiment": infer_sentiment(ticket),
        "tools_called": tools_called,
        "agent_messages": agent_messages,
        "agent_execution": agent_execution,
    }

    if not order_id:
        base_update["stop_reason"] = "ORDER_ID_MISSING"
        base_update["error"] = {
            "error": "ORDER_ID_MISSING",
            "message": "No order id in ORD-#### format was found in the customer ticket.",
        }
        agent_execution["researcher"] = _execution_packet(None, skipped=True)
        agent_messages.append({"agent": "Researcher & Fraud Auditor", "status": "stopped", "reason": "missing order id"})
        return base_update

    order = _call_tool("get_order_details", tools_called, order_id)
    base_update["order"] = order
    if _has_error(order):
        base_update["stop_reason"] = order["error"]
        base_update["error"] = order
        agent_execution["researcher"] = _execution_packet(None, skipped=True)
        agent_messages.append({"agent": "Researcher & Fraud Auditor", "status": "stopped", "reason": order["error"]})
        return base_update

    user = _call_tool("get_user_profile", tools_called, order["user_id"])
    base_update["user"] = user
    if _has_error(user):
        base_update["stop_reason"] = user["error"]
        base_update["error"] = user
        agent_execution["researcher"] = _execution_packet(None, skipped=True)
        agent_messages.append({"agent": "Researcher & Fraud Auditor", "status": "stopped", "reason": user["error"]})
        return base_update

    tools_called.append("audit_fraud_risk")
    fraud_report = mat.audit_fraud_risk(order, user)
    llm_packet = _agent1_llm_report(ticket, base_update["reason"], base_update["sentiment"], order, user, fraud_report)
    investigation_report = _validated_investigation_report(
        parsed=llm_packet.get("parsed"),
        ticket=ticket,
        order_id=order_id,
        reason=base_update["reason"],
        sentiment=base_update["sentiment"],
        order=order,
        user=user,
        fraud_report=fraud_report,
    )
    base_update["fraud_report"] = fraud_report
    base_update["investigation_report"] = investigation_report
    agent_execution["researcher"] = _execution_packet(llm_packet)
    agent_messages.append(
        {
            "agent": "Researcher & Fraud Auditor",
            "status": "completed",
            "execution": agent_execution["researcher"],
            "handoff": {
                "order_id": order_id,
                "user_id": user["user_id"],
                "risk_level": fraud_report["risk_level"],
                "matched_rule_ids": fraud_report["matched_rule_ids"],
                "investigation_status": investigation_report["investigation_status"],
            },
        }
    )
    return base_update


# Agent 2: Decision Maker / Operations Lead.
# Applies policy and refund guardrails using deterministic tool results.
def _decision_maker(state: MultiAgentState) -> MultiAgentState:
    tools_called = list(state.get("tools_called", []))
    agent_messages = list(state.get("agent_messages", []))
    agent_execution = dict(state.get("agent_execution", {}))
    if state.get("error"):
        decision_packet = {
            "decision": DECISION_NEED_MORE_INFO,
            "reason": state["error"]["message"],
            "canonical": True,
        }
        agent_execution["decision_maker"] = _execution_packet(None, skipped=True)
        agent_messages.append({"agent": "Decision Maker / Operations Lead", "status": "skipped", "reason": state["stop_reason"]})
        return {
            "decision_packet": decision_packet,
            "agent_messages": agent_messages,
            "tools_called": tools_called,
            "agent_execution": agent_execution,
        }

    policy = _call_tool("check_return_policy", tools_called, state["order_id"], state["reason"])
    update: MultiAgentState = {
        "policy": policy,
        "tools_called": tools_called,
        "agent_messages": agent_messages,
        "agent_execution": agent_execution,
    }
    if _has_error(policy):
        update["error"] = policy
        update["stop_reason"] = policy["error"]
        update["decision_packet"] = {
            "decision": DECISION_NEED_MORE_INFO,
            "reason": policy["message"],
            "canonical": True,
        }
        agent_execution["decision_maker"] = _execution_packet(None, skipped=True)
        agent_messages.append({"agent": "Decision Maker / Operations Lead", "status": "stopped", "reason": policy["error"]})
        return update

    fraud_report = state["fraud_report"]
    llm_packet = _agent2_llm_proposal(state, policy)
    agent_execution["decision_maker"] = _execution_packet(llm_packet)
    proposed_decision = _proposed_decision(llm_packet)
    proposed_summary = _proposed_summary(llm_packet)
    if not policy["eligible"]:
        decision_packet = {
            "decision": DECISION_REJECTED,
            "reason": policy["explanation"],
            "policy_ids": policy["applicable_policies"],
            "canonical": True,
            "llm_proposed_decision": proposed_decision,
        }
    elif fraud_report["requires_escalation"]:
        decision_packet = {
            "decision": DECISION_ESCALATED,
            "reason": f"Fraud audit requires human review: {', '.join(fraud_report['matched_rule_ids'])}",
            "fraud_rule_ids": fraud_report["matched_rule_ids"],
            "policy_ids": policy["applicable_policies"],
            "canonical": True,
            "llm_proposed_decision": proposed_decision,
        }
    elif float(state["order"]["total_amount"]) > float(policy["auto_refund_cap_usd"]):
        decision_packet = {
            "decision": DECISION_ESCALATED,
            "reason": (
                f"Requested amount {float(state['order']['total_amount']):.2f} USD exceeds "
                f"automatic authority {float(policy['auto_refund_cap_usd']):.2f} USD."
            ),
            "policy_ids": policy["applicable_policies"],
            "canonical": True,
            "llm_proposed_decision": proposed_decision,
        }
    elif policy.get("requires_escalation"):
        decision_packet = {
            "decision": DECISION_ESCALATED,
            "reason": "; ".join(policy.get("escalation_reasons", [])) or "Policy requires escalation.",
            "policy_ids": policy["applicable_policies"],
            "canonical": True,
            "llm_proposed_decision": proposed_decision,
        }
    else:
        refund = _call_tool("process_refund", tools_called, state["order_id"], float(state["order"]["total_amount"]), state["reason"])
        update["refund"] = refund
        if _has_error(refund):
            update["error"] = refund
            update["stop_reason"] = refund["error"]
            decision_packet = {
                "decision": DECISION_NEED_MORE_INFO,
                "reason": refund["message"],
                "canonical": True,
                "llm_proposed_decision": proposed_decision,
            }
        elif refund["status"] == "APPROVED":
            decision_packet = {
                "decision": DECISION_AUTO_REFUND_APPROVED,
                "reason": f"Refund approved by process_refund for {refund['approved_amount']:.2f} USD.",
                "policy_ids": policy["applicable_policies"],
                "canonical": True,
                "llm_proposed_decision": proposed_decision,
            }
        elif refund["status"] == "ESCALATION_REQUIRED":
            decision_packet = {
                "decision": DECISION_ESCALATED,
                "reason": refund.get("escalation_reason", "process_refund required escalation."),
                "policy_ids": policy["applicable_policies"],
                "canonical": True,
                "llm_proposed_decision": proposed_decision,
            }
        else:
            decision_packet = {
                "decision": DECISION_REJECTED,
                "reason": refund.get("message", f"process_refund returned {refund['status']}."),
                "policy_ids": policy["applicable_policies"],
                "canonical": True,
                "llm_proposed_decision": proposed_decision,
            }

    if proposed_summary:
        decision_packet["llm_decision_summary"] = proposed_summary
    update["decision_packet"] = decision_packet
    agent_messages.append(
        {
            "agent": "Decision Maker / Operations Lead",
            "status": "completed",
            "execution": agent_execution["decision_maker"],
            "handoff": {
                "decision": decision_packet["decision"],
                "reason": decision_packet["reason"],
                "llm_proposed_decision": proposed_decision,
            },
        }
    )
    return update


# Agent 3: Communications & Escalation Manager.
# Sends mock escalation alerts when needed and builds the final customer output.
def _communications_escalation_manager(state: MultiAgentState) -> MultiAgentState:
    tools_called = list(state.get("tools_called", []))
    agent_messages = list(state.get("agent_messages", []))
    agent_execution = dict(state.get("agent_execution", {}))
    decision_packet = state["decision_packet"]
    escalation_packet: dict[str, Any] = {}
    if decision_packet["decision"] == DECISION_ESCALATED:
        tools_called.append("send_slack_alert")
        escalation_packet = mat.send_slack_alert(
            state.get("order_id") or "UNKNOWN",
            decision_packet["decision"],
            decision_packet["reason"],
            _escalation_channel(state),
        )

    state_with_agent3: MultiAgentState = dict(state)
    reasoning_chain = _reasoning_chain(state_with_agent3, escalation_packet)
    preliminary_action = _action_taken(state_with_agent3, tools_called, decision_packet, escalation_packet)
    response_packet = _customer_response_packet(state_with_agent3, decision_packet, preliminary_action)
    agent_execution["communications"] = _execution_packet(response_packet)
    agent3_handoff = {
        "decision": decision_packet["decision"],
        "escalation_alert": escalation_packet.get("status"),
        "response_mode": response_packet["mode"],
        "llm_provider": response_packet["provider"],
    }
    if response_packet.get("error"):
        agent3_handoff["llm_error"] = response_packet["error"]
    agent_messages.append(
        {
            "agent": "Communications & Escalation Manager",
            "status": "completed",
            "execution": agent_execution["communications"],
            "handoff": agent3_handoff,
        }
    )
    state_with_agent3["agent_messages"] = agent_messages
    state_with_agent3["agent_execution"] = agent_execution
    action_taken = _action_taken(state_with_agent3, tools_called, decision_packet, escalation_packet)
    action_taken["agent3_response_mode"] = response_packet["mode"]
    action_taken["agent3_llm_provider"] = response_packet["provider"]
    if response_packet.get("error"):
        action_taken["agent3_llm_error"] = response_packet["error"]
    customer_response = response_packet["response"]
    result = _base_output(reasoning_chain=reasoning_chain, action_taken=action_taken, customer_response=customer_response)
    return {
        "tools_called": tools_called,
        "agent_messages": agent_messages,
        "agent_execution": agent_execution,
        "escalation_packet": escalation_packet,
        "reasoning_chain": reasoning_chain,
        "action_taken": action_taken,
        "customer_response": customer_response,
        "result": result,
    }


# Escalation routing remains deterministic: fraud, billing, then general ops.
def _escalation_channel(state: MultiAgentState) -> str:
    fraud_report = state.get("fraud_report", {})
    if fraud_report.get("requires_escalation"):
        return "fraud-review"
    policy = state.get("policy", {})
    if policy.get("verdict") == "ORDER_NOT_REFUNDABLE":
        return "billing-support"
    return "ops-refund-escalations"


# Final audit trail assembled from agent handoffs and source-of-truth tools.
def _reasoning_chain(state: MultiAgentState, escalation_packet: dict[str, Any]) -> list[str]:
    if state.get("error") and "order" not in state:
        return [
            f"Agent 1 stopped: {state['error']['error']} - {state['error']['message']}",
            "Dependent operations were not run because prerequisite business data was unavailable.",
        ]
    order = state.get("order", {})
    user = state.get("user", {})
    fraud_report = state.get("fraud_report", {})
    policy = state.get("policy", {})
    refund = state.get("refund")
    decision_packet = state["decision_packet"]
    reasoning = [
        f"Agent 1 investigated order {state.get('order_id')} and customer {user.get('user_id')}.",
        (
            f"Fraud audit risk is {fraud_report.get('risk_level')} with rules "
            f"{', '.join(fraud_report.get('matched_rule_ids', [])) or 'none'}."
        ),
    ]
    if policy:
        reasoning.append(
            f"Agent 2 policy result is {policy['verdict']} with policies {', '.join(policy['applicable_policies'])}: {policy['explanation']}"
        )
    if refund:
        reasoning.append(_refund_reasoning(refund))
    reasoning.append(f"Agent 2 canonical decision: {decision_packet['decision']} - {decision_packet['reason']}")
    if escalation_packet:
        reasoning.append(
            f"Agent 3 sent escalation alert {escalation_packet.get('alert_id')} to {escalation_packet.get('channel')}."
        )
    if order:
        reasoning.insert(1, f"Order facts: status {order.get('status')}, total {order.get('total_amount')} {order.get('currency')}.")
    return reasoning


# External action contract compatible with the Part 1 resolver output.
def _action_taken(
    state: MultiAgentState,
    tools_called: list[str],
    decision_packet: dict[str, Any],
    escalation_packet: dict[str, Any],
) -> dict[str, Any]:
    policy = state.get("policy", {})
    refund = state.get("refund")
    fraud_report = state.get("fraud_report", {})
    action: dict[str, Any] = {
        "mode": "multi-agent",
        "tools_called": tools_called,
        "decision": decision_packet["decision"],
        "order_id": state.get("order_id"),
        "reason": state.get("reason"),
        "sentiment": state.get("sentiment"),
        "fraud_risk_level": fraud_report.get("risk_level"),
        "fraud_rule_ids": fraud_report.get("matched_rule_ids", []),
        "escalation_triggered": bool(escalation_packet),
        "escalation_packet": escalation_packet or None,
        "agent_messages": state.get("agent_messages", []),
        "agent_execution": state.get("agent_execution", {}),
    }
    if policy:
        action["policy_verdict"] = policy.get("verdict")
        action["applicable_policies"] = policy.get("applicable_policies", [])
    if refund:
        action["refund_status"] = refund.get("status")
        action["refund_amount"] = refund.get("approved_amount", 0.0)
        action["refund_id"] = refund.get("refund_id")
    elif decision_packet["decision"] != DECISION_AUTO_REFUND_APPROVED:
        action["refund_status"] = "NOT_ATTEMPTED"
        action["refund_amount"] = 0.0
    if state.get("error"):
        action["error"] = state["error"]["error"]
    return action


# Customer-facing text can be LLM-assisted, but deterministic text remains the fallback.
def _customer_response_for_decision(state: MultiAgentState, decision_packet: dict[str, Any]) -> str:
    style = _sentiment_style(state.get("sentiment"))
    if state.get("error"):
        return (
            f"{style['opening']} I could not verify the order information needed to resolve this. "
            f"{style['next_step']} Please confirm the order number so our team can investigate."
        )
    user = state["user"]
    order = state["order"]
    first_name = user.get("name", "there").split()[0]
    greeting = f"Hi {first_name}, {style['opening'].lower()}"
    decision = decision_packet["decision"]
    refund = state.get("refund")
    if decision == DECISION_AUTO_REFUND_APPROVED and refund and refund.get("status") == "APPROVED":
        return (
            f"{greeting} our operations team reviewed order {order['order_id']} and approved a refund of "
            f"{refund['approved_amount']:.2f} {order['currency']}. {style['next_step']} "
            f"Your refund id is {refund.get('refund_id')}."
        )
    if decision == DECISION_ESCALATED:
        return (
            f"{greeting} our operations team reviewed order {order['order_id']} and escalated it for human review. "
            f"Reason: {decision_packet['reason']} {style['next_step']} No automatic refund has been issued."
        )
    if decision == DECISION_REJECTED:
        return (
            f"{greeting} our operations team reviewed order {order['order_id']} and cannot approve this refund. "
            f"Reason: {decision_packet['reason']}"
        )
    return (
        f"{greeting} we need one more piece of information before we can resolve order {order['order_id']}. "
        f"Reason: {decision_packet['reason']}"
    )


def _sentiment_style(sentiment: str | None) -> dict[str, str]:
    if sentiment == "urgent":
        return {
            "opening": "I understand this is urgent, and I am sorry for the trouble.",
            "next_step": "I am keeping this direct so you know exactly what happens next.",
            "instruction": "Use a direct, apologetic, reassuring tone that acknowledges urgency without overpromising.",
        }
    if sentiment == "concerned":
        return {
            "opening": "Thanks for explaining what happened; I am sorry this has been frustrating.",
            "next_step": "I will keep the next step clear and calm.",
            "instruction": "Use a warm, empathetic, calm tone.",
        }
    return {
        "opening": "Thanks for reaching out.",
        "next_step": "Here is the outcome.",
        "instruction": "Use a concise, professional tone and do not overstate urgency.",
    }


def _customer_response_packet(
    state: MultiAgentState,
    decision_packet: dict[str, Any],
    action_taken: dict[str, Any],
) -> dict[str, Any]:
    deterministic = _customer_response_for_decision(state, decision_packet)
    llm_packet = call_multi_agent_llm(
        agent_name="communications",
        system_prompt=AGENT3_PROMPT,
        payload=_llm_payload(state, decision_packet, action_taken),
        output_schema={"customer_response": "customer-facing message text only"},
        text_only=True,
    )
    if llm_packet["mode"] != "llm":
        return {
            "response": deterministic,
            "mode": "deterministic",
            "provider": llm_packet.get("provider"),
            "model": llm_packet.get("model"),
            "error": llm_packet.get("error"),
        }
    response = llm_packet.get("text", "").strip()
    if not response or not _customer_response_is_safe(response, action_taken, state, decision_packet):
        return {
            "response": deterministic,
            "mode": "deterministic",
            "provider": llm_packet.get("provider"),
            "model": llm_packet.get("model"),
            "error": "LLM response failed customer communication safety validation.",
        }
    return {
        "response": response,
        "mode": "llm",
        "provider": llm_packet.get("provider"),
        "model": llm_packet.get("model"),
    }


def _llm_payload(
    state: MultiAgentState,
    decision_packet: dict[str, Any],
    action_taken: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticket": state["ticket"],
        "sentiment": state.get("sentiment"),
        "sentiment_style_instruction": _sentiment_style(state.get("sentiment"))["instruction"],
        "order": _compact(state.get("order")),
        "user": _compact(state.get("user")),
        "fraud_report": _compact(state.get("fraud_report")),
        "policy": _compact(state.get("policy")),
        "refund": _compact(state.get("refund")),
        "decision_packet": decision_packet,
        "escalation_packet": _compact(action_taken.get("escalation_packet")),
        "action_taken": {
            "decision": action_taken["decision"],
            "refund_status": action_taken.get("refund_status"),
            "refund_amount": action_taken.get("refund_amount"),
            "refund_id": action_taken.get("refund_id"),
            "applicable_policies": action_taken.get("applicable_policies", []),
            "fraud_rule_ids": action_taken.get("fraud_rule_ids", []),
            "escalation_triggered": action_taken.get("escalation_triggered"),
        },
    }


def _customer_response_is_safe(
    response: str,
    action_taken: dict[str, Any],
    state: MultiAgentState,
    decision_packet: dict[str, Any],
) -> bool:
    text = response.lower()
    decision = decision_packet["decision"]
    if _response_contradicts_refund(response, action_taken):
        return False
    if decision == DECISION_AUTO_REFUND_APPROVED:
        refund = state.get("refund") or {}
        if refund.get("status") != "APPROVED":
            return False
        expected_amount = float(refund.get("approved_amount", 0.0))
        for amount in _money_amounts(text):
            if abs(amount - expected_amount) > 0.01:
                return False
    else:
        if _claims_refund_success(text):
            return False
    if not action_taken.get("escalation_triggered") and _claims_escalation(text):
        return False
    if decision == DECISION_REJECTED and any(phrase in text for phrase in ["eligible for a refund", "refund is approved", "approved a refund"]):
        return False
    if decision == DECISION_NEED_MORE_INFO and any(phrase in text for phrase in ["resolved", "approved", "issued"]):
        return False
    if state.get("error") and state["error"].get("error") == "ORDER_NOT_FOUND":
        return state.get("order_id", "").lower() in text or "confirm" in text or "verify" in text
    return True


def _claims_refund_success(text: str) -> bool:
    if re.search(r"\brefund\b.{0,80}\b(?:approved|issued|processed|completed)\b", text):
        return True
    phrases = [
        "refund has been issued",
        "refund was issued",
        "approved a refund",
        "refund is approved",
        "i have approved",
        "we approved",
    ]
    return any(phrase in text for phrase in phrases)


def _claims_escalation(text: str) -> bool:
    return any(phrase in text for phrase in ["escalated", "human review", "specialist review", "alerted our"])


def _money_amounts(text: str) -> list[float]:
    amounts: list[float] = []
    for match in re.finditer(r"(?:usd|\$)?\s*(\d+(?:\.\d{1,2})?)\s*(?:usd)?", text):
        try:
            amount = float(match.group(1))
        except ValueError:
            continue
        if amount >= 1:
            amounts.append(amount)
    return amounts


def _compact(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key not in {"email", "shipping_address", "payment_method_last4"}}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value
