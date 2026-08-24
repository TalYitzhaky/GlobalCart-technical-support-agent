from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .resolver import (
    DECISION_NEED_MORE_INFO,
    _base_output,
    _build_common_reasoning,
    _business_error_output,
    _call_tool,
    _customer_response,
    _decision_from_results,
    _has_error,
    _refund_reasoning,
    extract_order_id,
    infer_reason,
    infer_sentiment,
)


class ResolverState(TypedDict, total=False):
    ticket: str
    order_id: str | None
    reason: str
    sentiment: str
    order: dict[str, Any]
    user: dict[str, Any]
    policy: dict[str, Any]
    refund: dict[str, Any] | None
    tools_called: list[str]
    decision: str
    error: dict[str, Any]
    reasoning_chain: list[str]
    customer_response: str
    result: dict[str, Any]


def resolve_ticket_langgraph_local(ticket: str) -> dict[str, Any]:
    """Resolve a ticket through a deterministic LangGraph workflow."""
    graph = _build_graph()
    final_state = graph.invoke({"ticket": ticket, "tools_called": []})
    return final_state["result"]


def _build_graph():
    builder = StateGraph(ResolverState)
    builder.add_node("initialize_ticket", _initialize_ticket)
    builder.add_node("lookup_order", _lookup_order)
    builder.add_node("lookup_user", _lookup_user)
    builder.add_node("check_policy", _check_policy)
    builder.add_node("process_refund", _process_refund)
    builder.add_node("final_response", _final_response)

    builder.add_edge(START, "initialize_ticket")
    builder.add_conditional_edges(
        "initialize_ticket",
        _route_after_initialize,
        {"lookup_order": "lookup_order", "final_response": "final_response"},
    )
    builder.add_conditional_edges(
        "lookup_order",
        _route_after_tool_lookup,
        {"lookup_user": "lookup_user", "final_response": "final_response"},
    )
    builder.add_conditional_edges(
        "lookup_user",
        _route_after_tool_lookup,
        {"check_policy": "check_policy", "final_response": "final_response"},
    )
    builder.add_conditional_edges(
        "check_policy",
        _route_after_policy,
        {"process_refund": "process_refund", "final_response": "final_response"},
    )
    builder.add_edge("process_refund", "final_response")
    builder.add_edge("final_response", END)
    return builder.compile()


def _initialize_ticket(state: ResolverState) -> ResolverState:
    ticket = state["ticket"]
    order_id = extract_order_id(ticket)
    if not order_id:
        return {
            "order_id": None,
            "reason": infer_reason(ticket),
            "sentiment": infer_sentiment(ticket),
            "error": {
                "error": "ORDER_ID_MISSING",
                "message": "No order id in ORD-#### format was found in the customer ticket.",
            },
        }
    return {
        "order_id": order_id,
        "reason": infer_reason(ticket),
        "sentiment": infer_sentiment(ticket),
    }


def _lookup_order(state: ResolverState) -> ResolverState:
    tools_called = list(state.get("tools_called", []))
    order = _call_tool("get_order_details", tools_called, state["order_id"])
    update: ResolverState = {"tools_called": tools_called, "order": order}
    if _has_error(order):
        update["error"] = order
    return update


def _lookup_user(state: ResolverState) -> ResolverState:
    tools_called = list(state.get("tools_called", []))
    user = _call_tool("get_user_profile", tools_called, state["order"]["user_id"])
    update: ResolverState = {"tools_called": tools_called, "user": user}
    if _has_error(user):
        update["error"] = user
    return update


def _check_policy(state: ResolverState) -> ResolverState:
    tools_called = list(state.get("tools_called", []))
    policy = _call_tool("check_return_policy", tools_called, state["order_id"], state["reason"])
    update: ResolverState = {"tools_called": tools_called, "policy": policy}
    if _has_error(policy):
        update["error"] = policy
    return update


def _process_refund(state: ResolverState) -> ResolverState:
    tools_called = list(state.get("tools_called", []))
    refund = _call_tool(
        "process_refund",
        tools_called,
        state["order_id"],
        float(state["order"]["total_amount"]),
        state["reason"],
    )
    update: ResolverState = {"tools_called": tools_called, "refund": refund}
    if _has_error(refund):
        update["error"] = refund
    return update


def _final_response(state: ResolverState) -> ResolverState:
    if state.get("error"):
        result = _error_result(state)
        result["action_taken"]["mode"] = "langgraph-local"
        return {"result": result}

    order = state["order"]
    user = state["user"]
    policy = state["policy"]
    refund = state.get("refund")
    decision = _decision_from_results(policy, refund)
    reasoning = _build_common_reasoning(order, user, policy, state["reason"], state["sentiment"])
    if refund:
        reasoning.append(_refund_reasoning(refund))

    action_taken: dict[str, Any] = {
        "mode": "langgraph-local",
        "tools_called": state.get("tools_called", []),
        "decision": decision,
        "order_id": state["order_id"],
        "reason": state["reason"],
        "policy_verdict": policy["verdict"],
        "applicable_policies": policy["applicable_policies"],
    }
    if refund:
        action_taken.update(
            {
                "refund_status": refund["status"],
                "refund_amount": refund.get("approved_amount", 0.0),
                "refund_id": refund.get("refund_id"),
            }
        )
        if refund.get("escalation_reason"):
            action_taken["escalation_reason"] = refund["escalation_reason"]
    elif not policy["eligible"]:
        action_taken["refund_amount"] = 0.0

    return {
        "decision": decision,
        "reasoning_chain": reasoning,
        "customer_response": _customer_response(state["ticket"], order, user, policy, refund, decision),
        "result": _base_output(
            reasoning_chain=reasoning,
            action_taken=action_taken,
            customer_response=_customer_response(state["ticket"], order, user, policy, refund, decision),
        ),
    }


def _error_result(state: ResolverState) -> dict[str, Any]:
    error = state["error"]
    if error["error"] == "ORDER_ID_MISSING":
        return _base_output(
            reasoning_chain=[
                error["message"],
                "The agent cannot inspect order, customer, policy, or refund data without an order id.",
            ],
            action_taken={
                "mode": "langgraph-local",
                "tools_called": state.get("tools_called", []),
                "decision": DECISION_NEED_MORE_INFO,
            },
            customer_response=(
                "Thanks for reaching out. I could not find an order number in your message. "
                "Please send the order id in the format ORD-1234 so I can investigate it."
            ),
        )
    return _business_error_output(
        state["ticket"],
        state.get("tools_called", []),
        error,
        state.get("order_id") or "UNKNOWN",
        "langgraph-local",
    )


def _route_after_initialize(state: ResolverState) -> Literal["lookup_order", "final_response"]:
    return "final_response" if state.get("error") else "lookup_order"


def _route_after_tool_lookup(state: ResolverState) -> Literal["lookup_user", "check_policy", "final_response"]:
    if state.get("error"):
        return "final_response"
    if "user" in state:
        return "check_policy"
    return "lookup_user"


def _route_after_policy(state: ResolverState) -> Literal["process_refund", "final_response"]:
    if state.get("error") or not state["policy"]["eligible"]:
        return "final_response"
    return "process_refund"
