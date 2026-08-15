from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STARTER_KIT = ROOT / "starter-kit"
if str(STARTER_KIT) not in sys.path:
    sys.path.insert(0, str(STARTER_KIT))

import mock_services as gc  # noqa: E402


DECISION_AUTO_REFUND_APPROVED = "AUTO_REFUND_APPROVED"
DECISION_ESCALATED = "ESCALATED_TO_HUMAN"
DECISION_REJECTED = "REJECTED"
DECISION_NEED_MORE_INFO = "NEED_MORE_INFO"

RETURN_REASONS = {
    "damaged_on_arrival",
    "wrong_item",
    "item_missing",
    "late_delivery",
    "changed_mind",
}


def resolve_ticket(ticket: str, mode: str = "auto") -> dict[str, Any]:
    """Resolve a GlobalCart support ticket with the requested agent mode."""
    selected_mode = _select_mode(mode)
    if selected_mode in {"openai", "grok", "gemini"}:
        try:
            if selected_mode == "openai":
                return resolve_ticket_openai(ticket)
            if selected_mode == "grok":
                return resolve_ticket_grok(ticket)
            return resolve_ticket_gemini(ticket)
        except Exception as exc:
            if mode in {"openai", "grok", "gemini"}:
                raise
            fallback = resolve_ticket_local(ticket)
            fallback["action_taken"]["mode"] = "local_fallback"
            fallback["action_taken"][f"{selected_mode}_error"] = str(exc)
            fallback["reasoning_chain"].insert(
                0,
                f"{selected_mode} mode was unavailable or returned an unusable response; local deterministic fallback handled the ticket.",
            )
            return fallback
    return resolve_ticket_local(ticket)


def resolve_ticket_local(ticket: str) -> dict[str, Any]:
    """Deterministic fallback resolver that uses the same GlobalCart tools."""
    tools_called: list[str] = []
    order_id = extract_order_id(ticket)
    reason = infer_reason(ticket)
    sentiment = infer_sentiment(ticket)

    if not order_id:
        return _base_output(
            reasoning_chain=[
                "No order id in ORD-#### format was found in the customer ticket.",
                "The agent cannot inspect order, customer, policy, or refund data without an order id.",
            ],
            action_taken={
                "mode": "local",
                "tools_called": tools_called,
                "decision": DECISION_NEED_MORE_INFO,
            },
            customer_response=(
                "Thanks for reaching out. I could not find an order number in your message. "
                "Please send the order id in the format ORD-1234 so I can investigate it."
            ),
        )

    order = _call_tool("get_order_details", tools_called, order_id)
    if _has_error(order):
        return _business_error_output(ticket, tools_called, order, order_id, "local")

    user = _call_tool("get_user_profile", tools_called, order["user_id"])
    if _has_error(user):
        return _business_error_output(ticket, tools_called, user, order_id, "local")

    policy = _call_tool("check_return_policy", tools_called, order_id, reason)
    if _has_error(policy):
        return _business_error_output(ticket, tools_called, policy, order_id, "local")

    reasoning = _build_common_reasoning(order, user, policy, reason, sentiment)

    refund = None
    if policy["eligible"]:
        refund = _call_tool("process_refund", tools_called, order_id, float(order["total_amount"]), reason)
        if _has_error(refund):
            return _business_error_output(ticket, tools_called, refund, order_id, "local")
        reasoning.append(_refund_reasoning(refund))

    decision = _decision_from_results(policy, refund)
    action_taken: dict[str, Any] = {
        "mode": "local",
        "tools_called": tools_called,
        "decision": decision,
        "order_id": order_id,
        "reason": reason,
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

    return _base_output(
        reasoning_chain=reasoning,
        action_taken=action_taken,
        customer_response=_customer_response(ticket, order, user, policy, refund, decision),
    )


def resolve_ticket_openai(ticket: str) -> dict[str, Any]:
    """Resolve a ticket through OpenAI function tool calling."""
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-5")
    return _resolve_ticket_responses_api(
        ticket=ticket,
        client=client,
        model=model,
        provider="openai",
        use_previous_response_id=False,
    )


def resolve_ticket_grok(ticket: str) -> dict[str, Any]:
    """Resolve a ticket through Grok function tool calling."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    model = os.environ.get("GROK_MODEL", "grok-4.6")
    return _resolve_ticket_responses_api(
        ticket=ticket,
        client=client,
        model=model,
        provider="grok",
        use_previous_response_id=True,
    )


def resolve_ticket_gemini(ticket: str) -> dict[str, Any]:
    """Resolve a ticket through Gemini function calling."""
    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    tools = [_schema_to_gemini_tool(schema) for schema in gc.TOOL_SCHEMAS]
    history: list[dict[str, Any]] = [
        {
            "type": "user_input",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{SYSTEM_PROMPT}\n\nResolve this GlobalCart support ticket. "
                        "Return only JSON with reasoning_chain, action_taken, and customer_response.\n\n"
                        f"Ticket: {ticket}"
                    ),
                }
            ],
        }
    ]
    tools_called: list[str] = []

    for _ in range(8):
        interaction = client.interactions.create(model=model, store=False, input=history, tools=tools)
        steps = list(getattr(interaction, "steps", []) or [])
        function_calls = [step for step in steps if _item_type(step) == "function_call"]
        if not function_calls:
            parsed = _parse_json_output(_interaction_text(interaction))
            _validate_agent_output(parsed)
            parsed["action_taken"].setdefault("mode", "gemini")
            parsed["action_taken"].setdefault("tools_called", tools_called)
            return parsed

        history.extend(_items_as_dicts(steps))
        for call in function_calls:
            name = _item_get(call, "name")
            call_id = _item_get(call, "id")
            arguments = _item_get(call, "arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            tool_result = _execute_registered_tool(name, arguments, tools_called)
            history.append(
                {
                    "type": "function_result",
                    "name": name,
                    "call_id": call_id,
                    "result": [{"type": "text", "text": json.dumps(tool_result, ensure_ascii=False)}],
                }
            )

    raise RuntimeError("Gemini tool loop reached the maximum number of iterations.")


def _resolve_ticket_responses_api(
    ticket: str,
    client: Any,
    model: str,
    provider: str,
    use_previous_response_id: bool,
) -> dict[str, Any]:
    """Resolve a ticket with an OpenAI-compatible Responses API tool loop."""
    tools = [_schema_to_openai_tool(schema) for schema in gc.TOOL_SCHEMAS]
    input_items: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Resolve this GlobalCart support ticket. Return only JSON with "
                "reasoning_chain, action_taken, and customer_response.\n\n"
                f"Ticket: {ticket}"
            ),
        },
    ]
    tools_called: list[str] = []
    previous_response_id: str | None = None

    for _ in range(8):
        request: dict[str, Any] = {"model": model, "input": input_items, "tools": tools, "tool_choice": "auto"}
        if use_previous_response_id and previous_response_id:
            request["previous_response_id"] = previous_response_id
        response = client.responses.create(**request)
        previous_response_id = getattr(response, "id", None)
        output = _response_output(response)
        function_calls = [item for item in output if _item_type(item) == "function_call"]
        if not function_calls:
            parsed = _parse_json_output(_response_text(response))
            _validate_agent_output(parsed)
            parsed["action_taken"].setdefault("mode", provider)
            parsed["action_taken"].setdefault("tools_called", tools_called)
            return parsed

        if not use_previous_response_id:
            input_items.extend(_items_as_dicts(output))
        next_input_items: list[dict[str, Any]] = []
        for call in function_calls:
            name = _item_get(call, "name")
            call_id = _item_get(call, "call_id")
            arguments = json.loads(_item_get(call, "arguments") or "{}")
            tool_result = _execute_registered_tool(name, arguments, tools_called)
            next_input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(tool_result, ensure_ascii=False),
                }
            )
        if use_previous_response_id:
            input_items = next_input_items
        else:
            input_items.extend(next_input_items)

    raise RuntimeError(f"{provider} tool loop reached the maximum number of iterations.")


SYSTEM_PROMPT = """You are GlobalCart's Operations Resolver Agent.

Use the provided tools as the only source of truth. Never invent order dates,
amounts, customer tier, policy ids, refund ids, or refund status.

Required tool policy:
- Call get_order_details first when an order id appears.
- Call get_user_profile after a valid order lookup.
- Call check_return_policy before making a refund decision.
- Call process_refund only when policy says the claim is eligible.
- Stop honestly on any tool response containing an error key.
- Never say a refund was issued unless process_refund returns status APPROVED.

Return only parseable JSON with exactly these top-level keys:
- reasoning_chain: list of concise audit statements based on tool outputs.
- action_taken: object containing tools_called, decision, and relevant amounts/statuses.
- customer_response: customer-facing reply.
"""


def extract_order_id(ticket: str) -> str | None:
    match = re.search(r"\bORD-\d{4,}\b", ticket.upper())
    return match.group(0) if match else None


def infer_reason(ticket: str) -> str:
    text = ticket.lower()
    if any(word in text for word in ["wrong item", "wrong product", "incorrect item"]):
        return "wrong_item"
    if any(word in text for word in ["missing", "not in the box", "never arrived"]):
        return "item_missing"
    if any(word in text for word in ["late", "delayed"]):
        return "late_delivery"
    if any(word in text for word in ["changed my mind", "by accident", "accident", "return it"]):
        return "changed_mind"
    if any(word in text for word in ["cracked", "dented", "leaking", "smashed", "damaged", "broken"]):
        return "damaged_on_arrival"
    return "damaged_on_arrival"


def infer_sentiment(ticket: str) -> str:
    text = ticket.lower()
    urgent_words = ["today", "now", "immediately", "angry", "furious", "refund me"]
    disappointed_words = ["please", "can you", "sort this out", "would like"]
    if any(word in text for word in urgent_words):
        return "urgent"
    if any(word in text for word in disappointed_words):
        return "concerned"
    return "neutral"


def _call_tool(name: str, tools_called: list[str], *args: Any) -> dict[str, Any]:
    tools_called.append(name)
    return gc.TOOL_REGISTRY[name](*args)


def _execute_registered_tool(name: str, arguments: dict[str, Any], tools_called: list[str]) -> dict[str, Any]:
    if name not in gc.TOOL_REGISTRY:
        return {"error": "UNKNOWN_TOOL", "message": f"Tool {name!r} is not registered."}
    tools_called.append(name)
    return gc.TOOL_REGISTRY[name](**arguments)


def _has_error(result: dict[str, Any]) -> bool:
    return "error" in result


def _business_error_output(
    ticket: str,
    tools_called: list[str],
    error_result: dict[str, Any],
    order_id: str,
    mode: str,
) -> dict[str, Any]:
    return _base_output(
        reasoning_chain=[
            f"Ticket mentioned order {order_id}.",
            f"{tools_called[-1]} returned {error_result['error']}: {error_result['message']}",
            "The agent stopped instead of guessing missing business data or retrying in a loop.",
        ],
        action_taken={
            "mode": mode,
            "tools_called": tools_called,
            "decision": DECISION_NEED_MORE_INFO,
            "order_id": order_id,
            "error": error_result["error"],
        },
        customer_response=(
            "Thanks for reaching out. I could not verify that order with the details provided. "
            "Please confirm the order number so I can investigate this properly."
        ),
    )


def _build_common_reasoning(
    order: dict[str, Any],
    user: dict[str, Any],
    policy: dict[str, Any],
    reason: str,
    sentiment: str,
) -> list[str]:
    return [
        (
            f"Ticket sentiment appears {sentiment}; inferred return reason is {reason}."
        ),
        (
            f"Order {order['order_id']} is {order['status']}, delivered {order.get('delivery_date')}, "
            f"total {float(order['total_amount']):.2f} {order['currency']}."
        ),
        (
            f"Customer {user['user_id']} is {user['tier']} with fraud score "
            f"{user['initial_fraud_score']} and {user['prior_fraud_flags']} prior fraud flag(s)."
        ),
        (
            f"check_return_policy returned {policy['verdict']} with policies "
            f"{', '.join(policy['applicable_policies'])}: {policy['explanation']}"
        ),
    ]


def _refund_reasoning(refund: dict[str, Any]) -> str:
    status = refund["status"]
    approved = float(refund.get("approved_amount", 0.0))
    if status == "APPROVED":
        return f"process_refund returned APPROVED for {approved:.2f} USD with refund id {refund.get('refund_id')}."
    if status == "ESCALATION_REQUIRED":
        return f"process_refund returned ESCALATION_REQUIRED and approved 0.00 USD."
    return f"process_refund returned {status} and approved {approved:.2f} USD."


def _decision_from_results(policy: dict[str, Any], refund: dict[str, Any] | None) -> str:
    if not policy["eligible"]:
        return DECISION_REJECTED
    if refund and refund["status"] == "APPROVED":
        return DECISION_AUTO_REFUND_APPROVED
    if refund and refund["status"] == "ESCALATION_REQUIRED":
        return DECISION_ESCALATED
    if policy.get("requires_escalation"):
        return DECISION_ESCALATED
    return DECISION_REJECTED


def _customer_response(
    ticket: str,
    order: dict[str, Any],
    user: dict[str, Any],
    policy: dict[str, Any],
    refund: dict[str, Any] | None,
    decision: str,
) -> str:
    first_name = user.get("name", "there").split()[0]
    if decision == DECISION_AUTO_REFUND_APPROVED:
        amount = float(refund.get("approved_amount", 0.0)) if refund else 0.0
        return (
            f"Hi {first_name}, I checked order {order['order_id']} and confirmed it is eligible. "
            f"I have approved a refund of {amount:.2f} {order['currency']}. "
            f"Your refund id is {refund.get('refund_id')}."
        )
    if decision == DECISION_ESCALATED:
        reason = "; ".join(policy.get("escalation_reasons", [])) or "the refund is above automatic approval authority"
        return (
            f"Hi {first_name}, I checked order {order['order_id']} and your claim is valid, "
            f"but it needs a human operations review because {reason}. "
            "I have escalated the case and no refund has been issued automatically."
        )
    return (
        f"Hi {first_name}, I checked order {order['order_id']} and cannot approve a refund automatically. "
        f"The policy result is {policy['verdict']}: {policy['explanation']}"
    )


def _base_output(
    reasoning_chain: list[str],
    action_taken: dict[str, Any],
    customer_response: str,
) -> dict[str, Any]:
    return {
        "reasoning_chain": reasoning_chain,
        "action_taken": action_taken,
        "customer_response": customer_response,
    }


def _schema_to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": schema["name"],
        "description": schema["description"],
        "parameters": schema["input_schema"],
    }


def _schema_to_gemini_tool(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": schema["name"],
        "description": schema["description"],
        "parameters": schema["input_schema"],
    }


def _select_mode(mode: str) -> str:
    normalized = mode.lower()
    if normalized == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("XAI_API_KEY"):
            return "grok"
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        return "local"
    if normalized in {"openai", "grok", "gemini", "local"}:
        return normalized
    raise ValueError("mode must be one of: auto, local, openai, grok, gemini")


def _parse_json_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def _validate_agent_output(value: dict[str, Any]) -> None:
    required = {"reasoning_chain", "action_taken", "customer_response"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"Agent output is missing required field(s): {', '.join(sorted(missing))}")
    if not isinstance(value["reasoning_chain"], list):
        raise ValueError("reasoning_chain must be a list")
    if not isinstance(value["action_taken"], dict):
        raise ValueError("action_taken must be an object")
    if not isinstance(value["customer_response"], str):
        raise ValueError("customer_response must be a string")


def _response_output(response: Any) -> list[Any]:
    output = getattr(response, "output", None)
    if output is not None:
        return list(output)
    if isinstance(response, dict):
        return list(response.get("output", []))
    return []


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    if isinstance(response, dict) and response.get("output_text"):
        return response["output_text"]
    chunks: list[str] = []
    for item in _response_output(response):
        if _item_type(item) == "message":
            for content in _item_get(item, "content") or []:
                if _item_get(content, "type") in {"output_text", "text"}:
                    chunks.append(_item_get(content, "text") or "")
    return "".join(chunks)


def _interaction_text(interaction: Any) -> str:
    text = getattr(interaction, "output_text", None)
    if text:
        return text
    if isinstance(interaction, dict) and interaction.get("output_text"):
        return interaction["output_text"]
    chunks: list[str] = []
    for step in getattr(interaction, "steps", []) or []:
        if _item_type(step) in {"message", "text"}:
            content = _item_get(step, "content")
            if isinstance(content, str):
                chunks.append(content)
            for part in content or []:
                if _item_get(part, "type") in {"output_text", "text"}:
                    chunks.append(_item_get(part, "text") or "")
        if _item_get(step, "text"):
            chunks.append(_item_get(step, "text"))
    return "".join(chunks)


def _items_as_dicts(items: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            converted.append(item.model_dump())
        elif isinstance(item, dict):
            converted.append(item)
    return converted


def _item_type(item: Any) -> str | None:
    return _item_get(item, "type")


def _item_get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)
