from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"


def _load(filename: str) -> dict[str, Any]:
    with (DATA_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_date(value: str | None):
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


def audit_fraud_risk(order: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Deterministically audit refund fraud risk for a customer/order pair."""
    rules_doc = _load("fraud_rules.json")
    reference_date = _parse_date(rules_doc["reference_date"])
    window_days = int(rules_doc["repeat_claim_window_days"])
    refund_history = user.get("refund_history", [])
    repeat_claims = 0
    for claim in refund_history:
        claim_date = _parse_date(claim.get("date"))
        if claim_date and (reference_date - claim_date).days <= window_days:
            repeat_claims += 1

    matched_rules: list[dict[str, Any]] = []
    if user.get("initial_fraud_score", 0) >= 60:
        matched_rules.append(_rule(rules_doc, "FRD-001"))
    if user.get("prior_fraud_flags", 0) > 0:
        matched_rules.append(_rule(rules_doc, "FRD-002"))
    if repeat_claims >= 3:
        matched_rules.append(_rule(rules_doc, "FRD-003"))
    if order.get("address_changed_at"):
        matched_rules.append(_rule(rules_doc, "FRD-004"))
    if user.get("orders_count", 0) <= 1 and float(order.get("total_amount", 0.0)) >= 500:
        matched_rules.append(_rule(rules_doc, "FRD-005"))

    requires_escalation = any(rule["requires_escalation"] for rule in matched_rules)
    if any(rule["risk_level"] == "high" for rule in matched_rules):
        risk_level = "high"
    elif matched_rules:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "status": "COMPLETED",
        "order_id": order["order_id"],
        "user_id": user["user_id"],
        "risk_level": risk_level,
        "requires_escalation": requires_escalation,
        "fraud_score": user.get("initial_fraud_score", 0),
        "prior_fraud_flags": user.get("prior_fraud_flags", 0),
        "repeat_claims_in_window": repeat_claims,
        "matched_rule_ids": [rule["rule_id"] for rule in matched_rules],
        "matched_rules": matched_rules,
    }


def send_slack_alert(
    order_id: str,
    decision: str,
    escalation_reason: str,
    channel: str | None = None,
) -> dict[str, Any]:
    """Write a deterministic mock Slack alert result without external I/O."""
    channels = _load("escalation_channels.json")
    selected = channel or channels["default_channel"]
    channel_record = next((item for item in channels["channels"] if item["channel"] == selected), None)
    if channel_record is None:
        return {
            "status": "ERROR",
            "error": "CHANNEL_NOT_FOUND",
            "message": f"No escalation channel configured for {selected}.",
        }
    return {
        "status": "SENT",
        "destination": channel_record["destination"],
        "channel": channel_record["channel"],
        "owner": channel_record["owner"],
        "alert_id": f"ALERT-{order_id}-{_stable_suffix(order_id, decision, selected)}",
        "order_id": order_id,
        "decision": decision,
        "escalation_reason": escalation_reason,
    }


def _rule(rules_doc: dict[str, Any], rule_id: str) -> dict[str, Any]:
    return next(rule for rule in rules_doc["rules"] if rule["rule_id"] == rule_id)


def _stable_suffix(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:8].upper()
