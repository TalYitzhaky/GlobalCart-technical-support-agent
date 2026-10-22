"""HTTP contract checks backed by the real deterministic workflows."""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app


class ResolveTicketApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_local_refund(self):
        response = self.client.post("/resolve-ticket", json={
            "ticket": "Order ORD-1001 arrived cracked.", "mode": "local",
        })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(set(result), {"reasoning_chain", "action_taken", "customer_response"})
        self.assertEqual(result["action_taken"]["refund_status"], "APPROVED")

    @patch.dict(os.environ, {"MULTI_AGENT_LLM_PROVIDER": "deterministic"})
    def test_default_workflow_escalates_fraud(self):
        response = self.client.post("/resolve-ticket", json={
            "ticket": "Order ORD-1005 arrived smashed. Refund me.",
        })
        self.assertEqual(response.status_code, 200)
        action = response.json()["action_taken"]
        self.assertEqual(action["mode"], "multi-agent")
        self.assertEqual(action["decision"], "ESCALATED_TO_HUMAN")
        self.assertIn("send_slack_alert", action["tools_called"])
        self.assertNotIn("process_refund", action["tools_called"])

    def test_missing_order_is_business_result(self):
        response = self.client.post("/resolve-ticket", json={
            "ticket": "Order ORD-2222 never arrived.", "mode": "local",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action_taken"]["decision"], "NEED_MORE_INFO")

    def test_invalid_requests(self):
        for payload in ({}, {"ticket": ""}, {"ticket": " \n\t "},
                        {"ticket": None}, {"ticket": 123},
                        {"ticket": "Help", "mode": "unknown"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/resolve-ticket", json=payload).status_code, 422)


if __name__ == "__main__":
    unittest.main()
