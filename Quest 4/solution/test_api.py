"""HTTP contract checks; business behavior belongs in the scenario suites."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app


class ResolveTicketApiTests(unittest.TestCase):
    def setUp(self):
        # Patch where the endpoint looks up the resolver, before sending requests.
        resolver_patch = patch("app.resolve_ticket", autospec=True)
        self.resolver = resolver_patch.start()
        self.addCleanup(resolver_patch.stop)
        self.result = {
            "reasoning_chain": ["Verified ticket information."],
            "action_taken": {
                "decision": "NEED_MORE_INFO",
                "metadata": {"count": 1, "amount": 12.5, "ready": False, "provider": None},
                "tools_called": [],
            },
            "customer_response": "שלום — please confirm your order number.",
        }
        self.resolver.return_value = self.result
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_health_endpoints_do_not_call_resolver(self):
        for path, status in (("/health/live", "alive"), ("/health/ready", "ready")):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": status})
                self.resolver.assert_not_called()

    def test_default_mode_and_json_response(self):
        # Response-model filtering must keep internal top-level fields off the wire.
        self.resolver.return_value = {**self.result, "internal_debug": "not public"}
        response = self.client.post("/resolve-ticket", json={"ticket": "Please help."})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.json(), self.result)
        self.resolver.assert_called_once_with("Please help.", mode="multi-agent")

    def test_supported_modes_and_trimmed_ticket_are_forwarded(self):
        for mode in ("auto", "local", "langgraph-local", "multi-agent", "openai", "grok", "gemini"):
            with self.subTest(mode=mode):
                self.resolver.reset_mock()
                response = self.client.post("/resolve-ticket", json={
                    "ticket": " \n Please help.\nKeep this line. \t ", "mode": mode,
                })
                self.assertEqual(response.status_code, 200)
                self.resolver.assert_called_once_with("Please help.\nKeep this line.", mode=mode)

    def test_business_error_result_remains_http_200(self):
        # The resolver owns the decision; the API only carries its result.
        self.result["action_taken"]["error"] = "ORDER_NOT_FOUND"
        response = self.client.post("/resolve-ticket", json={"ticket": "Please help."})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.result)
        self.resolver.assert_called_once_with("Please help.", mode="multi-agent")

    def test_invalid_fields_are_rejected_before_resolution(self):
        cases = [("missing ticket", {}, "ticket")]
        cases.extend(
            (f"invalid ticket {value!r}", {"ticket": value}, "ticket")
            for value in ("", " \n\t ", None, 123, True, [], {})
        )
        cases.extend(
            (f"invalid mode {value!r}", {"ticket": "Help", "mode": value}, "mode")
            for value in ("unknown", "", None, 123, True, [], {})
        )
        for label, payload, field in cases:
            with self.subTest(case=label):
                response = self.client.post("/resolve-ticket", json=payload)
                self.assertEqual(response.status_code, 422)
                locations = [error["loc"] for error in response.json()["detail"]]
                self.assertIn(["body", field], locations)
                self.resolver.assert_not_called()

    def test_invalid_request_bodies_are_rejected_before_resolution(self):
        # Raw content covers absent, malformed, and non-object JSON bodies.
        for body in (b"", b'{"ticket":', b"null", b"[]", b'"Help"', b"123"):
            with self.subTest(body=body):
                response = self.client.post(
                    "/resolve-ticket", content=body,
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(response.status_code, 422)
                self.assertTrue(response.json()["detail"])
                self.resolver.assert_not_called()

    def test_invalid_workflow_results_are_server_errors(self):
        invalid_results = [
            {key: value for key, value in self.result.items() if key != "customer_response"},
            {**self.result, "reasoning_chain": "not a list"},
            {**self.result, "action_taken": "not an object"},
            {**self.result, "customer_response": None},
        ]
        with TestClient(app, raise_server_exceptions=False) as client:
            for result in invalid_results:
                with self.subTest(result=result):
                    self.resolver.reset_mock()
                    self.resolver.return_value = result
                    response = client.post("/resolve-ticket", json={"ticket": "Help"})
                    self.assertEqual(response.status_code, 500)
                    self.resolver.assert_called_once_with("Help", mode="multi-agent")

    def test_unexpected_workflow_exception_returns_generic_server_error(self):
        self.resolver.side_effect = RuntimeError("Private workflow failure details")

        # Observe the HTTP response instead of TestClient re-raising the exception.
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/resolve-ticket", json={"ticket": "Help"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.text, "Internal Server Error")
        self.resolver.assert_called_once_with("Help", mode="multi-agent")


if __name__ == "__main__":
    unittest.main()
