from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from contract_env.server.app import app


class TestAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "ok")

    def test_reset_returns_observation(self) -> None:
        r = self.client.post("/reset")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        obs = data["observation"]
        self.assertIn("clause_type", obs)
        self.assertIn("risk_level", obs)
        self.assertIn("negotiation_history", obs)
        self.assertTrue(any("opponent|" in x for x in obs["negotiation_history"]))

    def test_step_invalid_action_low_reward(self) -> None:
        self.client.post("/reset")
        r = self.client.post(
            "/step",
            json={"action_type": "EDIT_CLAUSE", "content": ""},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("error", body["info"])
        self.assertEqual(body["reward"]["score"], 0.001)

    def test_state_endpoint(self) -> None:
        self.client.post("/reset")
        r = self.client.get("/state")
        self.assertEqual(r.status_code, 200)
        self.assertIn("current_step", r.json())

    def test_tasks_endpoint(self) -> None:
        r = self.client.get("/tasks")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertGreaterEqual(data["total"], 8)
        self.assertGreaterEqual(data["graded"], 8)
        self.assertEqual(len(data["tasks"]), data["total"])
        for t in data["tasks"]:
            self.assertIn("id", t)
            self.assertIn("clause_type", t)
            self.assertIn("has_grader", t)

    def test_schema_endpoint(self) -> None:
        r = self.client.get("/schema")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("Action", data)
        self.assertIn("Observation", data)
        self.assertIn("Reward", data)

    def test_root_endpoint(self) -> None:
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "ok")

    def test_evaluate_quality_missing_field(self) -> None:
        """POST /evaluate-quality with empty body should return 422."""
        self.client.post("/reset")
        r = self.client.post("/evaluate-quality", json={})
        self.assertEqual(r.status_code, 422)

    def test_evaluate_quality_empty_text(self) -> None:
        """POST /evaluate-quality with empty string should return 422."""
        self.client.post("/reset")
        r = self.client.post("/evaluate-quality", json={"contract_text": ""})
        self.assertEqual(r.status_code, 422)

    def test_evaluate_quality_success(self) -> None:
        """POST /evaluate-quality with valid text should return scores."""
        self.client.post("/reset")
        r = self.client.post(
            "/evaluate-quality",
            json={"contract_text": "Liability capped at fees paid in preceding twelve months."},
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("quality_score", data)
        self.assertIn("risk_score", data)
        self.assertGreater(data["quality_score"], 0.0)

    def test_step_invalid_action_type_rejected(self) -> None:
        """POST /step with an invalid action_type should return 422."""
        self.client.post("/reset")
        r = self.client.post("/step", json={"action_type": "INVALID_ACTION"})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
