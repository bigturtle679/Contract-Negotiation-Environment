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
        self.assertEqual(body["reward"]["score"], 0.01)

    def test_state_endpoint(self) -> None:
        self.client.post("/reset")
        r = self.client.get("/state")
        self.assertEqual(r.status_code, 200)
        self.assertIn("current_step", r.json())


if __name__ == "__main__":
    unittest.main()
