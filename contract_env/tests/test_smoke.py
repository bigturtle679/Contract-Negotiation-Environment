from __future__ import annotations

import unittest

from contract_env.env.environment import ContractEnv
from contract_env.env.models import Action
from contract_env.env.tasks import TASKS


class TestContractEnv(unittest.TestCase):
    def test_reset_cycles_tasks_deterministically(self) -> None:
        env = ContractEnv()
        num_tasks = len(TASKS)
        clause_types = [env.reset().clause_type for _ in range(num_tasks * 2)]
        self.assertEqual(clause_types[:num_tasks], clause_types[num_tasks: num_tasks * 2])

    def test_flag_then_edit_improves_contract(self) -> None:
        env = ContractEnv()
        env.reset()
        env.step(Action(action_type="FLAG_RISK", content="flag note"))
        r = env.step(
            Action(action_type="EDIT_CLAUSE", content=env.current_task.expected_safe_edit)
        )
        self.assertGreater(r[1], 0.0)
        self.assertLess(r[1], 1.0)

    def test_accept_high_risk_zero_reward(self) -> None:
        env = ContractEnv()
        env.reset()
        if env.current_task.risk_level != "HIGH":
            env.reset()
        if env.current_task.risk_level != "HIGH":
            env.reset()
        o, r, done, _ = env.step(Action(action_type="ACCEPT", content=None))
        self.assertTrue(done)
        self.assertEqual(r, 0.001)

    # ── NEW: Tests for opponent simulation ──────────────────────────────
    def test_opponent_reply_added_to_history(self) -> None:
        """After an action, the opponent should reply and appear in history."""
        env = ContractEnv()
        env.reset()
        obs, _, _, info = env.step(Action(action_type="FLAG_RISK", content="risk identified"))
        # At least one opponent reply should be in the history
        opponent_entries = [h for h in obs.negotiation_history if h.startswith("opponent|")]
        # There's at least the opening + possibly a reply
        self.assertGreaterEqual(len(opponent_entries), 1)

    def test_opponent_reply_in_info(self) -> None:
        """Opponent replies should appear in step info when available."""
        env = ContractEnv()
        env.reset()
        task = env.current_task
        # If this task has opponent_responses for FLAG_RISK, info should have the reply
        if task.opponent_responses.get("FLAG_RISK"):
            _, _, _, info = env.step(Action(action_type="FLAG_RISK", content="risk found"))
            self.assertIn("opponent_reply", info)

    # ── NEW: Tests for new tasks ────────────────────────────────────────
    def test_eight_tasks_exist(self) -> None:
        """Verify all 8 tasks are defined."""
        self.assertEqual(len(TASKS), 8)

    def test_all_tasks_cycle(self) -> None:
        """All 8 tasks should be visited when cycling through resets."""
        env = ContractEnv()
        visited_ids = set()
        for _ in range(len(TASKS)):
            env.reset()
            visited_ids.add(env.current_task.id)
        self.assertEqual(len(visited_ids), len(TASKS))

    def test_new_task_confidentiality(self) -> None:
        """Verify the confidentiality task exists and has correct properties."""
        task = next(t for t in TASKS if t.id == "medium_confidentiality_nda")
        self.assertEqual(task.clause_type, "confidentiality")
        self.assertEqual(task.risk_level, "MODERATE")
        self.assertTrue(task.has_grader())
        self.assertTrue(len(task.opponent_responses) > 0)
        self.assertTrue(len(task.required_elements) > 0)

    def test_new_task_termination(self) -> None:
        """Verify the termination task exists and has correct properties."""
        task = next(t for t in TASKS if t.id == "hard_termination_convenience")
        self.assertEqual(task.clause_type, "termination")
        self.assertEqual(task.risk_level, "HIGH")
        self.assertTrue(task.has_grader())
        self.assertTrue(len(task.trap_markers) > 0)

    def test_new_task_data_protection(self) -> None:
        """Verify the data protection task exists and has correct properties."""
        task = next(t for t in TASKS if t.id == "expert_data_protection")
        self.assertEqual(task.clause_type, "data_protection")
        self.assertEqual(task.risk_level, "HIGH")
        self.assertTrue(task.has_grader())
        self.assertTrue(len(task.trap_markers) > 0)
        self.assertTrue(len(task.required_elements) > 0)

    def test_edit_new_task_safe_edit_scores_well(self) -> None:
        """Editing with the expected safe edit should produce a decent reward for new tasks."""
        env = ContractEnv()
        # Cycle to the confidentiality task (index 5)
        for _ in range(6):
            env.reset()
        task = env.current_task
        self.assertEqual(task.id, "medium_confidentiality_nda")
        _, r, _, _ = env.step(
            Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit)
        )
        self.assertGreater(r, 0.1)

    def test_opponent_stance_parsing(self) -> None:
        """Opponent concession/firmness signals should be detected correctly."""
        from inference import _parse_opponent_stance

        # Conceding
        history_concede = [
            "opponent|[Counterparty] We can accept a cap but consequential damages must remain."
        ]
        self.assertEqual(_parse_opponent_stance(history_concede), "conceding")

        # Firm
        history_firm = [
            "opponent|[Counterparty] This is non-negotiable and standard."
        ]
        self.assertEqual(_parse_opponent_stance(history_firm), "firm")

        # Neutral
        history_neutral = [
            "opponent|[Counterparty] Our legal team considers this standard."
        ]
        self.assertEqual(_parse_opponent_stance(history_neutral), "neutral")

        # Empty
        self.assertEqual(_parse_opponent_stance([]), "neutral")


if __name__ == "__main__":
    unittest.main()
