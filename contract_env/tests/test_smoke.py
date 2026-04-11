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

    def test_concession_tracking(self) -> None:
        """Track which specific topics the opponent has conceded on."""
        from inference import _track_concessions

        history = [
            "opponent|[Counterparty] We can accept a cap on liability.",
            "agent|step=1 action=FLAG_RISK content_len=10",
            "opponent|[Counterparty] Termination flexibility is non-negotiable.",
        ]
        concessions = _track_concessions(history)
        # "cap" should be conceded, "termination" should be firm
        self.assertEqual(concessions.get("cap"), "conceded")
        self.assertEqual(concessions.get("termination"), "firm")

    def test_concession_summary_format(self) -> None:
        """Concession summary should produce a readable string."""
        from inference import _concession_summary

        concessions = {"cap": "conceded", "termination": "firm"}
        summary = _concession_summary(concessions)
        self.assertIn("WILLING", summary)
        self.assertIn("HOLDING FIRM", summary)

        # Empty case
        self.assertEqual(_concession_summary({}), "")


    def test_content_length_validation(self) -> None:
        """Oversized content should be rejected gracefully."""
        env = ContractEnv()
        env.reset()
        huge_content = "x" * (env.max_content_length + 1)
        obs, r, done, info = env.step(
            Action(action_type="EDIT_CLAUSE", content=huge_content)
        )
        self.assertEqual(r, 0.001)
        self.assertIn("error", info)
        self.assertIn("exceeds maximum length", info["error"])

    def test_episode_runs_to_max_steps(self) -> None:
        """Episode should terminate at max_steps if agent never accepts."""
        env = ContractEnv()
        env.reset()
        for i in range(env.max_steps):
            obs, r, done, info = env.step(
                Action(action_type="FLAG_RISK", content="risk")
            )
            if i < env.max_steps - 1:
                self.assertFalse(done)
        self.assertTrue(done)
        self.assertEqual(info.get("termination_reason"), "max_steps_reached")

    def test_step_after_done_returns_error(self) -> None:
        """Stepping after episode is done should return an error."""
        env = ContractEnv()
        env.reset()
        env.step(Action(action_type="ACCEPT"))
        obs, r, done, info = env.step(Action(action_type="FLAG_RISK", content="test"))
        self.assertTrue(done)
        self.assertEqual(info.get("error"), "already_done")

    def test_unicode_content_handled(self) -> None:
        """Non-ASCII characters should not crash the environment."""
        env = ContractEnv()
        env.reset()
        obs, r, done, info = env.step(
            Action(
                action_type="EDIT_CLAUSE",
                content="Les parties s'engagent à limiter la responsabilité — §12 Haftung"
            )
        )
        self.assertGreater(r, 0.0)
        self.assertLess(r, 1.0)


if __name__ == "__main__":
    unittest.main()
