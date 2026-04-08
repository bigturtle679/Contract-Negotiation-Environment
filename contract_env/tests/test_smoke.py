from __future__ import annotations

import unittest

from contract_env.env.environment import ContractEnv
from contract_env.env.models import Action


class TestContractEnv(unittest.TestCase):
    def test_reset_cycles_tasks_deterministically(self) -> None:
        env = ContractEnv()
        clause_types = [env.reset().clause_type for _ in range(6)]
        self.assertEqual(clause_types[:3], clause_types[3:6])

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
        self.assertEqual(r, 0.01)


if __name__ == "__main__":
    unittest.main()
