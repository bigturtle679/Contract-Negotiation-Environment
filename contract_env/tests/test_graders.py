from __future__ import annotations

import unittest

from env.graders import (
    contract_quality_score,
    effective_risk_high,
    evaluate_action,
    grade_action,
    token_overlap_ratio,
)
from env.models import Action
from env.tasks import TASKS


class TestGraders(unittest.TestCase):
    def test_overlap_symmetric(self) -> None:
        self.assertEqual(token_overlap_ratio("a b c", "b c d"), 0.5)

    def test_score_bounds(self) -> None:
        task = TASKS[0]
        a = Action(action_type="FLAG_RISK", content="note")
        r = grade_action(task, task.contract_text, a, task.contract_text)
        self.assertGreaterEqual(r.score, 0.0)
        self.assertLessEqual(r.score, 1.0)

    def test_accept_high_risk_forces_zero(self) -> None:
        task = next(t for t in TASKS if t.name == "EASY")
        prop = task.contract_text
        r, info = evaluate_action(task, prop, Action(action_type="ACCEPT"), prop)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(info.get("accept_blocked"))

    def test_moderate_accept_on_draft_forces_zero(self) -> None:
        task = next(t for t in TASKS if t.name == "MEDIUM")
        prop = task.contract_text
        r, _ = evaluate_action(task, prop, Action(action_type="ACCEPT"), prop)
        self.assertEqual(r.score, 0.0)

    def test_edit_improves_metrics(self) -> None:
        task = TASKS[0]
        before = task.contract_text
        edit = Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit)
        prop = task.expected_safe_edit
        r_bad = grade_action(task, before, Action(action_type="REJECT", content="no"), before)
        r_good = grade_action(task, before, edit, prop)
        self.assertGreater(r_good.score, r_bad.score)

    def test_contract_quality_after_safe_edit(self) -> None:
        task = TASKS[0]
        q0 = contract_quality_score(task, task.contract_text)
        q1 = contract_quality_score(task, task.expected_safe_edit)
        self.assertGreater(q1, q0)

    def test_effective_high_hard_trap(self) -> None:
        task = next(t for t in TASKS if t.name == "HARD")
        self.assertTrue(effective_risk_high(task, task.contract_text))
        self.assertFalse(
            effective_risk_high(task, task.expected_safe_edit),
        )


if __name__ == "__main__":
    unittest.main()
