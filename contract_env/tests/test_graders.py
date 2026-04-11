from __future__ import annotations

import unittest

from contract_env.env.graders import (
    contract_quality_score,
    effective_risk_high,
    evaluate_action,
    grade_action,
    grade_easy,
    grade_medium,
    grade_hard,
    grade_easy_plus,
    grade_hard_plus,
    grade_medium_plus,
    grade_hard_plus2,
    grade_expert,
    clause_completeness_score,
    semantic_similarity,
    token_overlap_ratio,
)
from contract_env.env.models import Action
from contract_env.env.tasks import TASKS


class TestGraders(unittest.TestCase):
    def test_overlap_symmetric(self) -> None:
        self.assertEqual(token_overlap_ratio("a b c", "b c d"), 0.5)

    def test_score_bounds(self) -> None:
        task = TASKS[0]
        a = Action(action_type="FLAG_RISK", content="note")
        r = grade_action(task, task.contract_text, a, task.contract_text)
        self.assertGreater(r.score, 0.0)
        self.assertLess(r.score, 1.0)

    def test_accept_high_risk_forces_zero(self) -> None:
        task = next(t for t in TASKS if t.name == "EASY")
        prop = task.contract_text
        r, info = evaluate_action(task, prop, Action(action_type="ACCEPT"), prop)
        self.assertEqual(r.score, 0.001)
        self.assertTrue(info.get("accept_blocked"))

    def test_moderate_accept_on_draft_forces_zero(self) -> None:
        task = next(t for t in TASKS if t.name == "MEDIUM")
        prop = task.contract_text
        r, _ = evaluate_action(task, prop, Action(action_type="ACCEPT"), prop)
        self.assertEqual(r.score, 0.001)

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

    def test_effective_high_covers_hard_plus2(self) -> None:
        """HARD_PLUS2 tasks with unresolved trap markers should be effectively high risk."""
        task = next(t for t in TASKS if t.name == "HARD_PLUS2")
        self.assertTrue(len(task.trap_markers) > 0, "HARD_PLUS2 must have trap markers")
        self.assertTrue(effective_risk_high(task, task.contract_text))
        self.assertFalse(effective_risk_high(task, task.expected_safe_edit))

    def test_effective_high_covers_expert(self) -> None:
        """EXPERT tasks with unresolved trap markers should be effectively high risk."""
        task = next(t for t in TASKS if t.name == "EXPERT")
        self.assertTrue(len(task.trap_markers) > 0, "EXPERT must have trap markers")
        self.assertTrue(effective_risk_high(task, task.contract_text))
        self.assertFalse(effective_risk_high(task, task.expected_safe_edit))

    def test_effective_high_covers_medium_plus(self) -> None:
        """MEDIUM_PLUS tasks with unresolved trap markers should be effectively high risk."""
        task = next(t for t in TASKS if t.name == "MEDIUM_PLUS")
        self.assertTrue(len(task.trap_markers) > 0, "MEDIUM_PLUS must have trap markers")
        self.assertTrue(effective_risk_high(task, task.contract_text))
        self.assertFalse(effective_risk_high(task, task.expected_safe_edit))

    # ── Differentiated grader tests ─────────────────────────────────────
    def test_grade_easy_rewards_safe_edit(self) -> None:
        task = next(t for t in TASKS if t.name == "EASY")
        action = Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit)
        r = grade_easy(task, task.contract_text, action, task.expected_safe_edit)
        self.assertGreater(r.score, 0.0)
        self.assertLess(r.score, 1.0)

    def test_grade_medium_penalises_premature_accept(self) -> None:
        task = next(t for t in TASKS if t.name == "MEDIUM")
        r = grade_medium(task, task.contract_text, Action(action_type="ACCEPT"), task.contract_text)
        self.assertLessEqual(r.score, 0.01)

    def test_grade_hard_penalises_unresolved_trap(self) -> None:
        task = next(t for t in TASKS if t.name == "HARD")
        # Accepting original text with traps should score low
        action = Action(action_type="EDIT_CLAUSE", content=task.contract_text)
        r = grade_hard(task, task.contract_text, action, task.contract_text)
        r_safe = grade_hard(task, task.contract_text,
                            Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit),
                            task.expected_safe_edit)
        self.assertGreater(r_safe.score, r.score)

    def test_grade_easy_plus_bounds(self) -> None:
        task = next(t for t in TASKS if t.name == "EASY_PLUS")
        a = Action(action_type="FLAG_RISK", content="note")
        r = grade_easy_plus(task, task.contract_text, a, task.contract_text)
        self.assertGreater(r.score, 0.0)
        self.assertLess(r.score, 1.0)

    def test_grade_hard_plus_penalises_unresolved_trap(self) -> None:
        task = next(t for t in TASKS if t.name == "HARD_PLUS")
        # Edit that keeps trap markers should score lower than safe edit
        action = Action(action_type="EDIT_CLAUSE", content=task.contract_text)
        r_trap = grade_hard_plus(task, task.contract_text, action, task.contract_text)
        r_safe = grade_hard_plus(task, task.contract_text,
                                 Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit),
                                 task.expected_safe_edit)
        self.assertGreater(r_safe.score, r_trap.score)

    def test_all_tasks_have_graders(self) -> None:
        for task in TASKS:
            self.assertTrue(task.has_grader(), f"Task {task.id} missing grader")

    # ── NEW: Tests for new graders ──────────────────────────────────────
    def test_grade_medium_plus_rewards_scoped_nda(self) -> None:
        task = next(t for t in TASKS if t.name == "MEDIUM_PLUS")
        action = Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit)
        r = grade_medium_plus(task, task.contract_text, action, task.expected_safe_edit)
        self.assertGreater(r.score, 0.0)
        self.assertLess(r.score, 1.0)

    def test_grade_medium_plus_penalises_overbroad_accept(self) -> None:
        task = next(t for t in TASKS if t.name == "MEDIUM_PLUS")
        r = grade_medium_plus(task, task.contract_text,
                              Action(action_type="ACCEPT"), task.contract_text)
        # Accepting overbroad NDA should be penalised
        r_edit = grade_medium_plus(task, task.contract_text,
                                   Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit),
                                   task.expected_safe_edit)
        self.assertGreater(r_edit.score, r.score)

    def test_grade_hard_plus2_rewards_cure_period(self) -> None:
        task = next(t for t in TASKS if t.name == "HARD_PLUS2")
        action = Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit)
        r = grade_hard_plus2(task, task.contract_text, action, task.expected_safe_edit)
        self.assertGreater(r.score, 0.0)
        self.assertLess(r.score, 1.0)

    def test_grade_hard_plus2_penalises_unresolved(self) -> None:
        task = next(t for t in TASKS if t.name == "HARD_PLUS2")
        action = Action(action_type="EDIT_CLAUSE", content=task.contract_text)
        r_bad = grade_hard_plus2(task, task.contract_text, action, task.contract_text)
        r_good = grade_hard_plus2(task, task.contract_text,
                                  Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit),
                                  task.expected_safe_edit)
        self.assertGreater(r_good.score, r_bad.score)

    def test_grade_expert_rewards_gdpr_language(self) -> None:
        task = next(t for t in TASKS if t.name == "EXPERT")
        action = Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit)
        r = grade_expert(task, task.contract_text, action, task.expected_safe_edit)
        self.assertGreater(r.score, 0.0)
        self.assertLess(r.score, 1.0)

    def test_grade_expert_penalises_unresolved_data_traps(self) -> None:
        task = next(t for t in TASKS if t.name == "EXPERT")
        action = Action(action_type="EDIT_CLAUSE", content=task.contract_text)
        r_bad = grade_expert(task, task.contract_text, action, task.contract_text)
        r_good = grade_expert(task, task.contract_text,
                              Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit),
                              task.expected_safe_edit)
        self.assertGreater(r_good.score, r_bad.score)

    # ── NEW: Tests for enhanced scoring metrics ─────────────────────────
    def test_clause_completeness_score_full(self) -> None:
        score = clause_completeness_score("capped at twelve months, no consequential or punitive",
                                          ["capped", "twelve", "consequential", "punitive"])
        self.assertEqual(score, 1.0)

    def test_clause_completeness_score_partial(self) -> None:
        score = clause_completeness_score("capped at twelve months",
                                          ["capped", "twelve", "consequential", "punitive"])
        self.assertEqual(score, 0.5)

    def test_clause_completeness_score_empty_requirements(self) -> None:
        score = clause_completeness_score("any text", [])
        self.assertEqual(score, 1.0)

    def test_semantic_similarity_identical(self) -> None:
        sim = semantic_similarity("hello world test", "hello world test")
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_semantic_similarity_different(self) -> None:
        sim = semantic_similarity("hello world test", "completely unrelated xyz")
        self.assertLess(sim, 0.5)

    def test_evaluate_action_returns_new_grade_fields(self) -> None:
        """Verify evaluate_action returns semantic_similarity and completeness in grade info."""
        task = TASKS[0]
        action = Action(action_type="EDIT_CLAUSE", content=task.expected_safe_edit)
        _, info = evaluate_action(task, task.contract_text, action, task.expected_safe_edit)
        grade = info["grade"]
        self.assertIn("semantic_similarity", grade)
        self.assertIn("completeness", grade)

    def test_eight_graded_tasks(self) -> None:
        """Ensure we have at least 8 graded tasks."""
        graded = [t for t in TASKS if t.has_grader()]
        self.assertGreaterEqual(len(graded), 8)

    def test_observation_risk_float_trap_bonus_all_tasks(self) -> None:
        """All tasks with trap_markers should get a risk boost in observation_risk_float."""
        from contract_env.env.graders import observation_risk_float
        for task in TASKS:
            if task.trap_markers:
                risk_with_trap = observation_risk_float(task, task.contract_text)
                # Contract text with trap markers should have elevated risk
                self.assertGreater(risk_with_trap, 0.1,
                                   f"Task {task.id} trap-bearing text should have elevated risk")

    def test_accept_blocked_on_expert_unresolved(self) -> None:
        """Accepting EXPERT task with unresolved traps should be blocked."""
        task = next(t for t in TASKS if t.name == "EXPERT")
        r, info = evaluate_action(task, task.contract_text,
                                  Action(action_type="ACCEPT"), task.contract_text)
        self.assertEqual(r.score, 0.001)
        self.assertTrue(info.get("accept_blocked"))

    def test_accept_blocked_on_hard_plus2_unresolved(self) -> None:
        """Accepting HARD_PLUS2 task with unresolved traps should be blocked."""
        task = next(t for t in TASKS if t.name == "HARD_PLUS2")
        r, info = evaluate_action(task, task.contract_text,
                                  Action(action_type="ACCEPT"), task.contract_text)
        self.assertEqual(r.score, 0.001)
        self.assertTrue(info.get("accept_blocked"))


    def test_empty_risk_keywords_handled(self) -> None:
        """Tasks with empty risk_keywords should not crash scoring."""
        from contract_env.env.graders import keyword_match_score
        score = keyword_match_score("any text here", [])
        self.assertEqual(score, 0.0)

    def test_unicode_in_contract_text(self) -> None:
        """Non-ASCII contract text should be scored without errors."""
        task = TASKS[0]
        action = Action(
            action_type="EDIT_CLAUSE",
            content="Haftungsbeschränkung: Begrenzung auf gezahlte Gebühren der letzten 12 Monate.",
        )
        r = grade_action(task, task.contract_text, action, action.content)
        self.assertGreater(r.score, 0.0)
        self.assertLess(r.score, 1.0)

    def test_opponent_response_key_validation(self) -> None:
        """Invalid action type keys in opponent_responses should be rejected."""
        from contract_env.env.tasks import NegotiationTask
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            NegotiationTask(
                id="test",
                name="TEST",
                contract_text="test",
                clause_type="liability",
                risk_keywords=["test"],
                safe_keywords=["test"],
                expected_safe_edit="test",
                risk_level="HIGH",
                hidden_trap="",
                opponent_responses={"INVALID_ACTION": ["reply"]},
                grader_func=grade_easy,
                grader_name="grade_easy",
            )

    def test_evaluate_quality_endpoint_max_length(self) -> None:
        """API should reject excessively long contract_text."""
        from fastapi.testclient import TestClient
        from contract_env.server.app import app, _env
        client = TestClient(app)
        _env.reset()
        r = client.post(
            "/evaluate-quality",
            json={"contract_text": "x" * 100_001},
        )
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
