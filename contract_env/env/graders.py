from __future__ import annotations

import re
from typing import Any, Tuple

from contract_env.env.models import Action, Reward
from contract_env.env.tasks import NegotiationTask


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_overlap_ratio(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    sa, sb = set(ta), set(tb)

    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def _weighted_risk_hits(text: str, risk_keywords: list[str]) -> float:
    low = text.lower()
    if not risk_keywords:
        return 0.0

    hits = 0
    for phrase in risk_keywords:
        if phrase.lower() in low:
            hits += 1

    return min(1.0, hits / max(len(risk_keywords), 1))


def keyword_match_score(text: str, risk_keywords: list[str]) -> float:
    return _weighted_risk_hits(text, risk_keywords)


def _safe_overlap(text: str, safe_keywords: list[str], expected_safe: str) -> float:
    if not text.strip():
        return 0.0

    kw_score = sum(1 for k in safe_keywords if k.lower() in text.lower()) / max(len(safe_keywords), 1)
    exp_score = token_overlap_ratio(text, expected_safe)

    return min(1.0, 0.5 * kw_score + 0.5 * exp_score)


def trap_unresolved(task: NegotiationTask, contract_text: str) -> bool:
    low = contract_text.lower()
    return any(m in low for m in task.trap_markers)


def effective_risk_high(task: NegotiationTask, contract_text: str) -> bool:
    if task.name == "HARD":
        return trap_unresolved(task, contract_text)

    hits = _weighted_risk_hits(contract_text, task.risk_keywords)

    if task.risk_level.upper() == "HIGH":
        return hits > 0.35
    if task.risk_level.upper() == "MODERATE":
        return hits > 0.42

    return False


def action_risk_alignment(action_type: str, effective_high: bool, task: NegotiationTask) -> float:
    clause_boost = task.clause_type_weight

    if action_type == "ACCEPT":
        return 0.2 / clause_boost if effective_high else 0.9 * min(clause_boost, 1.0)

    if action_type in ("FLAG_RISK", "REJECT"):
        return 0.95 * min(clause_boost, 1.2) if effective_high else 0.45

    if action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        return 0.92 * min(clause_boost, 1.2) if effective_high else 0.7

    return 0.5


def evaluate_action(
    task: NegotiationTask,
    contract_before: str,
    action: Action,
    proposed_contract_text: str,
) -> Tuple[Reward, dict[str, Any]]:

    content = (action.content or "").strip()
    eval_text = content if content else proposed_contract_text

    correctness_kw = keyword_match_score(eval_text, task.risk_keywords)
    token_ov = token_overlap_ratio(eval_text, contract_before)

    correctness = min(
        1.0, 0.65 * correctness_kw + 0.35 * token_ov * task.clause_type_weight
    )

    improvement = _safe_overlap(content, task.safe_keywords, task.expected_safe_edit)

    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and content:
        improvement = max(
            improvement,
            token_overlap_ratio(proposed_contract_text, task.expected_safe_edit),
        )

    eff_high = effective_risk_high(task, proposed_contract_text)
    risk_al = action_risk_alignment(action.action_type, eff_high, task)

    # ---------- FINAL SCORE ----------
    score = 0.4 * correctness + 0.3 * improvement + 0.3 * risk_al

    # ✅ STRICT RANGE FIX (0 < score < 1)
    score = max(0.001, min(0.999, score))

    reward = Reward(score=round(score, 4))

    info = {
        "action_type": action.action_type,
        "grade": {
            "correctness": round(correctness, 4),
            "improvement": round(improvement, 4),
            "risk_alignment": round(risk_al, 4),
        },
    }

    # Prevent invalid accept on risky contract
    if action.action_type == "ACCEPT" and eff_high:
        reward.score = 0.001
        info["accept_blocked"] = True

    return reward, info


def score_action_hypothetical(task, state_data, action) -> float:
    contract_before = state_data.get("contract_text", "")
    proposed = build_proposed_contract_for_step(contract_before, action)
    return evaluate_action(task, contract_before, action, proposed)[0].score


def build_proposed_contract_for_step(contract_before: str, action: Action) -> str:
    content = (action.content or "").strip()

    if action.action_type == "EDIT_CLAUSE" and content:
        return content

    if action.action_type == "PROPOSE_COUNTER" and content:
        return f"{contract_before}\n\n[COUNTERPROPOSAL]\n{content}"

    return contract_before


def observation_risk_float(task: NegotiationTask, contract_text: str) -> float:
    base = _weighted_risk_hits(contract_text, task.risk_keywords)

    if task.name == "HARD" and trap_unresolved(task, contract_text):
        base = min(1.0, base + 0.25)

    base = min(0.999, max(0.001, base))
    return round(base, 4)


def contract_quality_score(task: NegotiationTask, contract_text: str) -> float:
    return 1.0 - observation_risk_float(task, contract_text)


def grade_action(
    task: NegotiationTask,
    contract_before: str,
    action: Action,
    proposed_contract_text: str,
) -> Reward:
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    return reward


# Specific graders for each task
def grade_easy(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    return grade_action(task, contract_before, action, proposed_contract_text)


def grade_medium(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    return grade_action(task, contract_before, action, proposed_contract_text)


def grade_hard(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    return grade_action(task, contract_before, action, proposed_contract_text)