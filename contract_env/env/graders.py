from __future__ import annotations

import re
from typing import Any

from env.models import Action, Reward
from env.tasks import NegotiationTask


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _word_overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = len(ta & tb)
    total = len(ta | tb)
    return common / total if total else 0.0


def _weighted_risk_hits(text: str, risk_keywords: list[tuple[str, float]]) -> float:
    low = text.lower()
    if not risk_keywords:
        return 0.0
    total_w = sum(w for _, w in risk_keywords) or 1.0
    score = 0.0
    for phrase, w in risk_keywords:
        if phrase.lower() in low:
            score += w
    return min(1.0, score / total_w)


def _safe_overlap(text: str, safe_keywords: list[str], expected_safe: str) -> float:
    if not text.strip():
        return 0.0
    kw_score = sum(1 for k in safe_keywords if k.lower() in text.lower()) / max(
        len(safe_keywords), 1
    )
    exp_score = _word_overlap(text, expected_safe)
    return min(1.0, 0.5 * kw_score + 0.5 * exp_score)


def trap_unresolved(task: NegotiationTask, contract_text: str) -> bool:
    low = contract_text.lower()
    return any(m in low for m in task.trap_markers)


def effective_risk_high(task: NegotiationTask, contract_text: str) -> bool:
    """
    HIGH effective risk ⇒ ACCEPT yields score 0. HARD uses trap markers on the live contract;
    other tasks use nominal HIGH risk only while weighted risk phrases remain material.
    """
    if task.name == "HARD":
        return trap_unresolved(task, contract_text)
    if task.risk_level.upper() == "HIGH":
        return _weighted_risk_hits(contract_text, task.risk_keywords) > 0.35
    return False


def action_risk_alignment(
    action_type: str,
    effective_high: bool,
    task: NegotiationTask,
) -> float:
    clause_boost = task.clause_type_weight
    if action_type == "ACCEPT":
        return 0.2 / clause_boost if effective_high else 0.9 * min(clause_boost, 1.0)
    if action_type in ("FLAG", "REJECT"):
        return 0.95 * min(clause_boost, 1.2) if effective_high else 0.45
    if action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        return 0.92 * min(clause_boost, 1.2) if effective_high else 0.7
    return 0.5


def grade_action(
    task: NegotiationTask,
    contract_before: str,
    action: Action,
    proposed_contract_text: str,
) -> Reward:
    """
    Deterministic reward in [0,1]. For edit-like actions, proposed_contract_text is the
    full contract text after the proposed change (used before env mutation).
    """
    content = (action.content or "").strip()
    eval_text = content if content else proposed_contract_text

    correctness_kw = _weighted_risk_hits(eval_text, task.risk_keywords)
    token_ov = _word_overlap(eval_text, contract_before)
    correctness = min(
        1.0, 0.65 * correctness_kw + 0.35 * token_ov * task.clause_type_weight
    )

    improvement = _safe_overlap(content, task.safe_keywords, task.expected_safe_edit)
    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and content:
        improvement = max(
            improvement,
            _safe_overlap(content, task.safe_keywords, task.expected_safe_edit),
        )
        improvement = max(
            improvement,
            _word_overlap(proposed_contract_text, task.expected_safe_edit),
        )

    eff_high = effective_risk_high(task, proposed_contract_text)
    risk_al = action_risk_alignment(action.action_type, eff_high, task)

    score = 0.4 * correctness + 0.3 * improvement + 0.3 * risk_al
    score = max(0.0, min(1.0, score))

    if action.action_type == "ACCEPT" and eff_high:
        score = 0.0

    fb_parts = [
        f"corr={correctness:.2f}",
        f"imp={improvement:.2f}",
        f"risk_al={risk_al:.2f}",
        f"eff_high={int(eff_high)}",
    ]
    return Reward(score=round(score, 4), feedback="; ".join(fb_parts))


def score_action_hypothetical(
    task: NegotiationTask,
    state_data: dict[str, Any],
    action: Action,
) -> float:
    """Read-only grader for inference (same scoring, no state advance)."""
    contract_before = state_data.get("contract_text", "")
    proposed = build_proposed_contract_for_step(contract_before, action)
    return grade_action(task, contract_before, action, proposed).score


def build_proposed_contract_for_step(contract_before: str, action: Action) -> str:
    content = (action.content or "").strip()
    if action.action_type == "EDIT_CLAUSE" and content:
        return content
    if action.action_type == "PROPOSE_COUNTER" and content:
        return f"{contract_before}\n\n[COUNTERPROPOSAL]\n{content}"
    return contract_before
