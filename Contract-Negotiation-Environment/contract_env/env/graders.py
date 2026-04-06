from __future__ import annotations

import re
from typing import Any

from env.models import Action, Reward
from env.tasks import NegotiationTask


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_overlap_ratio(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    sa, sb = set(ta), set(tb)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    common = len(sa & sb)
    total = len(sa | sb)
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


def keyword_match_score(text: str, risk_keywords: list[tuple[str, float]]) -> float:
    return _weighted_risk_hits(text, risk_keywords)


def _safe_overlap(text: str, safe_keywords: list[str], expected_safe: str) -> float:
    if not text.strip():
        return 0.0
    kw_score = sum(1 for k in safe_keywords if k.lower() in text.lower()) / max(
        len(safe_keywords), 1
    )
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


def action_risk_alignment(
    action_type: str,
    effective_high: bool,
    task: NegotiationTask,
) -> float:
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
) -> tuple[Reward, dict[str, Any]]:
    """Compute graded reward plus structured diagnostics for logging and APIs."""
    content = (action.content or "").strip()
    eval_text = content if content else proposed_contract_text

    kw_hit = keyword_match_score(eval_text, task.risk_keywords)
    tok_ov = token_overlap_ratio(eval_text, contract_before)
    correctness = min(
        1.0, 0.65 * kw_hit + 0.35 * tok_ov * task.clause_type_weight
    )

    improvement = _safe_overlap(content, task.safe_keywords, task.expected_safe_edit)
    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and content:
        improvement = max(
            improvement,
            _safe_overlap(content, task.safe_keywords, task.expected_safe_edit),
        )
        improvement = max(
            improvement,
            token_overlap_ratio(proposed_contract_text, task.expected_safe_edit),
        )

    eff_high = effective_risk_high(task, proposed_contract_text)
    risk_al = action_risk_alignment(action.action_type, eff_high, task)

    score = 0.4 * correctness + 0.3 * improvement + 0.3 * risk_al
    score = max(0.0, min(1.0, score))
    accept_blocked = action.action_type == "ACCEPT" and eff_high
    if accept_blocked:
        score = 0.0

    metrics: dict[str, float] = {
        "correctness": round(correctness, 4),
        "improvement": round(improvement, 4),
        "risk_alignment": round(risk_al, 4),
        "keyword_coverage": round(kw_hit, 4),
        "token_overlap_prior": round(tok_ov, 4),
        "effective_high_risk": float(eff_high),
    }

    fb_parts = [
        f"corr={correctness:.2f}",
        f"imp={improvement:.2f}",
        f"risk_al={risk_al:.2f}",
        f"eff_high={int(eff_high)}",
    ]
    reward = Reward(
        score=round(score, 4),
        feedback="; ".join(fb_parts),
        metrics=metrics,
    )
    info: dict[str, Any] = {
        "action_type": action.action_type,
        "accept_blocked": accept_blocked,
        "grade": metrics,
        "trap_unresolved": trap_unresolved(task, proposed_contract_text),
    }
    return reward, info


def grade_action(
    task: NegotiationTask,
    contract_before: str,
    action: Action,
    proposed_contract_text: str,
) -> Reward:
    r, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    return r


def score_action_hypothetical(
    task: NegotiationTask,
    state_data: dict[str, Any],
    action: Action,
) -> float:
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
    base = min(
        1.0,
        _weighted_risk_hits(contract_text, task.risk_keywords) * task.clause_type_weight,
    )
    if task.name == "HARD" and trap_unresolved(task, contract_text):
        base = min(1.0, base + 0.25)
    return round(max(0.0, min(1.0, base)), 4)


def contract_quality_score(task: NegotiationTask, contract_text: str) -> float:
    """
    Static quality proxy: low risk weight on live text + overlap with reference safe edit.
    Deterministic; useful for episode summaries (not step reward).
    """
    risk = _weighted_risk_hits(contract_text, task.risk_keywords)
    safe_sim = token_overlap_ratio(contract_text, task.expected_safe_edit)
    trap_pen = 0.25 if trap_unresolved(task, contract_text) else 0.0
    raw = 0.55 * (1.0 - risk) + 0.45 * safe_sim - trap_pen
    return round(max(0.0, min(1.0, raw)), 4)
