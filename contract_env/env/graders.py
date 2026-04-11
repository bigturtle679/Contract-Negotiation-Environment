from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Tuple, TYPE_CHECKING

from contract_env.env.models import Action, Reward

if TYPE_CHECKING:
    from contract_env.env.tasks import NegotiationTask


# ── TEXT ANALYSIS UTILITIES ──────────────────────────────────────────────

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


def _cosine_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between two texts using term frequency vectors."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    all_tokens = set(ca) | set(cb)
    dot = sum(ca.get(t, 0) * cb.get(t, 0) for t in all_tokens)
    mag_a = math.sqrt(sum(v * v for v in ca.values()))
    mag_b = math.sqrt(sum(v * v for v in cb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Negation prefixes that invert the meaning of a risk keyword.
# E.g. "no consequential damages" is safe, not risky.
_NEGATION_PREFIXES = (
    "no ", "not ", "neither ", "without any ", "exclud", "except for ",
    "not liable for ", "no party is liable for ", "shall not include ",
    "does not cover ", "not responsible for ",
)


def _is_negated(text_lower: str, keyword_lower: str) -> bool:
    """Return True if *every* occurrence of keyword_lower in text_lower is preceded
    by a negation phrase, meaning the keyword appears only in a 'safe' context.

    Returns False if the keyword is not found at all (no occurrence to negate).
    """
    idx = 0
    all_negated = True
    found_any = False
    while True:
        pos = text_lower.find(keyword_lower, idx)
        if pos == -1:
            break
        found_any = True
        # Check the 60-character window before the match for negation cues
        window_start = max(0, pos - 60)
        preceding = text_lower[window_start:pos]
        if not any(neg in preceding for neg in _NEGATION_PREFIXES):
            all_negated = False
            break
        idx = pos + len(keyword_lower)
    return found_any and all_negated


def _weighted_risk_hits(text: str, risk_keywords: list[str]) -> float:
    """Return the fraction of *risk_keywords* found (un-negated) in *text* ∈ [0, 1]."""
    low = text.lower()
    if not risk_keywords:
        return 0.0

    hits = 0
    for phrase in risk_keywords:
        kw = phrase.lower()
        if kw in low and not _is_negated(low, kw):
            hits += 1

    return min(1.0, hits / max(len(risk_keywords), 1))


def keyword_match_score(text: str, risk_keywords: list[str]) -> float:
    """Public alias for :func:`_weighted_risk_hits` — negation-aware risk score."""
    return _weighted_risk_hits(text, risk_keywords)


def _safe_overlap(text: str, safe_keywords: list[str], expected_safe: str) -> float:
    """Score how well *text* overlaps with safe keywords and the expected safe edit ∈ [0, 1]."""
    if not text.strip():
        return 0.0

    kw_score = sum(1 for k in safe_keywords if k.lower() in text.lower()) / max(len(safe_keywords), 1)
    exp_score = token_overlap_ratio(text, expected_safe)

    return min(1.0, 0.5 * kw_score + 0.5 * exp_score)


def clause_completeness_score(text: str, required_elements: list[str]) -> float:
    """Score how many required legal elements are present in the clause text.

    Args:
        text: The clause text to evaluate.
        required_elements: Lowercase phrases that a well-drafted clause should contain.

    Returns:
        Float in [0, 1] — fraction of required elements found.
    """
    if not required_elements:
        return 1.0
    low = text.lower()
    found = sum(1 for elem in required_elements if elem in low)
    return found / len(required_elements)


def semantic_similarity(text: str, reference: str) -> float:
    """Combined semantic similarity using Jaccard + cosine similarity."""
    jaccard = token_overlap_ratio(text, reference)
    cosine = _cosine_similarity(text, reference)
    return 0.5 * jaccard + 0.5 * cosine


def trap_unresolved(task: NegotiationTask, contract_text: str) -> bool:
    """Return True if any of the task's trap markers still appear in *contract_text*."""
    low = contract_text.lower()
    return any(m in low for m in task.trap_markers)


def effective_risk_high(task: NegotiationTask, contract_text: str) -> bool:
    """Return True if the contract is still effectively high-risk for grading purposes.

    A contract is "effectively high" when:
    - Any trap marker remains unresolved, **or**
    - The weighted risk-keyword density exceeds the task's risk-level threshold.
    """
    # Any task that defines explicit trap markers (HARD, HARD_PLUS, HARD_PLUS2,
    # EXPERT, MEDIUM_PLUS) is still "effectively high risk" as long as any trap
    # marker remains in the text.
    if task.trap_markers and trap_unresolved(task, contract_text):
        return True

    hits = _weighted_risk_hits(contract_text, task.risk_keywords)

    if task.risk_level.upper() == "HIGH":
        return hits > 0.35
    if task.risk_level.upper() == "MODERATE":
        return hits > 0.42

    return False


def action_risk_alignment(action_type: str, effective_high: bool, task: NegotiationTask) -> float:
    """Score how well the chosen action type matches the current risk level ∈ [0, 1]."""
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
    """Score an agent action using a 5-dimensional rubric.

    Dimensions (weights sum to 1.0):
        - **correctness** (0.35): Risk-keyword identification or removal.
        - **improvement** (0.25): Overlap with safe keywords / expected safe edit.
        - **risk_alignment** (0.25): Whether the action type fits the risk level.
        - **semantic_similarity** (0.10): Cosine + Jaccard similarity to expected safe edit.
        - **completeness** (0.05): Required legal elements present in the rewrite.

    Returns:
        (Reward, info_dict) where info_dict contains per-dimension scores and
        an ``accept_blocked`` flag when ACCEPT is attempted on a still-risky contract.
    """

    content = (action.content or "").strip()
    eval_text = content if content else proposed_contract_text

    # Correctness: what the agent knows about the risks in the current context.
    # For FLAG_RISK / REJECT / ACCEPT: reward identifying risk keywords in the text.
    # For EDIT_CLAUSE / PROPOSE_COUNTER: reward *removing* risk keywords — a good
    # rewrite eliminates risky language, so fewer keywords = better correctness.
    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and content:
        risk_before = _weighted_risk_hits(contract_before, task.risk_keywords)
        risk_after = _weighted_risk_hits(content, task.risk_keywords)
        risk_reduction = max(0.0, risk_before - risk_after)          # ∈ [0, 1]
        token_ov = token_overlap_ratio(content, contract_before)
        correctness = min(
            1.0, 0.70 * risk_reduction + 0.30 * token_ov * task.clause_type_weight
        )
    else:
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

    # Semantic quality bonus: reward rewrites that are semantically close to the
    # expected safe edit (uses combined Jaccard + cosine similarity).
    sem_bonus = 0.0
    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and content:
        sem_bonus = semantic_similarity(content, task.expected_safe_edit)

    # Completeness bonus: reward rewrites that include required legal elements
    # defined on the task (if any).
    completeness = 0.0
    required_elems: list[str] = getattr(task, "required_elements", [])
    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and content and required_elems:
        completeness = clause_completeness_score(content, required_elems)

    # Weights: 0.35 correctness + 0.25 improvement + 0.25 risk_alignment
    #        + 0.10 semantic similarity + 0.05 completeness  (total = 1.0)
    score = (
        0.35 * correctness
        + 0.25 * improvement
        + 0.25 * risk_al
        + 0.10 * sem_bonus
        + 0.05 * completeness
    )

    # Clamp strictly between 0 and 1 → [0.001, 0.999]
    score = max(0.001, min(0.999, score))

    reward = Reward(score=round(score, 4))

    info = {
        "action_type": action.action_type,
        "grade": {
            "correctness": round(correctness, 4),
            "improvement": round(improvement, 4),
            "risk_alignment": round(risk_al, 4),
            "semantic_similarity": round(sem_bonus, 4),
            "completeness": round(completeness, 4),
        },
    }

    # Prevent invalid accept on risky contract
    if action.action_type == "ACCEPT" and eff_high:
        reward.score = 0.001
        info["accept_blocked"] = True

    return reward, info


def score_action_hypothetical(task: NegotiationTask, state_data: dict, action: Action) -> float:
    """Score an action without stepping the environment (read-only / dry-run)."""
    contract_before = state_data.get("contract_text", "")
    proposed = build_proposed_contract_for_step(contract_before, action)
    return evaluate_action(task, contract_before, action, proposed)[0].score


def build_proposed_contract_for_step(contract_before: str, action: Action) -> str:
    """Build the proposed contract text that would result from applying *action*."""
    content = (action.content or "").strip()

    if action.action_type == "EDIT_CLAUSE" and content:
        return content

    if action.action_type == "PROPOSE_COUNTER" and content:
        return f"{contract_before}\n\n[COUNTERPROPOSAL]\n{content}"

    return contract_before


def observation_risk_float(task: NegotiationTask, contract_text: str) -> float:
    """Compute the risk-level float for the observation, clamped to (0, 1)."""
    base = _weighted_risk_hits(contract_text, task.risk_keywords)

    # Boost risk observation when any task's trap markers remain unresolved
    if task.trap_markers and trap_unresolved(task, contract_text):
        base = min(1.0, base + 0.25)

    # Clamp strictly between 0 and 1 → [0.001, 0.999]
    base = min(0.999, max(0.001, base))
    return round(base, 4)


def contract_quality_score(task: NegotiationTask, contract_text: str) -> float:
    """Return a quality score for *contract_text* ∈ (0, 1), where 1 = fully safe."""
    return 1.0 - observation_risk_float(task, contract_text)


def grade_action(
    task: NegotiationTask,
    contract_before: str,
    action: Action,
    proposed_contract_text: str,
) -> Reward:
    """Convenience wrapper: grade an action and return only the Reward (drop info)."""
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    return reward


# ============ TASK-SPECIFIC GRADERS ============
# Each grader applies difficulty-specific adjustments on top of the base evaluation.

# -- Grading multipliers (named constants for clarity) --
_EASY_SAFE_EDIT_BONUS = 1.08          # +8 % for well-matched safe edits
_MEDIUM_PREMATURE_ACCEPT_PENALTY = 0.65  # −35 % for accepting risky terms
_HARD_UNRESOLVED_TRAP_PENALTY = 0.5   # −50 % when hidden traps remain
_EASY_PLUS_NOTIFICATION_BONUS = 1.06  # +6 % for breach-notification language
_HARD_PLUS_TRAP_PENALTY = 0.55        # −45 % for unresolved IP traps
_HARD_PLUS_OWNERSHIP_BONUS = 1.07     # +7 % for explicit customer-ownership
_MEDIUM_PLUS_CONFIDENTIALITY_BONUS = 1.08  # +8 % for strong confidentiality terms
_MEDIUM_PLUS_OVERBROAD_PENALTY = 0.70     # −30 % for accepting overbroad NDA
_HARD_PLUS2_TERMINATION_PENALTY = 0.55    # −45 % for accepting one-sided termination
_HARD_PLUS2_CURE_BONUS = 1.09            # +9 % for proper cure-period language
_EXPERT_DATA_PENALTY = 0.50              # −50 % for missing data-protection safeguards
_EXPERT_DATA_BONUS = 1.10               # +10 % for strong GDPR/privacy language


def grade_easy(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade easy tasks with a bias toward accepting safe-looking clauses quickly."""
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        safe = _safe_overlap(
            (action.content or "").strip(), task.safe_keywords, task.expected_safe_edit
        )
        if safe > 0.5:
            reward.score = max(0.001, min(0.999, reward.score * _EASY_SAFE_EDIT_BONUS))
    return reward


def grade_medium(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade medium tasks, penalising premature acceptance of risky auto-renewal terms."""
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    if action.action_type == "ACCEPT":
        risk = _weighted_risk_hits(proposed_contract_text, task.risk_keywords)
        if risk >= 0.3:
            reward.score = max(0.001, min(0.999, reward.score * _MEDIUM_PREMATURE_ACCEPT_PENALTY))
    return reward


def grade_hard(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade hard tasks with trap-resolution checking and heavier penalty for missed traps."""
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    if action.action_type in ("ACCEPT", "EDIT_CLAUSE", "PROPOSE_COUNTER"):
        if trap_unresolved(task, proposed_contract_text):
            reward.score = max(0.001, min(0.999, reward.score * _HARD_UNRESOLVED_TRAP_PENALTY))
    return reward


def grade_easy_plus(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade easy-plus compliance tasks, rewarding mention of notification obligations."""
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    content = (action.content or "").strip().lower()
    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        if any(kw in content for kw in ("notify", "notification", "promptly inform")):
            reward.score = max(0.001, min(0.999, reward.score * _EASY_PLUS_NOTIFICATION_BONUS))
    return reward


def grade_hard_plus(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade hard-plus IP tasks with trap-resolution + ownership-clarity checks."""
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    content = (action.content or "").strip().lower()
    if trap_unresolved(task, proposed_contract_text):
        reward.score = max(0.001, min(0.999, reward.score * _HARD_PLUS_TRAP_PENALTY))
    if any(kw in content for kw in ("customer owns", "customer-owned", "owned by customer")):
        reward.score = max(0.001, min(0.999, reward.score * _HARD_PLUS_OWNERSHIP_BONUS))
    return reward


def grade_medium_plus(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade medium-plus confidentiality/NDA tasks.

    Rewards narrowly scoped confidentiality; penalises accepting overbroad NDAs
    that lack time limits or carve-outs for public information.
    """
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    content = (action.content or "").strip().lower()

    if action.action_type == "ACCEPT":
        risk = _weighted_risk_hits(proposed_contract_text, task.risk_keywords)
        if risk >= 0.3:
            reward.score = max(0.001, min(0.999, reward.score * _MEDIUM_PLUS_OVERBROAD_PENALTY))

    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        # Reward well-scoped confidentiality rewrites
        conf_indicators = ("time limit", "expir", "carve-out", "public information", "exclude")
        if any(kw in content for kw in conf_indicators):
            reward.score = max(0.001, min(0.999, reward.score * _MEDIUM_PLUS_CONFIDENTIALITY_BONUS))

    return reward


def grade_hard_plus2(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade hard-plus-2 termination-for-convenience tasks.

    Penalises accepting one-sided termination; rewards proper cure periods,
    wind-down provisions, and mutual termination rights.
    """
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    content = (action.content or "").strip().lower()

    if action.action_type == "ACCEPT":
        if trap_unresolved(task, proposed_contract_text):
            reward.score = max(0.001, min(0.999, reward.score * _HARD_PLUS2_TERMINATION_PENALTY))

    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        # Reward proper cure-period and mutual-termination language
        cure_indicators = ("cure period", "cure-period", "30 days", "thirty", "wind-down", "mutual")
        if any(kw in content for kw in cure_indicators):
            reward.score = max(0.001, min(0.999, reward.score * _HARD_PLUS2_CURE_BONUS))
        if trap_unresolved(task, proposed_contract_text):
            reward.score = max(0.001, min(0.999, reward.score * _HARD_PLUS2_TERMINATION_PENALTY))

    return reward


def grade_expert(task: NegotiationTask, contract_before: str, action: Action, proposed_contract_text: str) -> Reward:
    """Grade expert-level data-protection / GDPR tasks.

    Penalises missing data-protection safeguards heavily; rewards clauses that
    include DPA references, data-subject rights, breach notification timelines,
    and data-minimisation language.
    """
    reward, _ = evaluate_action(task, contract_before, action, proposed_contract_text)
    content = (action.content or "").strip().lower()

    if action.action_type == "ACCEPT":
        if trap_unresolved(task, proposed_contract_text):
            reward.score = max(0.001, min(0.999, reward.score * _EXPERT_DATA_PENALTY))

    if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        gdpr_indicators = (
            "data processing agreement", "dpa", "data subject",
            "72 hours", "breach notification", "data minimisation",
            "data minimization", "sub-processor", "supervisory authority",
        )
        hits = sum(1 for kw in gdpr_indicators if kw in content)
        if hits >= 2:
            reward.score = max(0.001, min(0.999, reward.score * _EXPERT_DATA_BONUS))
        if trap_unresolved(task, proposed_contract_text):
            reward.score = max(0.001, min(0.999, reward.score * _EXPERT_DATA_PENALTY))

    return reward


# ============ GRADER REGISTRY ============
# Explicit mapping of task IDs to their grader functions
# This ensures the validator can detect that all graded tasks have graders
TASK_GRADERS = {
    "easy_unlimited_liability": grade_easy,
    "medium_auto_renewal": grade_medium,
    "hard_conflicting_obligations": grade_hard,
    "easy_compliance_agreement": grade_easy_plus,
    "hard_intellectual_property": grade_hard_plus,
    "medium_confidentiality_nda": grade_medium_plus,
    "hard_termination_convenience": grade_hard_plus2,
    "expert_data_protection": grade_expert,
}

# List of graded task IDs for validator inspection
GRADED_TASKS = list(TASK_GRADERS.keys())

# Count of tasks with graders
NUM_GRADED_TASKS = len(GRADED_TASKS)

if NUM_GRADED_TASKS < 3:
    raise ValueError(f"Expected at least 3 graded tasks, got {NUM_GRADED_TASKS}")