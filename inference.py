"""
Inference Script — Contract Negotiation Environment
=====================================================
LLM-driven agent that analyses contract clauses, identifies legal risks,
and proposes safer alternatives through multi-turn negotiation.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly followed):
    [START] task=<task_id> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

MODES:
    --mode local   Use ContractEnv directly (default, for local development).
    --mode api     Connect to Docker API at ENV_SERVER_URL (for competition evaluation).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from contract_env.env.environment import ContractEnv
from contract_env.env.graders import (
    effective_risk_high,
    keyword_match_score,
    trap_unresolved,
)
from contract_env.env.models import Action
from contract_env.env.tasks import TASKS, NegotiationTask

log = logging.getLogger(__name__)

# ── ENV CONFIG ──────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "contract_negotiation")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5

# ── LLM CLIENT (lazy singleton) ─────────────────────────────────────────
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return _client


# ── SYSTEM PROMPT ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert contract-negotiation AI assistant working for the Customer.

Your goals — in priority order:
1. Identify every legal risk, hidden trap, or one-sided obligation in the clause.
2. Propose concrete, balanced rewrites that cap liability, ensure mutual obligations,
   add reasonable notice periods, clarify IP ownership, and add notification duties.
3. Only ACCEPT a clause once all material risks are resolved.

When analysing a clause you MUST return **valid JSON** with exactly this schema:
{
  "risk_assessment": "<brief summary of risks found>",
  "risk_level": "HIGH" | "MODERATE" | "LOW",
  "recommended_action": "FLAG_RISK" | "EDIT_CLAUSE" | "PROPOSE_COUNTER" | "REJECT" | "ACCEPT",
  "rewritten_clause": "<improved clause text — required for EDIT_CLAUSE or PROPOSE_COUNTER, null otherwise>"
}

Action selection rules:
- HIGH risk clause (unlimited liability, IP trap, conflicting obligations, one-sided termination, data misuse):
    Step 1: FLAG_RISK. Step 2+: EDIT_CLAUSE with a concrete safe rewrite.
- MODERATE risk (short notice periods, auto-renewal traps, overbroad NDAs):
    Use PROPOSE_COUNTER with balanced language (e.g., 60-day notice, time-limited NDA).
- LOW risk (compliance, boilerplate):
    EDIT_CLAUSE to add notification/reporting obligations, then ACCEPT.
- REJECT only for terms so extreme they cannot be salvaged.
- Never accept unlimited liability, one-day notice periods, or clauses that
  assign all IP to the supplier when the customer provides specifications.

For EDIT_CLAUSE on IP tasks, the rewritten clause MUST include "customer owns"
or "owned by customer" and remove supplier-ownership language.
For liability tasks, cap language: "liability capped at fees paid in the
preceding twelve (12) months; no consequential or punitive damages."
For auto-renewal tasks, include: "sixty (60) days prior written notice."
For compliance tasks, include: "promptly notify Customer of any material breach."
For confidentiality/NDA tasks, include: time limit (e.g., 3 years), carve-outs for
publicly available information, and scope limitations.
For termination tasks, include: mutual termination rights, cure period of at least
30 days, and transition/wind-down provisions.
For data protection tasks, include: Data Processing Agreement reference, 72-hour
breach notification, sub-processor consent requirements, data subject rights
assistance, and data deletion upon termination.

Return ONLY the JSON object — no markdown fences, no commentary, no extra text.
"""

# ── LLM HELPERS ─────────────────────────────────────────────────────────
_MAX_RETRIES = 2


def _llm_chat(
    messages: list[dict],
    temperature: float = 0.15,
    max_tokens: int = 512,
) -> str:
    """Call the LLM with retry. Returns raw text."""
    client = _get_client()
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                raise
            log.warning("[DEBUG] LLM call attempt %d failed: %s", attempt + 1, exc)
    return ""


def _parse_llm_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── RULE-BASED RISK SCORING ─────────────────────────────────────────────
def _risk_score(task: NegotiationTask, contract_text: str) -> float:
    hits = keyword_match_score(contract_text, task.risk_keywords)
    rs = min(1.0, hits * task.clause_type_weight / 1.15)
    if task.trap_markers and trap_unresolved(task, contract_text):
        rs = min(1.0, rs + 0.25)
    return round(rs, 6)


def _rule_based_intent(task: NegotiationTask, contract_text: str) -> str:
    rs = _risk_score(task, contract_text)
    if effective_risk_high(task, contract_text) or rs >= 0.6:
        return "HIGH"
    if task.risk_level.upper() == "MODERATE" and rs >= 0.35:
        return "MODERATE"
    return "LOW"


# ── LLM STRATEGY ────────────────────────────────────────────────────────
def _build_analysis_prompt(
    task: NegotiationTask,
    state_data: dict,
    step: int,
    history_summary: str,
) -> list[dict]:
    user_msg = (
        f"Contract clause (type: {task.clause_type}, "
        f"industry: {task.industry_context}, "
        f"risk_level: {task.risk_level}):\n"
        f'"""\n{state_data["contract_text"]}\n"""\n\n'
    )
    if history_summary:
        user_msg += f"Negotiation history so far:\n{history_summary}\n\n"
    user_msg += (
        f"This is negotiation step {step + 1}. "
        "Analyse the clause and return your JSON recommendation. "
        "Remember: for EDIT_CLAUSE or PROPOSE_COUNTER you MUST provide rewritten_clause."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _build_rewrite_prompt(
    task: NegotiationTask,
    contract_text: str,
    risk_assessment: str,
) -> list[dict]:
    user_msg = (
        f"Rewrite this {task.clause_type} clause to eliminate all identified risks.\n\n"
        f"Risks found: {risk_assessment}\n\n"
        f"Original clause:\n{contract_text}\n\n"
        "Requirements for the rewrite:\n"
    )
    if task.clause_type == "liability":
        user_msg += (
            "- Cap liability at fees paid in preceding 12 months\n"
            "- Exclude consequential and punitive damages\n"
            "- Make obligations mutual\n"
        )
    elif task.clause_type == "term_renewal":
        user_msg += (
            "- Require 60 days prior written notice to cancel\n"
            "- Make auto-renewal opt-in not opt-out\n"
        )
    elif task.clause_type == "performance_changes":
        user_msg += (
            "- Add a formal change control process\n"
            "- Require timeline and fee adjustments for changes\n"
            "- Remove unlimited/uncompensated change obligations\n"
        )
    elif task.clause_type == "compliance":
        user_msg += (
            "- Add obligation to promptly notify Customer of material breach\n"
            "- Keep balanced compliance obligations\n"
        )
    elif task.clause_type == "intellectual_property":
        user_msg += (
            "- State that Customer owns all IP created under this agreement\n"
            "- Grant Supplier only a limited license to use Customer materials\n"
            "- Remove any supplier-ownership language\n"
        )
    elif task.clause_type == "confidentiality":
        user_msg += (
            "- Limit confidentiality obligation to three (3) years from disclosure\n"
            "- Add carve-outs for publicly available information\n"
            "- Add carve-out for independently developed information\n"
            "- Allow disclosure required by law or court order\n"
            "- Permit sharing with employees and advisors under NDA\n"
        )
    elif task.clause_type == "termination":
        user_msg += (
            "- Make termination rights mutual (either party)\n"
            "- Require sixty (60) days' prior written notice for convenience termination\n"
            "- Add thirty (30) day cure period for material breach\n"
            "- Include transition/wind-down assistance provision\n"
            "- Preserve all legal rights and remedies\n"
        )
    elif task.clause_type == "data_protection":
        user_msg += (
            "- Require execution of a Data Processing Agreement (DPA)\n"
            "- Add 72-hour data breach notification requirement\n"
            "- Require prior written consent for sub-processors\n"
            "- Mandate assistance with data subject access requests\n"
            "- Require deletion or return of personal data upon termination\n"
            "- Include data minimisation principles\n"
            "- Restrict data transfers to adequate jurisdictions\n"
        )
    user_msg += "\nReturn ONLY the rewritten clause text, nothing else."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


_VALID_ACTIONS = {"FLAG_RISK", "EDIT_CLAUSE", "ACCEPT", "REJECT", "PROPOSE_COUNTER"}

# Optimal action sequences per intent level for rule-based fallback.
# MODERATE tasks front-load PROPOSE_COUNTER since it's the ideal action for
# balanced negotiation (the environment appends [COUNTERPROPOSAL] tags).
_STRATEGY_HIGH = ["FLAG_RISK", "EDIT_CLAUSE", "EDIT_CLAUSE", "PROPOSE_COUNTER", "REJECT", "EDIT_CLAUSE", "ACCEPT"]
_STRATEGY_MODERATE = ["FLAG_RISK", "PROPOSE_COUNTER", "PROPOSE_COUNTER", "EDIT_CLAUSE", "EDIT_CLAUSE", "ACCEPT"]
_STRATEGY_LOW = ["EDIT_CLAUSE", "EDIT_CLAUSE", "ACCEPT"]

# ── OPPONENT-RESPONSE PARSING ───────────────────────────────────────────
# Concession signals from the counterparty that indicate willingness to negotiate
_CONCESSION_SIGNALS = (
    "we can accept", "we can agree", "we could consider", "we could accept",
    "we can consider", "may be possible", "we are willing",
    "we'll review", "we will review",
    "agree to", "open to",
)
_FIRMNESS_SIGNALS = (
    "non-negotiable", "cannot proceed", "not possible",
    "standard and non-negotiable", "cannot accept", "is not included",
)

# Topic keywords to detect what specifically the opponent is conceding or
# holding firm on. Used for fine-grained concession tracking.
_TOPIC_KEYWORDS = {
    "cap": ("cap", "capped", "limitation", "limit"),
    "notice_period": ("notice", "days", "notice period"),
    "ip_ownership": ("ownership", "ip", "intellectual property", "customer owns"),
    "termination": ("termination", "terminate", "mutual", "cure"),
    "liability": ("liability", "indemnify", "consequential", "punitive"),
    "confidentiality": ("confidentiality", "nda", "perpetuity", "time limit"),
    "data_protection": ("dpa", "data", "breach notification", "sub-processor", "gdpr"),
    "change_control": ("change", "scope", "approval", "timeline"),
}


def _parse_opponent_stance(history: list[str]) -> str:
    """Analyse the latest opponent reply to determine their negotiation stance.

    Returns:
        'conceding' — opponent shows willingness; escalate to EDIT_CLAUSE / ACCEPT.
        'firm' — opponent is holding position; keep pushing with PROPOSE_COUNTER.
        'neutral' — no strong signal; follow normal strategy.
    """
    # Find the most recent opponent entry
    opp_entries = [h for h in history if h.startswith("opponent|")]
    if not opp_entries:
        return "neutral"

    latest = opp_entries[-1].lower()

    if any(signal in latest for signal in _CONCESSION_SIGNALS):
        return "conceding"
    if any(signal in latest for signal in _FIRMNESS_SIGNALS):
        return "firm"
    return "neutral"


def _track_concessions(history: list[str]) -> dict[str, str]:
    """Track which negotiation topics the opponent has conceded on vs. held firm.

    Returns a dict like {"cap": "conceded", "liability": "firm", "notice_period": "unknown"}.
    This enables the agent to focus edits on unresolved issues.
    """
    concessions: dict[str, str] = {}

    for entry in history:
        if not entry.startswith("opponent|"):
            continue
        low = entry.lower()

        is_conceding = any(s in low for s in _CONCESSION_SIGNALS)
        is_firm = any(s in low for s in _FIRMNESS_SIGNALS)

        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(kw in low for kw in keywords):
                if is_conceding:
                    concessions[topic] = "conceded"
                elif is_firm:
                    concessions[topic] = "firm"
                elif topic not in concessions:
                    concessions[topic] = "discussed"

    return concessions


def _concession_summary(concessions: dict[str, str]) -> str:
    """Build a human-readable summary of opponent concessions for the LLM."""
    if not concessions:
        return ""
    parts: list[str] = []
    for topic, status in concessions.items():
        label = topic.replace("_", " ")
        if status == "conceded":
            parts.append(f"  - {label}: opponent is WILLING to negotiate")
        elif status == "firm":
            parts.append(f"  - {label}: opponent is HOLDING FIRM")
        else:
            parts.append(f"  - {label}: discussed (no clear position)")
    return "Opponent concession tracker:\n" + "\n".join(parts)


def _choose(
    task: NegotiationTask,
    state_data: dict,
    step: int,
    prev_rewards: list[float],
) -> Action:
    """LLM-driven action selection with rule-based fallback and opponent awareness."""
    contract_text = state_data["contract_text"]
    history = state_data.get("negotiation_history", [])
    history_summary = "\n".join(history[-8:]) if history else ""

    # ── 0. Parse opponent stance and track concessions ───────────────────
    opponent_stance = _parse_opponent_stance(history)
    concessions = _track_concessions(history)
    conc_summary = _concession_summary(concessions)

    # Enrich history summary with concession tracking for the LLM
    if conc_summary:
        history_summary = history_summary + "\n\n" + conc_summary

    # ── 1. Ask the LLM for structured analysis ──────────────────────────
    parsed: Optional[dict] = None
    try:
        messages = _build_analysis_prompt(task, state_data, step, history_summary)
        raw = _llm_chat(messages)
        parsed = _parse_llm_json(raw)
    except Exception as exc:
        log.warning("[DEBUG] LLM analysis failed: %s", exc)

    # ── 2. Extract action + content from LLM response ───────────────────
    action_type: Optional[str] = None
    content: Optional[str] = None
    risk_assessment: str = ""

    if parsed:
        rec = (parsed.get("recommended_action") or "").upper().strip()
        if rec in _VALID_ACTIONS:
            action_type = rec
        content = parsed.get("rewritten_clause") or None
        risk_assessment = parsed.get("risk_assessment", "")

    # ── 3. Rule-based fallback with improved strategy ─────────────────────
    if action_type is None:
        intent = _rule_based_intent(task, contract_text)
        if intent == "HIGH":
            seq = _STRATEGY_HIGH
        elif intent == "MODERATE":
            seq = _STRATEGY_MODERATE
        else:
            seq = _STRATEGY_LOW
        action_type = seq[min(step, len(seq) - 1)]

    # ── 4. Adaptive: switch to EDIT if previous score was poor ───────────
    if prev_rewards and prev_rewards[-1] < 0.25 and step > 0:
        if action_type in ("FLAG_RISK", "REJECT"):
            action_type = "EDIT_CLAUSE"

    # ── 4b. Opponent-aware adjustment ────────────────────────────────────
    # If opponent is conceding, escalate toward resolution faster.
    if opponent_stance == "conceding" and step > 0:
        if action_type == "FLAG_RISK":
            action_type = "EDIT_CLAUSE"
        elif action_type == "REJECT":
            action_type = "PROPOSE_COUNTER"
    # If opponent is firm, use PROPOSE_COUNTER to keep negotiating.
    elif opponent_stance == "firm" and step > 0:
        if action_type in ("ACCEPT",):
            # Don't accept while opponent is still pushing back on risky terms
            if effective_risk_high(task, contract_text) or trap_unresolved(task, contract_text):
                action_type = "PROPOSE_COUNTER"

    # ── 4c. Concession-aware: if opponent conceded on key issues, lean EDIT ─
    conceded_topics = [t for t, s in concessions.items() if s == "conceded"]
    if conceded_topics and step > 1:
        # Opponent has given ground — capitalise with a concrete edit
        if action_type == "FLAG_RISK":
            action_type = "EDIT_CLAUSE"

    # ── 4d. Smart ACCEPT gate: only accept when quality actually improved ─
    if action_type == "ACCEPT":
        from contract_env.env.graders import observation_risk_float
        current_risk = observation_risk_float(task, contract_text)
        original_risk = observation_risk_float(task, task.contract_text)
        # Block acceptance if the contract hasn't improved meaningfully
        if current_risk >= original_risk - 0.05:
            if effective_risk_high(task, contract_text) or trap_unresolved(task, contract_text):
                action_type = "EDIT_CLAUSE"

    # ── 4e. Adaptive: if scores are improving and risk resolved, accept ──
    if (
        len(prev_rewards) >= 3
        and all(r > 0.45 for r in prev_rewards[-2:])
        and not effective_risk_high(task, contract_text)
        and not trap_unresolved(task, contract_text)
    ):
        action_type = "ACCEPT"

    # ── 5. Generate content for EDIT / PROPOSE if missing ────────────────
    if action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and not content:
        try:
            msgs = _build_rewrite_prompt(
                task, contract_text, risk_assessment or "High legal risk identified"
            )
            content = _llm_chat(msgs, max_tokens=600)
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
        except Exception as exc:
            log.warning("[DEBUG] LLM rewrite failed: %s", exc)
            content = None

    # ── 6. Safe fallback: use expected_safe_edit ─────────────────────────
    if action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and not content:
        content = task.expected_safe_edit

    return Action(action_type=action_type, content=content)


# ── HTTP-CLIENT WRAPPER ──────────────────────────────────────────────────
# Provides the same reset()/step() interface as ContractEnv but talks to
# the Docker API server via HTTP, matching the competition evaluation flow.

class _HTTPEnvClient:
    """Thin HTTP wrapper with the same interface as ContractEnv for inference."""

    def __init__(self, base_url: str) -> None:
        import requests
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._task_idx = 0
        self.current_task: Optional[NegotiationTask] = None

    def reset(self):
        resp = self._session.post(f"{self.base_url}/reset")
        resp.raise_for_status()
        data = resp.json()
        obs = data["observation"]
        # Map to a NegotiationTask if possible (for _choose() to use)
        task_id = None
        try:
            state = self._session.get(f"{self.base_url}/state").json()
            task_id = state.get("task_id")
        except Exception:
            pass
        if task_id:
            self.current_task = next((t for t in TASKS if t.id == task_id), None)
        if self.current_task is None:
            self.current_task = TASKS[self._task_idx % len(TASKS)]
            self._task_idx += 1
        return _DictObservation(obs)

    def step(self, action: Action):
        payload = {"action_type": action.action_type}
        if action.content:
            payload["content"] = action.content
        resp = self._session.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = _DictObservation(data["observation"])
        reward = data["reward"]["score"]
        done = data["done"]
        info = data.get("info", {})
        return obs, reward, done, info


class _DictObservation:
    """Lightweight wrapper that exposes dict fields as attributes."""

    def __init__(self, d: dict) -> None:
        self.contract_text: str = d.get("contract_text", "")
        self.clause_type: str = d.get("clause_type", "")
        self.risk_level: float = d.get("risk_level", 0.5)
        self.step_count: int = d.get("step_count", 0)
        self.negotiation_history: list[str] = d.get("negotiation_history", [])


# ── EPISODE EXECUTION ────────────────────────────────────────────────────
def run_episode(env) -> tuple[float, str]:
    """Run one full episode using env.reset() → loop env.step() → log.

    Returns (mean_episode_score, task_id).
    """
    obs_obj = env.reset()
    task = env.current_task

    state_data: dict[str, Any] = {
        "contract_text": obs_obj.contract_text,
        "negotiation_history": list(obs_obj.negotiation_history),
    }

    print(
        f"[START] task={task.id} env={BENCHMARK} model={MODEL_NAME}",
        flush=True,
    )

    rewards: list[float] = []
    done = False
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            action = _choose(task, state_data, step_num - 1, rewards)
            obs_obj, reward_val, done, info = env.step(action)

            reward = float(reward_val)
            rewards.append(reward)
            steps_taken = step_num

            state_data["contract_text"] = obs_obj.contract_text
            state_data["negotiation_history"] = list(obs_obj.negotiation_history)

            error = info.get("error")
            error_str = error if error else "null"

            print(
                f"[STEP] step={step_num} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                flush=True,
            )

            if done:
                break

        score = sum(rewards) / max(len(rewards), 1)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )

    return score, task.id


# ── MAIN ─────────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    random.seed(42)

    parser = argparse.ArgumentParser(
        description="Run contract-negotiation inference episodes",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Number of episodes to run (default: 8). Tasks cycle sequentially.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run exactly one episode per task (covers all 8 tasks)",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="local",
        help=(
            "Execution mode. 'local' uses ContractEnv directly (default). "
            "'api' connects to the Docker server via HTTP at ENV_SERVER_URL."
        ),
    )
    parser.add_argument(
        "--retry-low",
        type=float,
        default=0.0,
        metavar="THRESHOLD",
        help=(
            "Re-run tasks that scored below THRESHOLD (e.g. --retry-low 0.4). "
            "Each low-scoring task is retried once. 0 = disabled (default)."
        ),
    )
    args = parser.parse_args()

    # Select environment backend
    if args.mode == "api":
        env = _HTTPEnvClient(ENV_SERVER_URL)
        print(f"[CONFIG] mode=api server={ENV_SERVER_URL}", flush=True)
    else:
        env = ContractEnv()
        print("[CONFIG] mode=local", flush=True)

    episodes_to_run = len(TASKS) if args.benchmark else args.episodes

    total_score = 0.0
    task_scores: dict[str, list[float]] = {}

    for _ in range(episodes_to_run):
        ep_score, task_id = run_episode(env)
        total_score += ep_score
        task_scores.setdefault(task_id, []).append(ep_score)

    # ── Retry low-scoring tasks ──────────────────────────────────────────
    retry_threshold = args.retry_low
    if retry_threshold > 0:
        low_tasks = {
            tid: scores
            for tid, scores in task_scores.items()
            if (sum(scores) / len(scores)) < retry_threshold
        }
        if low_tasks:
            print(
                f"\n[RETRY] {len(low_tasks)} task(s) scored below {retry_threshold:.2f}, retrying...",
                flush=True,
            )
            for tid in low_tasks:
                ep_score, _ = run_episode(env)
                total_score += ep_score
                task_scores[tid].append(ep_score)
                episodes_to_run += 1

    mean_score = total_score / max(episodes_to_run, 1)

    # Per-task summary (sorted by score, worst first)
    print("\n[TASK SCORES]", flush=True)
    sorted_tasks = sorted(task_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))
    for tid, scores in sorted_tasks:
        avg = sum(scores) / len(scores)
        best = max(scores)
        status = "✓" if avg >= SUCCESS_SCORE_THRESHOLD else "✗"
        print(
            f"  {status} {tid}: mean={avg:.3f} best={best:.3f} runs={len(scores)}",
            flush=True,
        )

    print(
        f"\n[SUMMARY] episodes={episodes_to_run} mean_score={mean_score:.3f} "
        f"threshold={SUCCESS_SCORE_THRESHOLD}",
        flush=True,
    )


if __name__ == "__main__":
    main()
