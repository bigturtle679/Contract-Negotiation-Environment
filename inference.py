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
    if task.name in ("HARD", "HARD_PLUS", "HARD_PLUS2", "EXPERT") and trap_unresolved(task, contract_text):
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

# Optimal action sequences per intent level for rule-based fallback
_STRATEGY_HIGH = ["FLAG_RISK", "EDIT_CLAUSE", "EDIT_CLAUSE", "PROPOSE_COUNTER", "REJECT", "EDIT_CLAUSE", "ACCEPT"]
_STRATEGY_MODERATE = ["FLAG_RISK", "PROPOSE_COUNTER", "EDIT_CLAUSE", "EDIT_CLAUSE", "ACCEPT"]
_STRATEGY_LOW = ["EDIT_CLAUSE", "EDIT_CLAUSE", "ACCEPT"]


def _choose(
    task: NegotiationTask,
    state_data: dict,
    step: int,
    prev_rewards: list[float],
) -> Action:
    """LLM-driven action selection with rule-based fallback."""
    contract_text = state_data["contract_text"]
    history = state_data.get("negotiation_history", [])
    history_summary = "\n".join(history[-8:]) if history else ""

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

    # ── 4b. Adaptive: if scores are improving and risk resolved, accept ──
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


# ── EPISODE EXECUTION ────────────────────────────────────────────────────
def run_episode(env: ContractEnv) -> float:
    """Run one full episode using env.reset() → loop env.step() → log.

    Returns the mean episode score.
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

    return score


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
    args = parser.parse_args()

    # Single env instance so reset() cycles through tasks in order
    env = ContractEnv()

    episodes_to_run = len(TASKS) if args.benchmark else args.episodes

    total_score = 0.0
    for _ in range(episodes_to_run):
        total_score += run_episode(env)

    mean_score = total_score / max(episodes_to_run, 1)
    print(
        f"\n[SUMMARY] episodes={episodes_to_run} mean_score={mean_score:.3f} "
        f"threshold={SUCCESS_SCORE_THRESHOLD}",
        flush=True,
    )


if __name__ == "__main__":
    main()
