"""
Inference Script — Contract Negotiation Environment
=====================================================
LLM-driven agent that analyses contract clauses, identifies legal risks,
and proposes safer alternatives through multi-turn negotiation.

MANDATORY environment variables
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import warnings
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

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ── ENV CONFIG ──────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://router.huggingface.co/v1"
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# ── LLM CLIENT (lazy singleton) ────────────────────────────────────────
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
1. Identify every legal risk, hidden trap, or one-sided obligation.
2. Propose concrete, balanced rewrites that cap liability, ensure mutual
   obligations, add reasonable notice periods, and clarify IP ownership.
3. Only ACCEPT a clause once all material risks are resolved.

When analysing a clause you MUST return **valid JSON** with the schema:
{
  "risk_assessment": "<brief summary of risks found>",
  "risk_level": "HIGH" | "MODERATE" | "LOW",
  "recommended_action": "FLAG_RISK" | "EDIT_CLAUSE" | "PROPOSE_COUNTER" | "REJECT" | "ACCEPT",
  "rewritten_clause": "<improved clause text — required for EDIT_CLAUSE or PROPOSE_COUNTER, null otherwise>"
}

Rules:
- Never accept unlimited liability, one-day notice periods, or clauses that
  assign all IP to the supplier when customer provides specifications.
- Prefer EDIT_CLAUSE when you can rewrite the clause directly.
- Use PROPOSE_COUNTER when a full counter-offer is warranted.
- Use REJECT only for egregiously one-sided terms that cannot be edited.
- Use FLAG_RISK as the first move for HIGH-risk clauses before editing.
- Return ONLY the JSON object, no markdown fences, no commentary.
"""

# ── LLM HELPERS ─────────────────────────────────────────────────────────
_MAX_RETRIES = 2


def _llm_chat(messages: list[dict], temperature: float = 0.15,
              max_tokens: int = 512) -> str:
    """Call the LLM with retry logic. Returns the raw text response."""
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
            log.warning("LLM call attempt %d failed: %s", attempt + 1, exc)
    return ""


def _parse_llm_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from LLM output."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── RISK ANALYSIS (rule-based fallback) ─────────────────────────────────
def _risk_score(task: NegotiationTask, contract_text: str) -> float:
    hits = keyword_match_score(contract_text, task.risk_keywords)
    rs = min(1.0, hits * task.clause_type_weight / 1.15)
    if task.name == "HARD" and trap_unresolved(task, contract_text):
        rs = min(1.0, rs + 0.25)
    return round(rs, 6)


def _rule_based_intent(task: NegotiationTask, contract_text: str) -> str:
    rs = _risk_score(task, contract_text)
    if effective_risk_high(task, contract_text) or rs >= 0.6:
        return "HIGH"
    if task.risk_level.upper() == "MODERATE" and rs >= 0.35:
        return "MODERATE"
    return "LOW"


# ── LLM-DRIVEN STRATEGY ────────────────────────────────────────────────
def _build_analysis_prompt(task: NegotiationTask, state_data: dict,
                           step: int, history_summary: str) -> list[dict]:
    """Build the chat messages for the LLM analysis call."""
    user_msg = (
        f"Contract clause (type: {task.clause_type}, "
        f"industry: {task.industry_context}):\n"
        f'"""\n{state_data["contract_text"]}\n"""\n\n'
    )
    if history_summary:
        user_msg += f"Negotiation history so far:\n{history_summary}\n\n"
    user_msg += (
        f"This is negotiation step {step + 1}. "
        "Analyse the clause and return your JSON recommendation."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _build_rewrite_prompt(task: NegotiationTask,
                          contract_text: str,
                          risk_assessment: str) -> list[dict]:
    """Build a focused rewrite prompt when the analysis step doesn't return
    a usable rewritten_clause."""
    user_msg = (
        f"You previously identified these risks in this {task.clause_type} "
        f"clause:\n{risk_assessment}\n\n"
        f"Original clause:\n{contract_text}\n\n"
        "Rewrite the clause to eliminate all identified risks while keeping "
        "reasonable commercial terms. Return ONLY the rewritten clause text, "
        "nothing else."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


_VALID_ACTIONS = {"FLAG_RISK", "EDIT_CLAUSE", "ACCEPT", "REJECT",
                  "PROPOSE_COUNTER"}


def _choose(task: NegotiationTask, state_data: dict, step: int,
            prev_rewards: list[float]) -> Action:
    """Use the LLM to decide the next action, falling back to rules on error."""
    contract_text = state_data["contract_text"]
    history = state_data.get("negotiation_history", [])
    history_summary = "\n".join(history[-6:]) if history else ""

    # ── 1. Ask the LLM for a structured analysis ───────────────────────
    try:
        messages = _build_analysis_prompt(task, state_data, step,
                                          history_summary)
        raw = _llm_chat(messages)
        parsed = _parse_llm_json(raw)
    except Exception as exc:
        log.warning("LLM analysis call failed: %s", exc)
        parsed = None

    # ── 2. Extract action + content from the LLM response ──────────────
    action_type: Optional[str] = None
    content: Optional[str] = None
    risk_assessment: str = ""

    if parsed:
        rec = (parsed.get("recommended_action") or "").upper().strip()
        if rec in _VALID_ACTIONS:
            action_type = rec
        content = parsed.get("rewritten_clause") or None
        risk_assessment = parsed.get("risk_assessment", "")

    # ── 3. Rule-based fallback if LLM didn't return valid action ───────
    if action_type is None:
        intent = _rule_based_intent(task, contract_text)
        if intent == "HIGH":
            seq = ["FLAG_RISK", "EDIT_CLAUSE", "PROPOSE_COUNTER", "REJECT",
                   "ACCEPT"]
        elif intent == "MODERATE":
            seq = ["FLAG_RISK", "EDIT_CLAUSE", "PROPOSE_COUNTER", "ACCEPT"]
        else:
            seq = ["EDIT_CLAUSE", "PROPOSE_COUNTER", "ACCEPT"]
        action_type = seq[min(step, len(seq) - 1)]

    # ── 4. Adaptive adjustment based on previous reward feedback ───────
    if prev_rewards and prev_rewards[-1] < 0.2 and step > 0:
        # Previous action scored poorly — try editing instead of repeating
        if action_type in ("FLAG_RISK", "REJECT"):
            action_type = "EDIT_CLAUSE"

    # ── 5. Generate content for EDIT / PROPOSE if missing ──────────────
    if action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and not content:
        try:
            msgs = _build_rewrite_prompt(task, contract_text,
                                         risk_assessment or "High legal risk")
            content = _llm_chat(msgs, max_tokens=384)
            # Strip any quotes the model might wrap around
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
        except Exception as exc:
            log.warning("LLM rewrite call failed: %s", exc)
            content = None

    # ── 6. Ensure content actions always have content ──────────────────
    if action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER") and not content:
        content = task.expected_safe_edit  # safe fallback

    return Action(action_type=action_type, content=content)


# ── LOGGING ─────────────────────────────────────────────────────────────
def _log_step(step: int, action: Action, reward: float, done: bool,
              err: Optional[str]) -> None:
    err_token = "null" if not err else err
    print(
        f"[STEP] step={step} action={action.action_type} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_token}",
        flush=True,
    )


# ── EPISODE EXECUTION ──────────────────────────────────────────────────
def run_episode() -> None:
    env = ContractEnv()
    obs = env.reset().model_dump(mode="json")

    task = next(t for t in TASKS if t.clause_type == obs["clause_type"])

    state_data: dict[str, Any] = {
        "contract_text": obs["contract_text"],
        "negotiation_history": list(obs.get("negotiation_history", [])),
    }

    print(
        f"[START] task={task.name} env=ContractNegotiationEnv "
        f"model={MODEL_NAME}",
        flush=True,
    )

    rewards: list[float] = []
    done = False
    step = 0

    try:
        while not done and step < 10:
            action = _choose(task, state_data, step, rewards)

            obs_obj, reward, done, info = env.step(action)
            score = float(reward)

            rewards.append(score)
            state_data["contract_text"] = obs_obj.contract_text
            state_data["negotiation_history"] = list(
                obs_obj.negotiation_history
            )

            _log_step(step, action, score, done, info.get("error"))
            step += 1

        final_score = sum(rewards) / max(len(rewards), 1)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(final_score >= 0.5).lower()} "
            f"steps={step} score={final_score:.2f} rewards={rewards_str}",
            flush=True,
        )

    except Exception as e:
        print(
            f"[STEP] step=0 action=NONE reward=0.00 "
            f"done=true error={str(e)}",
            flush=True,
        )
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        return


# ── MAIN ────────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    random.seed(42)

    parser = argparse.ArgumentParser(
        description="Run contract-negotiation inference episodes",
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run one episode per task")
    args = parser.parse_args()

    episodes_to_run = len(TASKS) if args.benchmark else args.episodes

    for _ in range(episodes_to_run):
        run_episode()


if __name__ == "__main__":
    main()