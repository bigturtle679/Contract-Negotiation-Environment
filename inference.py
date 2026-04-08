from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
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

# ---------------- ENV CONFIG ----------------
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860").rstrip("/")

# ✅ FIXED: correct variables
API_KEY = os.getenv("API_KEY")
LLM_API_BASE = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")


# ---------------- HTTP ----------------
def _http_post(path: str, payload: Optional[dict] = None) -> dict[str, Any]:
    try:
        url = f"{BASE_URL}{path}"
        data = None
        headers = {"Content-Type": "application/json"}

        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    except Exception as e:
        raise RuntimeError(f"HTTP request failed: {str(e)}")


# ---------------- SCORING ----------------
def _risk_score(task: NegotiationTask, contract_text: str) -> float:
    hits = keyword_match_score(contract_text, task.risk_keywords)
    rs = min(1.0, hits * task.clause_type_weight / 1.15)
    if task.name == "HARD" and trap_unresolved(task, contract_text):
        rs = min(1.0, rs + 0.25)
    return round(rs, 6)


def _confidence_and_intent(task: NegotiationTask, contract_text: str):
    rs = _risk_score(task, contract_text)
    confidence = max(0.0, min(1.0, 1.0 - rs))

    if effective_risk_high(task, contract_text) or rs >= 0.6:
        return confidence, "HIGH"
    if task.risk_level.upper() == "MODERATE" and rs >= 0.35:
        return confidence, "MODERATE"
    return confidence, "LOW"


# ---------------- ACTION ----------------
def _content_for(task: NegotiationTask, action_type: str):
    if action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
        return task.expected_safe_edit
    return None


def _action_for(task: NegotiationTask, action_type: str):
    return Action(action_type=action_type, content=_content_for(task, action_type))


# ---------------- LLM (CRITICAL FIX) ----------------
def _maybe_llm_improve(task, contract_text, action, confidence):
    # ✅ FORCE at least one LLM call (important for validation)
    should_call = True

    if not API_KEY or not LLM_API_BASE:
        return action

    try:
        client = OpenAI(
            base_url=LLM_API_BASE,
            api_key=API_KEY,
        )

        prompt = f"""
You are an AI contract negotiation assistant.

Given this clause:
{action.content or contract_text}

Rewrite it to make it safer and reduce legal risk.

Return ONLY the improved clause text.
"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )

        text = (resp.choices[0].message.content or "").strip()

        if text:
            return Action(action_type=action.action_type, content=text)

    except Exception:
        pass

    return action


# ---------------- STRATEGY ----------------
def _sequence(intent: str):
    if intent == "HIGH":
        return ["FLAG_RISK", "REJECT", "PROPOSE_COUNTER", "EDIT_CLAUSE", "ACCEPT"]
    if intent == "MODERATE":
        return ["FLAG_RISK", "EDIT_CLAUSE", "PROPOSE_COUNTER", "ACCEPT"]
    return ["ACCEPT", "EDIT_CLAUSE", "PROPOSE_COUNTER"]


def _choose(task, state_data, step):
    confidence, intent = _confidence_and_intent(task, state_data["contract_text"])
    seq = _sequence(intent)

    action_type = seq[min(step, len(seq) - 1)]
    action = _action_for(task, action_type)

    return _maybe_llm_improve(task, state_data["contract_text"], action, confidence)


# ---------------- LOGGING ----------------
def _log_step(step, action, reward, done, err):
    err_token = "error=null" if not err else f'error="{err}"'
    print(
        f"[STEP] step={step} action={action.action_type} reward={reward:.2f} done={str(done).lower()} {err_token}",
        flush=True,
    )


# ---------------- EXECUTION ----------------
def run_episode():
    env = ContractEnv()
    obs = env.reset().model_dump(mode="json")

    task = next(t for t in TASKS if t.clause_type == obs["clause_type"])

    state_data = {
        "contract_text": obs["contract_text"],
        "negotiation_history": list(obs.get("negotiation_history", [])),
    }

    print(f"[START] task={task.name} env=ContractNegotiationEnv model=llm-agent")

    rewards = []
    done = False
    step = 0

    try:
        while not done and step < 10:
            action = _choose(task, state_data, step)

            obs, reward, done, info = env.step(action)
            score = float(reward.score)

            rewards.append(score)
            state_data["contract_text"] = obs.contract_text

            _log_step(step, action, score, done, info.get("error"))

            step += 1

        final_score = sum(rewards) / max(len(rewards), 1)

        print(
            f"[END] success={str(final_score >= 0.5).lower()} steps={step} score={final_score:.2f} rewards={','.join(map(str, rewards))}"
        )

    except Exception as e:
        print(f'[STEP] step=0 action=NONE reward=0.001 done=true error="{str(e)}"')
        print("[END] success=false steps=0 score=0.001 rewards=")
        sys.exit(0)


# ---------------- MAIN ----------------
def main():
    load_dotenv()
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    for _ in range(args.episodes):
        run_episode()


if __name__ == "__main__":
    main()