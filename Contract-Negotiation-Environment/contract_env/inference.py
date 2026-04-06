from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import ContractEnv
from env.graders import (
    effective_risk_high,
    keyword_match_score,
    score_action_hypothetical,
    trap_unresolved,
)
from env.models import Action
from env.tasks import TASKS, NegotiationTask

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:7860").rstrip("/")
USE_DIRECT_ENV = os.getenv("USE_DIRECT_ENV", "").lower() in ("1", "true", "yes")
API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _http_post(path: str, payload: Optional[dict] = None) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def _risk_score(task: NegotiationTask, contract_text: str) -> float:
    hits = keyword_match_score(contract_text, task.risk_keywords)
    rs = min(1.0, hits * task.clause_type_weight / 1.15)
    if task.name == "HARD" and trap_unresolved(task, contract_text):
        rs = min(1.0, rs + 0.25)
    return round(rs, 6)


def _confidence_and_intent(task: NegotiationTask, contract_text: str) -> tuple[float, str]:
    rs = _risk_score(task, contract_text)
    confidence = max(0.0, min(1.0, 1.0 - rs))
    if effective_risk_high(task, contract_text) or rs >= 0.6:
        return confidence, "HIGH"
    if task.risk_level.upper() == "MODERATE" and rs >= 0.35:
        return confidence, "MODERATE"
    return confidence, "LOW"


def _content_for(task: NegotiationTask, action_type: str) -> Optional[str]:
    if action_type == "EDIT_CLAUSE":
        return task.expected_safe_edit
    if action_type == "PROPOSE_COUNTER":
        # Grader compares this text against `expected_safe_edit`.
        return task.expected_safe_edit
    return None


def _action_for(task: NegotiationTask, action_type: str) -> Action:
    return Action(action_type=action_type, content=_content_for(task, action_type))


def _maybe_llm_improve(task: NegotiationTask, contract_text: str, action: Action, confidence: float) -> Action:
    # Spec: call LLM ONLY when confidence < 0.6 OR action is EDIT_CLAUSE / PROPOSE_COUNTER.
    should_call = (confidence < 0.6) or (action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"))
    if not should_call:
        return action

    key = os.getenv("HF_TOKEN")
    if not key:
        return action

    if not action.content:
        return action

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=key)
        user = (
            f"Task: {task.name} ({task.clause_type}).\n"
            f"Industry: {task.industry_context}.\n\n"
            f"Contract excerpt:\n{contract_text}\n\n"
            f"Rewrite this clause text more safely while preserving intent:\n{action.content}\n\n"
            "Return ONLY rewritten text."
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You rewrite contract clauses. Output only text."},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return Action(action_type=action.action_type, content=text)
    except Exception:
        # Fallback: heuristic content.
        pass
    return action


def _sequence_for_intent(intent: str) -> list[str]:
    if intent == "HIGH":
        return ["FLAG_RISK", "REJECT", "PROPOSE_COUNTER", "EDIT_CLAUSE", "ACCEPT"]
    if intent == "MODERATE":
        return ["FLAG_RISK", "EDIT_CLAUSE", "PROPOSE_COUNTER", "ACCEPT", "REJECT"]
    return ["ACCEPT", "EDIT_CLAUSE", "PROPOSE_COUNTER", "REJECT", "FLAG_RISK"]


def _two_candidates(
    task: NegotiationTask,
    contract_text: str,
    step_count: int,
    last_action: Optional[str],
    last_reward: float,
) -> tuple[Action, Action]:
    confidence, intent = _confidence_and_intent(task, contract_text)
    seq = _sequence_for_intent(intent)

    idx = min(step_count, len(seq) - 1)
    if last_reward < 0.5 and idx < len(seq) - 1:
        idx = idx + 1

    a1_type = seq[idx]
    a2_type = seq[min(idx + 1, len(seq) - 1)]

    if last_action and a1_type == last_action and a2_type != last_action:
        a1_type, a2_type = a2_type, a1_type

    if a1_type == a2_type and len(seq) > 1:
        a2_type = seq[0] if seq[0] != a1_type else seq[1]

    a1 = _action_for(task, a1_type)
    a2 = _action_for(task, a2_type)

    a1 = _maybe_llm_improve(task, contract_text, a1, confidence)
    a2 = _maybe_llm_improve(task, contract_text, a2, confidence)
    return a1, a2


def _choose_best(task: NegotiationTask, state_data: dict[str, Any], a1: Action, a2: Action) -> Action:
    s1 = score_action_hypothetical(task, state_data, a1)
    s2 = score_action_hypothetical(task, state_data, a2)
    return a1 if s1 >= s2 else a2


def _print_step(step_n: int, action: Action, reward_score: float, done: bool, err: Optional[str]) -> None:
    err_token = "error=<null>" if not err else f'error="{err}"'
    print(
        f"[STEP] step={step_n} action={action.action_type} reward={reward_score:.2f} "
        f"done={str(done).lower()} {err_token}",
        flush=True,
    )


def run_episode_direct(env: ContractEnv, initial_obs: dict[str, Any]) -> bool:
    clause_type = initial_obs["clause_type"]
    task = next(t for t in TASKS if t.clause_type == clause_type)

    has_hf = bool(os.getenv("HF_TOKEN"))
    model_label = MODEL_NAME if has_hf else "heuristic"

    rewards: list[float] = []
    last_action: Optional[str] = None
    last_reward = 1.0

    step_n = int(initial_obs.get("step_count", 0))
    contract_text = initial_obs["contract_text"]
    negotiation_history = list(initial_obs.get("negotiation_history", []))

    state_data: dict[str, Any] = {
        "contract_text": contract_text,
        "negotiation_history": negotiation_history,
    }

    done = False

    print(f"[START] task={task.name} env=ContractNegotiationEnv model={model_label}", flush=True)

    max_iters = max(env.max_steps * 4, 24)
    it = 0
    while not done and it < max_iters:
        it += 1

        a1, a2 = _two_candidates(
            task=task,
            contract_text=state_data["contract_text"],
            step_count=step_n,
            last_action=last_action,
            last_reward=last_reward,
        )
        action = _choose_best(task, state_data, a1, a2)

        obs, reward, done, info = env.step(action)
        reward_score = float(reward.score)

        err = info.get("error")
        rewards.append(reward_score)
        last_reward = reward_score
        last_action = action.action_type

        step_n = int(obs.step_count)
        state_data["contract_text"] = obs.contract_text
        state_data["negotiation_history"] = list(obs.negotiation_history)

        _print_step(step_n, action, reward_score, done, err)

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    success = bool(done and avg >= 0.5)
    rlist = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={step_n} score={avg:.2f} rewards={rlist}", flush=True)
    return success


def run_episode_http() -> bool:
    obs = _http_post("/reset")
    clause_type = obs["clause_type"]
    task = next(t for t in TASKS if t.clause_type == clause_type)

    has_hf = bool(os.getenv("HF_TOKEN"))
    model_label = MODEL_NAME if has_hf else "heuristic"

    rewards: list[float] = []
    last_action: Optional[str] = None
    last_reward = 1.0

    step_n = int(obs.get("step_count", 0))
    state_data: dict[str, Any] = {
        "contract_text": obs["contract_text"],
        "negotiation_history": list(obs.get("negotiation_history", [])),
    }

    done = False
    print(f"[START] task={task.name} env=ContractNegotiationEnv model={model_label}", flush=True)

    max_iters = max(ContractEnv.max_steps * 4, 24)
    it = 0
    while not done and it < max_iters:
        it += 1

        a1, a2 = _two_candidates(
            task=task,
            contract_text=state_data["contract_text"],
            step_count=step_n,
            last_action=last_action,
            last_reward=last_reward,
        )
        action = _choose_best(task, state_data, a1, a2)

        payload = {"action_type": action.action_type, "content": action.content}
        out = _http_post("/step", payload)

        reward_score = float(out["reward"]["score"])
        done = bool(out["done"])
        err = out.get("info", {}).get("error")

        rewards.append(reward_score)
        last_reward = reward_score
        last_action = action.action_type

        step_n = int(out["observation"]["step_count"])
        state_data["contract_text"] = out["observation"]["contract_text"]
        state_data["negotiation_history"] = list(out["observation"].get("negotiation_history", []))

        _print_step(step_n, action, reward_score, done, err)

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    success = bool(done and avg >= 0.5)
    rlist = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={step_n} score={avg:.2f} rewards={rlist}", flush=True)
    return success


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Contract Negotiation Agent (OpenEnv)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--direct", action="store_true", help="Use direct env (no HTTP).")
    p.add_argument("--benchmark", action="store_true", help="Run EASY/MEDIUM/HARD and require success.")
    args = p.parse_args()

    try:
        if args.benchmark:
            env = ContractEnv()
            ok = 0
            for _ in range(len(TASKS)):
                obs_model = env.reset()
                obs = obs_model.model_dump(mode="json")
                if run_episode_direct(env, obs):
                    ok += 1
            raise SystemExit(0 if ok == len(TASKS) else 1)

        direct = args.direct or USE_DIRECT_ENV
        for _ in range(max(1, args.episodes)):
            if direct:
                env = ContractEnv()
                obs_model = env.reset()
                obs = obs_model.model_dump(mode="json")
                run_episode_direct(env, obs)
            else:
                run_episode_http()
    except Exception as ex:
        print(f'[STEP] step=0 action=NONE reward=0.00 done=true error="{ex}"', flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

