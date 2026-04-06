from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import ContractEnv
from env.graders import _weighted_risk_hits, score_action_hypothetical, trap_unresolved
from env.models import Action, Observation
from env.tasks import TASKS, NegotiationTask

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:7860").rstrip("/")
USE_DIRECT_ENV = os.getenv("USE_DIRECT_ENV", "").lower() in ("1", "true", "yes")
API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def _obs_dict(obs: dict[str, Any] | Observation) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    return obs.model_dump(mode="json")


def _risk_tier(task: NegotiationTask, contract: str) -> str:
    rs = min(
        1.0,
        _weighted_risk_hits(contract, task.risk_keywords) * task.clause_type_weight / 1.15,
    )
    if task.name == "HARD" and trap_unresolved(task, contract):
        rs = min(1.0, rs + 0.3)
    rs = min(1.0, rs + _trap_boost(contract))
    if rs >= 0.55:
        return "HIGH"
    if rs >= 0.28:
        return "MODERATE"
    return "LOW"


def _risk_score(task: NegotiationTask, contract: str) -> float:
    tier = _risk_tier(task, contract)
    return {"HIGH": 0.85, "MODERATE": 0.5, "LOW": 0.15}[tier]


def _branch_sequence(tier: str) -> list[str]:
    if tier == "HIGH":
        return ["FLAG_RISK", "REJECT", "PROPOSE_COUNTER", "EDIT_CLAUSE"]
    if tier == "MODERATE":
        return ["FLAG_RISK", "EDIT_CLAUSE", "PROPOSE_COUNTER", "REJECT"]
    return ["FLAG_RISK", "ACCEPT", "EDIT_CLAUSE"]


def _pick_branch_index(
    seq: list[str], last_action: Optional[str], low_reward_rotate: bool
) -> int:
    if not last_action:
        return 0
    try:
        i = seq.index(last_action)
    except ValueError:
        return 0
    if low_reward_rotate:
        return (i + 1) % len(seq)
    return (i + 1) % len(seq) if i == 0 else i


_MISLEADING_SAFE = (
    "industry standard",
    "boilerplate",
    "no additional obligations",
    "mutually agreed fair",
)


def _trap_boost(contract: str) -> float:
    low = contract.lower()
    return sum(0.15 for p in _MISLEADING_SAFE if p in low)


def _default_content(task: NegotiationTask, action_type: str) -> str:
    if action_type == "FLAG_RISK":
        return (
            f"Flag {task.clause_type} risks: weighted exposure noted; request redline per "
            f"commercial standards."
        )
    if action_type == "REJECT":
        return "Reject clause as written; terms are commercially unreasonable."
    if action_type == "PROPOSE_COUNTER":
        return task.expected_safe_edit[:400]
    if action_type == "EDIT_CLAUSE":
        return task.expected_safe_edit
    return ""


def _confidence(task: NegotiationTask, contract: str) -> float:
    return max(0.0, min(1.0, 1.0 - _risk_score(task, contract)))


def _maybe_llm_content(task: NegotiationTask, contract: str, action: Action) -> Action:
    need_llm = (
        action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER")
        or _confidence(task, contract) < 0.6
    )
    if not need_llm:
        return action
    key = os.getenv("HF_TOKEN")
    if not key:
        return action
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=key,
        )
        user = (
            f"Task: {task.name} ({task.clause_type}).\n"
            f"Contract excerpt:\n{contract}\n\n"
            f"Action: {action.action_type}.\n"
            "Output only improved contract or negotiation text. Do not output action labels."
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You improve contract clauses and negotiation messages. Output only the text.",
                },
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return Action(action_type=action.action_type, content=text)
    except Exception:
        pass
    return action


def _build_action(
    task: NegotiationTask,
    contract: str,
    seq: list[str],
    branch_idx: int,
    alt_offset: int,
) -> Action:
    t = seq[(branch_idx + alt_offset) % len(seq)]
    base = Action(
        action_type=t, content=_default_content(task, t) if t != "ACCEPT" else None
    )
    return _maybe_llm_content(task, contract, base)


def _two_candidates(
    task: NegotiationTask,
    state: dict[str, Any],
    last_action: Optional[str],
    low_reward: bool,
) -> tuple[Action, Action]:
    contract = state.get("contract_text", "")
    tier = _risk_tier(task, contract)
    seq = _branch_sequence(tier)
    bi = _pick_branch_index(seq, last_action, low_reward)
    a1 = _build_action(task, contract, seq, bi, 0)
    a2 = _build_action(task, contract, seq, bi, 1)
    if a1.model_dump() == a2.model_dump():
        a2 = _build_action(task, contract, seq, bi, 2)
    return a1, a2


def _choose_best(
    task: NegotiationTask,
    state: dict[str, Any],
    a1: Action,
    a2: Action,
    last_action: Optional[str],
) -> Action:
    s1 = score_action_hypothetical(task, state, a1)
    s2 = score_action_hypothetical(task, state, a2)
    first, second = (a1, a2) if s1 >= s2 else (a2, a1)
    if last_action and first.action_type == last_action and second.action_type != last_action:
        return second
    if last_action and first.action_type == last_action == second.action_type:
        return second
    return first


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


def run_episode() -> None:
    load_dotenv()
    has_hf = bool(os.getenv("HF_TOKEN"))
    model_label = MODEL_NAME if has_hf else "heuristic"
    env: Optional[ContractEnv] = None

    if USE_DIRECT_ENV:
        env = ContractEnv()
        obs = env.reset()
    else:
        obs = _http_post("/reset")

    od = _obs_dict(obs)
    task = next(t for t in TASKS if t.id == od["task_id"])

    rewards: list[float] = []
    last_action: Optional[str] = None
    last_reward = 1.0
    step_n = 0

    print(
        f"[START] task={task.name} env=ContractNegotiationEnv model={model_label}",
        flush=True,
    )

    done = bool(od.get("done", False))
    state: dict[str, Any] = {
        "contract_text": od["contract_text"],
        "negotiation_history": list(od.get("negotiation_history", [])),
    }

    max_iters = max(ContractEnv.max_steps * 4, 24)
    it = 0
    prev_action: Optional[str] = None
    while not done and it < max_iters:
        it += 1
        low_reward = last_reward < 0.5
        a1, a2 = _two_candidates(task, state, last_action, low_reward)
        action = _choose_best(task, state, a1, a2, last_action)
        if (
            prev_action
            and action.action_type == prev_action
            and action.action_type not in ("ACCEPT", "REJECT")
        ):
            action = a2 if a1.model_dump() == action.model_dump() else a1
        prev_action = action.action_type

        err_display = "error=<null>"
        if USE_DIRECT_ENV:
            assert env is not None
            o, r, done, info = env.step(action)
            last_reward = r.score
            rewards.append(r.score)
            state["contract_text"] = o.contract_text
            state["negotiation_history"] = list(o.negotiation_history)
            step_n = o.step_count
            if info.get("error"):
                err_display = f'error="{info["error"]}"'
            print(
                f"[STEP] step={step_n} action={action.action_type} reward={r.score:.2f} "
                f"done={str(done).lower()} {err_display}",
                flush=True,
            )
            last_action = action.action_type
        else:
            payload = {"action_type": action.action_type, "content": action.content}
            out = _http_post("/step", payload)
            last_reward = float(out["reward"]["score"])
            rewards.append(last_reward)
            obs = out["observation"]
            done = bool(out["done"])
            state["contract_text"] = obs["contract_text"]
            state["negotiation_history"] = obs["negotiation_history"]
            step_n = obs["step_count"]
            e = out.get("info", {}).get("error")
            if e:
                err_display = f'error="{e}"'
            print(
                f"[STEP] step={step_n} action={action.action_type} reward={last_reward:.2f} "
                f"done={str(done).lower()} {err_display}",
                flush=True,
            )
            last_action = action.action_type

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    success = bool(rewards and avg >= 0.5 and done)
    rlist = ",".join(f"{x:.2f}" for x in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_n} score={avg:.2f} rewards={rlist}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        run_episode()
    except Exception as ex:
        print(
            f"[STEP] step=0 action=NONE reward=0.00 done=true error=\"{ex}\"",
            flush=True,
        )
        print(
            "[END] success=false steps=0 score=0.00 rewards=",
            flush=True,
        )
        sys.exit(1)
