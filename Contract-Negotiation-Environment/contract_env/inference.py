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
    _weighted_risk_hits,
    contract_quality_score,
    score_action_hypothetical,
    trap_unresolved,
)
from env.models import Action, Observation
from env.tasks import TASKS, NegotiationTask

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:7860").rstrip("/")
USE_DIRECT_ENV = os.getenv("USE_DIRECT_ENV", "").lower() in ("1", "true", "yes")
API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

_ACTION_EVAL_ORDER = [
    "EDIT_CLAUSE",
    "PROPOSE_COUNTER",
    "FLAG_RISK",
    "REJECT",
    "ACCEPT",
]


def _obs_dict(obs: dict[str, Any] | Observation) -> dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    return obs.model_dump(mode="json")


def _default_content(task: NegotiationTask, action_type: str) -> Optional[str]:
    if action_type == "FLAG_RISK":
        return (
            f"Flag material {task.clause_type} risks; request cap, carve-outs, and "
            f"mutual escalation per {task.industry_context} practice."
        )
    if action_type == "REJECT":
        return (
            "Reject as drafted: terms are one-sided and create uncapped or "
            "non-volitional renewal exposure."
        )
    if action_type == "PROPOSE_COUNTER":
        return task.expected_safe_edit[:500]
    if action_type == "EDIT_CLAUSE":
        return task.expected_safe_edit
    if action_type == "ACCEPT":
        return None
    return None


def _action_for_type(task: NegotiationTask, action_type: str) -> Action:
    return Action(
        action_type=action_type,
        content=_default_content(task, action_type),
    )


def _confidence(task: NegotiationTask, contract: str) -> float:
    rs = min(
        1.0,
        _weighted_risk_hits(contract, task.risk_keywords) * task.clause_type_weight / 1.15,
    )
    if task.name == "HARD" and trap_unresolved(task, contract):
        rs = min(1.0, rs + 0.25)
    return max(0.0, min(1.0, 1.0 - rs))


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
            f"Task: {task.name} ({task.clause_type}, {task.industry_context}).\n"
            f"Contract excerpt:\n{contract}\n\n"
            f"Action: {action.action_type}.\n"
            "Output only improved contract or negotiation text. No action labels or preamble."
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior commercial counsel. Output only the clause text "
                        "or negotiation line requested."
                    ),
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


def _rank_heuristic_actions(
    task: NegotiationTask,
    state: dict[str, Any],
    last_action: Optional[str],
) -> list[tuple[float, Action]]:
    scored: list[tuple[float, str, Action]] = []
    for at in _ACTION_EVAL_ORDER:
        base = _action_for_type(task, at)
        s = score_action_hypothetical(task, state, base)
        scored.append((s, at, base))
    scored.sort(key=lambda x: (-x[0], _ACTION_EVAL_ORDER.index(x[1])))
    out: list[tuple[float, Action]] = []
    for s, _, a in scored:
        out.append((s, a))
    if (
        last_action
        and out[0][1].action_type == last_action
        and len(out) > 1
        and out[0][0] - out[1][0] <= 0.12
    ):
        i = 1
        while i < len(out) and out[i][1].action_type == last_action:
            i += 1
        if i < len(out):
            top = out[0]
            out[0] = out[i]
            out[i] = top
    return [(s, a) for s, a in out]


def select_action(
    task: NegotiationTask,
    state: dict[str, Any],
    last_action: Optional[str],
) -> Action:
    _, action = _rank_heuristic_actions(task, state, last_action)[0]
    return _maybe_llm_content(task, state.get("contract_text", ""), action)


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


def run_episode(
    env: Optional[ContractEnv] = None,
    http_mode: Optional[bool] = None,
    pre_reset_obs: Optional[dict[str, Any] | Observation] = None,
) -> tuple[bool, int, float, list[float], NegotiationTask, float]:
    use_http = http_mode if http_mode is not None else not USE_DIRECT_ENV
    has_hf = bool(os.getenv("HF_TOKEN"))
    model_label = MODEL_NAME if has_hf else "heuristic"

    local_env = env
    if pre_reset_obs is not None:
        obs = pre_reset_obs
        if not use_http and local_env is None:
            raise ValueError("pre_reset_obs requires env when not using http_mode")
    elif use_http:
        obs = _http_post("/reset")
    else:
        if local_env is None:
            local_env = ContractEnv()
        obs = local_env.reset()

    od = _obs_dict(obs)
    task = next(t for t in TASKS if t.id == od["task_id"])

    rewards: list[float] = []
    last_action: Optional[str] = None
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
    while not done and it < max_iters:
        it += 1
        action = select_action(task, state, last_action)
        err_display = "error=<null>"

        if use_http:
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
        else:
            assert local_env is not None
            o, r, done, info = local_env.step(action)
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

    avg = sum(rewards) / len(rewards) if rewards else 0.0
    quality = contract_quality_score(task, state["contract_text"])
    success = bool(rewards and avg >= 0.5 and done)
    rlist = ",".join(f"{x:.2f}" for x in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_n} score={avg:.2f} "
        f"quality={quality:.2f} rewards={rlist}",
        flush=True,
    )
    return success, step_n, avg, rewards, task, quality


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Hybrid contract negotiation agent")
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes (HTTP server cycles tasks via /reset).",
    )
    p.add_argument(
        "--direct",
        action="store_true",
        help="Use ContractEnv in-process (ignores BASE_URL).",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run exactly three in-process episodes (EASY/MEDIUM/HARD) from fresh envs.",
    )
    args = p.parse_args()

    if args.benchmark:
        print("[BENCHMARK] three-task sweep (direct env)", flush=True)
        ok = 0
        bench_env = ContractEnv()
        for _i in range(len(TASKS)):
            o = bench_env.reset()
            succ, _, avg, _, task, q = run_episode(
                env=bench_env, http_mode=False, pre_reset_obs=o
            )
            print(
                f"  {task.name}: success={succ} mean_reward={avg:.2f} quality={q:.2f}",
                flush=True,
            )
            if succ:
                ok += 1
        print(f"[BENCHMARK] passed {ok}/{len(TASKS)} success criterion", flush=True)
        raise SystemExit(0 if ok == len(TASKS) else 1)

    http_mode = not args.direct
    wins = 0
    for _ in range(max(1, args.episodes)):
        s, _, _, _, _, _ = run_episode(http_mode=http_mode)
        if s:
            wins += 1
    if args.episodes > 1:
        print(f"[SUMMARY] episodes={args.episodes} wins={wins}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as ex:
        print(
            f"[STEP] step=0 action=NONE reward=0.00 done=true error=\"{ex}\"",
            flush=True,
        )
        print(
            "[END] success=false steps=0 score=0.00 quality=0.00 rewards=",
            flush=True,
        )
        sys.exit(1)
