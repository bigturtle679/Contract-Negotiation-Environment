from __future__ import annotations

import random
from typing import Any, Optional, Tuple

from env.graders import (
    build_proposed_contract_for_step,
    evaluate_action,
    observation_risk_float,
)
from env.models import Action, Observation, Reward
from env.tasks import TASKS, NegotiationTask

random.seed(42)


class ContractEnv:
    max_steps: int = 5

    def __init__(self) -> None:
        self._reset_count: int = 0
        self.current_task: Optional[NegotiationTask] = None
        self.current_step: int = 0
        self.done: bool = False
        self.state_data: dict[str, Any] = {}

    def reset(self) -> Observation:
        self.done = False
        self.current_step = 0
        idx = self._reset_count % len(TASKS)
        self._reset_count += 1
        self.current_task = TASKS[idx]
        assert self.current_task is not None
        t = self.current_task
        history: list[str] = []
        for line in t.opponent_opening:
            history.append(f"opponent|{line}")
        self.state_data = {
            "task_id": t.id,
            "task_name": t.name,
            "contract_text": t.contract_text,
            "clause_type": t.clause_type,
            "negotiation_history": history,
        }
        return self._make_observation()

    def _make_observation(self) -> Observation:
        assert self.current_task is not None
        ct = self.state_data["contract_text"]
        t = self.current_task
        return Observation(
            contract_text=ct,
            clause_type=t.clause_type,
            risk_level=observation_risk_float(t, ct),
            step_count=self.current_step,
            negotiation_history=list(self.state_data["negotiation_history"]),
            task_id=t.id,
            task_name=t.name,
            done=self.done,
            max_steps=self.max_steps,
            industry_context=t.industry_context,
        )

    def _validate_action(self, action: Action) -> Optional[str]:
        c = (action.content or "").strip()
        if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
            if not c:
                return "EDIT_CLAUSE and PROPOSE_COUNTER require non-empty content"
        return None

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict[str, Any]]:
        assert self.current_task is not None
        info: dict[str, Any] = {}
        if self.done:
            r = Reward(score=0.0, feedback="episode_done", metrics=None)
            return self._make_observation(), r, True, {"error": "already_done"}

        err = self._validate_action(action)
        if err:
            self.current_step += 1
            r = Reward(score=0.0, feedback=f"invalid_action: {err}", metrics=None)
            if self.current_step >= self.max_steps:
                self.done = True
            info["error"] = err
            return self._make_observation(), r, self.done, info

        contract_before = self.state_data["contract_text"]
        proposed = build_proposed_contract_for_step(contract_before, action)
        reward, grade_info = evaluate_action(
            self.current_task, contract_before, action, proposed
        )
        info.update(grade_info)

        entry = (
            f"agent|step={self.current_step + 1} action={action.action_type} "
            f"content_len={len((action.content or '').strip())}"
        )
        self.state_data["negotiation_history"].append(entry)

        if action.action_type == "EDIT_CLAUSE":
            self.state_data["contract_text"] = (action.content or "").strip()
        elif action.action_type == "PROPOSE_COUNTER":
            self.state_data["contract_text"] = proposed

        self.current_step += 1
        if action.action_type == "ACCEPT" or self.current_step >= self.max_steps:
            self.done = True

        return self._make_observation(), reward, self.done, info

    def state(self) -> dict[str, Any]:
        out = dict(self.state_data)
        if self.current_task is not None:
            out["task"] = self.current_task.model_dump()
        out["current_step"] = self.current_step
        out["done"] = self.done
        out["max_steps"] = self.max_steps
        return out

    def close(self) -> None:
        return None
