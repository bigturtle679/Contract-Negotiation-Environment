from __future__ import annotations

import random
from typing import Any, Optional, Tuple

from contract_env.env.graders import (
    build_proposed_contract_for_step,
    evaluate_action,
    observation_risk_float,
)
from contract_env.env.models import Action, Observation
from contract_env.env.tasks import TASKS, NegotiationTask

random.seed(42)


class ContractEnv:
    max_steps: int = 7

    def __init__(self) -> None:
        self._reset_count: int = 0
        self.current_task: Optional[NegotiationTask] = None
        self.current_step: int = 0
        self.done: bool = False
        self.state_data: dict[str, Any] = {}
        self._rng = random.Random(42)

    @property
    def tasks(self) -> list[str]:
        from contract_env.env.tasks import TASKS
        return [task.id for task in TASKS]

    @property
    def graders(self) -> dict:
        from contract_env.env.graders import TASK_GRADERS
        return TASK_GRADERS

    def _opponent_reply(self, action_type: str) -> Optional[str]:
        """Generate an opponent response based on the action taken.

        If the current task defines opponent_responses for this action_type,
        pick one at random. Otherwise return None.
        """
        if self.current_task is None:
            return None
        responses = self.current_task.opponent_responses.get(action_type, [])
        if not responses:
            return None
        return self._rng.choice(responses)

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

        return Observation(
            contract_text=ct,
            clause_type=self.current_task.clause_type,
            risk_level=observation_risk_float(self.current_task, ct),
            step_count=self.current_step,
            negotiation_history=list(self.state_data["negotiation_history"]),
        )

    def _validate_action(self, action: Action) -> Optional[str]:
        c = (action.content or "").strip()

        if action.action_type in ("EDIT_CLAUSE", "PROPOSE_COUNTER"):
            if not c:
                return "EDIT_CLAUSE and PROPOSE_COUNTER require non-empty content"

        return None

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict[str, Any]]:
        assert self.current_task is not None

        info: dict[str, Any] = {}

        if self.done:
            return self._make_observation(), 0.0, True, {"error": "already_done"}

        err = self._validate_action(action)
        if err:
            self.current_step += 1

            if self.current_step >= self.max_steps:
                self.done = True

            info["error"] = err
            return self._make_observation(), 0.001, self.done, info

        contract_before = self.state_data["contract_text"]
        proposed = build_proposed_contract_for_step(contract_before, action)

        # Use the task-specific grader when available
        task = self.current_task
        if task.has_grader():
            reward_obj = task.grade(contract_before, action, proposed)
            # Collect grade info from evaluate_action for transparency
            _, grade_info = evaluate_action(task, contract_before, action, proposed)
        else:
            reward_obj, grade_info = evaluate_action(
                task, contract_before, action, proposed
            )

        reward = float(reward_obj.score)

        info.update(grade_info)

        entry = (
            f"agent|step={self.current_step + 1} action={action.action_type} "
            f"content_len={len((action.content or '').strip())}"
        )
        self.state_data["negotiation_history"].append(entry)

        # Opponent simulation: add a counterparty reply to the history
        opp_reply = self._opponent_reply(action.action_type)
        if opp_reply:
            self.state_data["negotiation_history"].append(f"opponent|{opp_reply}")
            info["opponent_reply"] = opp_reply

        if action.action_type == "EDIT_CLAUSE":
            self.state_data["contract_text"] = (action.content or "").strip()

        elif action.action_type == "PROPOSE_COUNTER":
            self.state_data["contract_text"] = proposed

        self.current_step += 1

        if action.action_type == "ACCEPT":
            self.done = True
            info["termination_reason"] = "agent_accepted"
        elif self.current_step >= self.max_steps:
            self.done = True
            info["termination_reason"] = "max_steps_reached"

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