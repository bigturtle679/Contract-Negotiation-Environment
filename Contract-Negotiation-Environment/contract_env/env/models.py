from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ActionType = Literal[
    "FLAG_RISK", "EDIT_CLAUSE", "ACCEPT", "REJECT", "PROPOSE_COUNTER"
]


class Action(BaseModel):
    action_type: ActionType
    content: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str


class Observation(BaseModel):
    contract_text: str
    clause_type: str
    risk_level: float = Field(ge=0.0, le=1.0)
    step_count: int
    negotiation_history: list[str]
    task_id: str
    task_name: str


class StepRequest(BaseModel):
    action_type: ActionType
    content: Optional[str] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict
