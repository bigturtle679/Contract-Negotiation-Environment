from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ActionType = Literal["FLAG", "REJECT", "EDIT_CLAUSE", "PROPOSE_COUNTER", "ACCEPT"]
RiskLevel = Literal["LOW", "MODERATE", "HIGH"]


class Action(BaseModel):
    action_type: ActionType
    content: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str


class Observation(BaseModel):
    task_id: str
    task_name: str
    contract_text: str
    clause_type: str
    risk_level: RiskLevel
    negotiation_history: list[str]
    step_count: int
    max_steps: int
    done: bool


class StepRequest(BaseModel):
    action_type: ActionType
    content: Optional[str] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict
