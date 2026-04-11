from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ActionType = Literal[
    "FLAG_RISK",
    "EDIT_CLAUSE",
    "ACCEPT",
    "REJECT",
    "PROPOSE_COUNTER",
]


class Action(BaseModel):
    """Agent action: one of the five negotiation moves with optional clause content."""

    action_type: ActionType
    content: Optional[str] = None


class Reward(BaseModel):
    """Step reward with a score strictly between 0 and 1 (exclusive)."""

    score: float = Field(gt=0.0, lt=1.0)

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"score must be strictly between 0 and 1, got {v}")
        return v


class Observation(BaseModel):
    """Environment observation returned after each reset/step."""

    contract_text: str
    clause_type: str
    risk_level: float = Field(gt=0.0, lt=1.0)
    step_count: int
    negotiation_history: list[str]

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"risk_level must be strictly between 0 and 1, got {v}")
        return v


class StepRequest(BaseModel):
    """Request body for the ``/step`` endpoint."""

    action_type: ActionType
    content: Optional[str] = None