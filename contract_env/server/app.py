from __future__ import annotations

import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from contract_env.env.environment import ContractEnv
from contract_env.env.graders import TASK_GRADERS, NUM_GRADED_TASKS, contract_quality_score
from contract_env.env.models import Action, Observation, Reward, StepRequest
from contract_env.env.tasks import TASKS

_env = ContractEnv()

app = FastAPI(
    title="Contract Negotiation OpenEnv",
    description=(
        "AI-driven environment for evaluating contract-negotiation agents. "
        "Agents analyse clauses, identify risks, and propose safer alternatives."
    ),
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── ROOT ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "contract-negotiation-env"}


# ── ERROR HANDLERS ──────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "trace": traceback.format_exc(),
        },
    )


# ── HEALTH ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── TASKS LISTING ───────────────────────────────────────────────────────
@app.get("/tasks")
def list_tasks():
    """Return metadata for every registered task."""
    return {
        "total": len(TASKS),
        "graded": NUM_GRADED_TASKS,
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "clause_type": t.clause_type,
                "risk_level": t.risk_level,
                "industry_context": t.industry_context,
                "has_grader": t.id in TASK_GRADERS,
            }
            for t in TASKS
        ],
    }


# ── STATE ───────────────────────────────────────────────────────────────
@app.get("/state")
def get_state():
    return _env.state()


# ── RESET ───────────────────────────────────────────────────────────────
@app.post("/reset")
def reset():
    try:
        obs = _env.reset()
        return {"observation": obs.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── STEP ────────────────────────────────────────────────────────────────
@app.post("/step")
def step(req: StepRequest):
    try:
        action = Action(action_type=req.action_type, content=req.content)
        obs, reward, done, info = _env.step(action)

        return {
            "observation": obs.model_dump(),
            "reward": {"score": reward},
            "done": done,
            "info": info,
        }

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


# ── SCHEMA ──────────────────────────────────────────────────────────────
@app.get("/schema")
def get_schema():
    """Return JSON Schema for Action, Observation, and Reward models."""
    return {
        "Action": Action.model_json_schema(),
        "Observation": Observation.model_json_schema(),
        "Reward": Reward.model_json_schema(),
    }


# ── EVALUATE QUALITY ─────────────────────────────────────────────────────
@app.post("/evaluate-quality")
def evaluate_quality(body: dict):
    """Score an arbitrary contract text against the current task.

    Body: {"contract_text": "..."}
    Returns: {"quality_score": float, "risk_score": float}
    where quality_score ∈ (0, 1), 1 = fully safe.
    """
    if _env.current_task is None:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")
    contract_text = body.get("contract_text", "")
    if not contract_text:
        raise HTTPException(status_code=422, detail="contract_text must be non-empty.")
    quality = contract_quality_score(_env.current_task, contract_text)
    return {"quality_score": round(quality, 4), "risk_score": round(1.0 - quality, 4)}


# ── MAIN ────────────────────────────────────────────────────────────────
def main():
    import uvicorn

    uvicorn.run(
        "contract_env.server.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
    )


if __name__ == "__main__":
    main()