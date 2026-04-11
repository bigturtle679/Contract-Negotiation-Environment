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
from contract_env.env.graders import TASK_GRADERS, NUM_GRADED_TASKS
from contract_env.env.models import Action, StepRequest
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