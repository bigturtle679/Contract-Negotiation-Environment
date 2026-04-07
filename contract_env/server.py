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
from contract_env.env.models import Action, StepRequest, StepResponse

_env = ContractEnv()

app = FastAPI(
    title="Contract Negotiation OpenEnv",
    version="1.1.0",
    description="Multi-step contract negotiation RL environment with graded rewards.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "path": str(request.url.path)},
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal_error",
            "path": str(request.url.path),
            "error": str(exc),
            "trace": traceback.format_exc() if os.getenv("DEBUG") == "1" else None,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "contract-negotiation-env"}


@app.get("/state")
def get_state() -> dict[str, Any]:
    return _env.state()


@app.post("/reset")
def reset() -> dict[str, object]:
    obs = _env.reset()
    return obs.model_dump(mode="json")


@app.post("/step")
def step(req: StepRequest) -> dict[str, object]:
    try:
        action = Action(action_type=req.action_type, content=req.content)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    obs, reward, done, info = _env.step(action)
    body = StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )
    return body.model_dump(mode="json")


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
