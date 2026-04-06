from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from env.environment import ContractEnv
from env.models import Action, StepRequest, StepResponse

_env = ContractEnv()

app = FastAPI(title="Contract Negotiation OpenEnv", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


def run() -> None:
    main()


if __name__ == "__main__":
    main()
