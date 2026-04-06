# Contract Negotiation Environment (OpenEnv-ready)

Deterministic, multi-step contract negotiation environment for RL-style agents. The agent flags risky language, edits clauses, proposes counters, and decides whether to accept or reject, with graded rewards in `[0, 1]`.

## Requirements

- **Python 3.10** matches `Dockerfile` and `pydantic==2.6.0` wheels. Newer interpreters may need a newer Pydantic build (or use Docker).
- Dependencies in `requirements.txt`.

## Local setup

Use a **virtual environment** so a third-party PyPI package named `env` (if installed) does not shadow this project’s `env` package.

```bash
cd contract_env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=%CD%
uvicorn server:app --host 0.0.0.0 --port 7860
```

In another shell:

```bash
set PYTHONPATH=%CD%
set BASE_URL=http://127.0.0.1:7860
set HF_TOKEN=your_hf_token
python inference.py
```

Offline / no server:

```bash
set USE_DIRECT_ENV=1
python inference.py
```

## Docker

```bash
cd contract_env
docker build -t contract-negotiation-env .
docker run --rm -p 7860:7860 contract-negotiation-env
```

Health check:

```bash
curl http://127.0.0.1:7860/health
```

Reset episode:

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d "{}"
```

## Hugging Face Space

1. Create a **Docker** Space using this `Dockerfile`.
2. Expose port **7860** (matches `CMD`).
3. Add secret **`HF_TOKEN`** for `https://router.huggingface.co/v1` (OpenAI-compatible client; no OpenAI billing).

## API

- `GET /health` — `{ "status": "ok" }`
- `POST /reset` — returns full `Observation` JSON (`model_dump`)
- `POST /step` — body `{ "action_type": "...", "content": "..." }`  
  Returns `{ observation, reward, done, info }`

**Actions:** `FLAG`, `REJECT`, `EDIT_CLAUSE`, `PROPOSE_COUNTER`, `ACCEPT`.  
`EDIT_CLAUSE` and `PROPOSE_COUNTER` require non-empty `content` (after strip).

## Episode semantics

After each successful `step`, `observation.step_count` is the number of steps taken so far. The episode ends on `ACCEPT` or when `step_count >= max_steps` (5).

## Architecture

- **`env/tasks.py`** — three fixed tasks (**EASY** / **MEDIUM** / **HARD**) with weighted `risk_keywords`, `safe_keywords`, `expected_safe_edit`, `risk_level`, and HARD `hidden_trap` / `trap_markers`.
- **`env/graders.py`** — `0.4 * correctness + 0.3 * improvement + 0.3 * risk_alignment`; **score 0** on `ACCEPT` when effective risk is high (trap rules + keyword materiality).
- **`env/environment.py`** — `ContractEnv`: `reset_count % 3` task cycling, `max_steps = 5`, reward **before** state mutation, then history + contract updates.
- **`server/app.py`** — FastAPI on shared `ContractEnv`.
- **`inference.py`** — heuristic + optional Qwen via HF router; two candidates scored with `score_action_hypothetical`.

## OpenEnv manifest

See `openenv.yaml` (`entry_point: env.environment:ContractEnv`). With the CLI installed:

```bash
cd contract_env
openenv validate .
openenv validate --url http://127.0.0.1:7860
```

`pyproject.toml` includes `openenv` from Git for editable/validate workflows; Docker + `requirements.txt` remain the minimal runtime path.

## Pre-validation

- Windows: `powershell -ExecutionPolicy Bypass -File scripts/prevalidate.ps1`
- Unix: `bash scripts/prevalidate.sh`

## Inference log format

```text
[START] task=<task_name> env=ContractNegotiationEnv model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

When there is no error, the line uses the literal token `error=<null>`.
