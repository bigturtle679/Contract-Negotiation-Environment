# Contract Negotiation Environment (OpenEnv)

Deterministic RL-style environment for multi-step contract negotiation: flag risky clauses, edit language, propose counters, accept or reject. Rewards are graded in `[0.0, 1.0]` with explicit trade-offs (correctness, improvement, risk alignment).

## Setup

- **Python 3.10** (matches `Dockerfile`).
- From `contract_env`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=%CD%
```

## Local server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

## Inference (OpenAI client → Hugging Face router)

```bash
set PYTHONPATH=%CD%
set BASE_URL=http://127.0.0.1:7860
set HF_TOKEN=your_hf_token
python inference.py
```

Heuristic-only (no `HF_TOKEN`):

```bash
python inference.py
```

Direct env (no HTTP):

```bash
set USE_DIRECT_ENV=1
python inference.py
```

The client uses:

```python
OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.getenv("HF_TOKEN"))
```

Default model: `Qwen/Qwen2.5-72B-Instruct` (override with `MODEL_NAME`).

LLM is used when confidence is below 0.6 or when the chosen action is `EDIT_CLAUSE` / `PROPOSE_COUNTER`. On API failure, heuristics apply.

## Docker

```bash
cd contract_env
docker build -t contract-negotiation-env .
docker run --rm -p 7860:7860 contract-negotiation-env
```

`POST /reset` must return HTTP 200 (empty JSON body is fine).

## Hugging Face Space

1. Create a **Docker** Space from this folder.
2. Expose port **7860** (see `Dockerfile` `CMD`).
3. Add secret **`HF_TOKEN`** for the HF router if agents should call the LLM from your host scripts; the Space server itself does not require it to serve `/reset` and `/step`.

## Architecture

| Component | Role |
|-----------|------|
| `env/tasks.py` | Three tasks: EASY (obvious liability), MEDIUM (auto-renewal), HARD (trade-offs + hidden trap markers). |
| `env/graders.py` | `score = 0.4*correctness + 0.3*improvement + 0.3*risk_alignment`; keyword + token-overlap; **`ACCEPT` with effective high risk ⇒ score 0**. |
| `env/environment.py` | `ContractEnv`: `random.seed(42)`, task cycle on reset, `max_steps=5`, reward computed before contract mutation from prior text + proposed text. |
| `server.py` | FastAPI: `POST /reset`, `POST /step`; JSON-serializable responses. |
| `inference.py` | Hybrid agent: weighted risk, clause-type boost, trap-phrase boost, branching policy, duplicate-action avoidance, two candidate actions graded hypothetically, optional Qwen via router. |

## Action space

`FLAG_RISK`, `EDIT_CLAUSE`, `ACCEPT`, `REJECT`, `PROPOSE_COUNTER`.  
`EDIT_CLAUSE` and `PROPOSE_COUNTER` require non-empty `content` (after strip).

## Reward system

- **Correctness**: weighted `risk_keywords` vs evaluation text + token overlap vs prior contract.
- **Improvement**: `safe_keywords` and overlap with `expected_safe_edit`.
- **Risk alignment**: action appropriateness vs effective high-risk state (including HARD trap markers).
- Final score clipped to `[0, 1]`; wrong accept under high effective risk forces **0**.

## OpenEnv

Manifest: `openenv.yaml` (`name: ContractNegotiationEnv`, `entry_point: env.environment:ContractEnv`).

```bash
openenv validate .
openenv validate --url http://127.0.0.1:7860
```

## Pre-validation

- Windows: `powershell -ExecutionPolicy Bypass -File scripts/prevalidate.ps1`

## Inference log format

```text
[START] task=<task_name> env=ContractNegotiationEnv model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

When there is no error, the line uses the literal token `error=<null>`.
