---
title: Contract Negotiation Environment
emoji: 🤝
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Contract Negotiation Environment

An OpenEnv-compliant environment where an AI agent negotiates real-world contract
clauses — identifying legal risks, proposing safer rewrites, and earning rewards
proportional to how well it protects the customer while keeping commercially
reasonable terms.

---

## Why contract negotiation?

Contract review is a high-stakes, cognitively demanding task performed daily by
lawyers, procurement teams, and founders. Key challenges for an AI agent:

- **Hidden traps**: one-sided clauses are often buried in boilerplate language.
- **Judgment under uncertainty**: the agent must decide *when* to flag, edit,
  counter, reject, or accept — each with different risk trade-offs.
- **Partial-progress rewards**: improving a clause partially (e.g., adding a
  liability cap without addressing IP ownership) deserves more reward than doing
  nothing — but less than resolving every risk.

---

## Tasks

| ID | Difficulty | Clause Type | Risk Level | Hidden Trap |
|----|-----------|-------------|-----------|-------------|
| `easy_unlimited_liability` | Easy (1/5) | Liability | HIGH | No |
| `medium_auto_renewal` | Medium (2/5) | Term/Renewal | MODERATE | No |
| `hard_conflicting_obligations` | Hard (4/5) | Performance/Changes | HIGH | Yes |
| `easy_compliance_agreement` | Easy+ (2/5) | Compliance | LOW | No |
| `hard_intellectual_property` | Hard+ (5/5) | IP Ownership | HIGH | Yes |

### Task descriptions

**easy_unlimited_liability** — A vendor clause imposes unlimited indemnity for
all claims without any cap. The correct action is to edit the clause to cap
liability at 12 months of fees paid and exclude punitive/consequential damages.

**medium_auto_renewal** — An auto-renewal clause gives only one calendar day of
cancellation notice. The agent should counter-propose at least 60 days notice
and make auto-renewal opt-in.

**hard_conflicting_obligations** — Two hidden, conflicting obligations: (1)
unlimited uncompensated change orders and (2) a "safeguard" clause that
contradicts the unlimited-changes obligation. Both traps must be resolved to
earn full marks.

**easy_compliance_agreement** — A low-risk compliance clause that needs a minor
improvement: adding explicit breach-notification obligations ("+6% bonus for
'promptly notify Customer'").

**hard_intellectual_property** — Supplier claims ownership of all IP, even when
the customer provides specifications. The agent must rewrite to assign IP to the
customer and limit the supplier to a scoped license.

---

## Observation Space

Every call to `/reset` or `/step` returns an `Observation`:

```json
{
  "contract_text": "string — the current clause text (may be rewritten after EDIT/PROPOSE)",
  "clause_type": "string — e.g. liability, term_renewal, intellectual_property",
  "risk_level": "float ∈ (0, 1) — observed risk density (0=safe, 1=highly risky)",
  "step_count": "int — steps taken so far (0 = just reset)",
  "negotiation_history": [
    "opponent|[Counterparty] Unlimited indemnity is standard.",
    "agent|step=1 action=FLAG_RISK content_len=0",
    "..."
  ]
}
```

`negotiation_history` entries are prefixed with `opponent|` or `agent|`.

---

## Action Space

Discrete, 5 choices:

| `action_type` | `content` required? | When to use |
|--------------|---------------------|-------------|
| `FLAG_RISK` | No | First move on HIGH-risk clauses to signal awareness |
| `EDIT_CLAUSE` | Yes | Directly rewrite the clause with safer language |
| `PROPOSE_COUNTER` | Yes | Submit a formal counter-offer (appended as `[COUNTERPROPOSAL]`) |
| `REJECT` | No | Refuse egregiously one-sided terms |
| `ACCEPT` | No | Accept when all material risks are resolved |

`EDIT_CLAUSE` and `PROPOSE_COUNTER` require non-empty `content`.
Sending empty content returns a validation error and a near-zero reward.

---

## Reward & Scoring

Every step returns a scalar `reward ∈ (0.001, 0.999)`, computed as:

```
reward = 0.40 × correctness
       + 0.30 × improvement
       + 0.30 × risk_alignment
```

| Component | What it measures |
|-----------|-----------------|
| **Correctness** (40%) | For EDIT/PROPOSE: how much risky language was *removed* from the original. For FLAG/REJECT/ACCEPT: how many risk keywords are identified in context. |
| **Improvement** (30%) | How well the proposed edit matches safe keywords and the expected safe rewrite. |
| **Risk Alignment** (30%) | Whether the chosen action is appropriate for the current risk level (e.g., editing a HIGH-risk clause scores 0.92×; accepting it scores 0.20×). |

### Task-specific adjustments

| Task | Adjustment |
|------|-----------|
| Easy | +8% bonus when safe edit matches well |
| Medium | −35% penalty for accepting risky auto-renewal terms |
| Hard | −50% penalty when hidden trap markers remain in the proposed text |
| Easy+ | +6% bonus for including breach-notification language |
| Hard+ | −45% penalty for unresolved IP traps; +7% bonus for explicit customer ownership |

Blocked accepts (accepting HIGH-risk text) are clamped to `0.001`.

### Episode score

The `[END]` line reports `score = mean(rewards over all steps)`.
An episode is considered successful if `score ≥ 0.50`.

---

## Reference Baseline Scores

Measured over 1 episode per task with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Avg reward/step | Episode score |
|------|----------------|---------------|
| `easy_unlimited_liability` | 0.64 | 0.64 |
| `medium_auto_renewal` | 0.58 | 0.58 |
| `hard_conflicting_obligations` | 0.45 | 0.45 |
| `easy_compliance_agreement` | 0.61 | 0.61 |
| `hard_intellectual_property` | 0.42 | 0.42 |

A random agent achieves approximately 0.28 average per step across all tasks.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/schema` | JSON Schema for Action, Observation, Reward models |
| `GET` | `/tasks` | All tasks + graded count |
| `GET` | `/state` | Full internal environment state |
| `POST` | `/reset` | Start a new episode, returns first Observation |
| `POST` | `/step` | Submit `{action_type, content}`, returns `{observation, reward, done, info}` |
| `POST` | `/evaluate-quality` | Score `{contract_text}` against current task without stepping |

---

## Quick Start

### Local development

```bash
pip install -e ".[dev]"
python -m unittest discover contract_env/tests/ -v
```

### Run the server

```bash
uvicorn contract_env.server.app:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
export HF_TOKEN="your-huggingface-token"
python inference.py --benchmark    # one episode per task (5 total)
python inference.py --episodes 3   # run 3 episodes cycling through tasks
```

### Docker

```bash
docker build -t contract-negotiation-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your-token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  contract-negotiation-env
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace / LLM API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `BENCHMARK` | No | `contract_negotiation` | Benchmark name in [START] log line |
| `PORT` | No | `7860` | Server port |

---

## Project Structure

```
contract_env/
├── env/
│   ├── environment.py   # ContractEnv — reset/step/state, 7-step episodes
│   ├── graders.py       # evaluate_action() + 5 task-specific grader functions
│   ├── models.py        # Pydantic v2 models: Action, Observation, Reward
│   └── tasks.py         # 5 NegotiationTask definitions with metadata
├── server/
│   └── app.py           # FastAPI server (port 7860)
├── tests/
│   ├── test_graders.py  # 13 unit tests covering all grader edge cases
│   ├── test_api.py      # API endpoint tests
│   └── test_smoke.py    # Smoke tests
└── client.py            # HTTP client helper
inference.py             # LLM-driven baseline agent
openenv.yaml             # OpenEnv manifest (spec_version: 1)
Dockerfile               # Python 3.10-slim container, port 7860
verify_graders.py        # Pre-submission grader validation script
```
