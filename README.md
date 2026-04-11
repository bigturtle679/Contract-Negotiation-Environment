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
- **Multi-turn dynamics**: the counterparty pushes back on proposals, requiring
  adaptive negotiation strategies across multiple rounds.

---

## Tasks

| ID | Difficulty | Clause Type | Risk Level | Hidden Trap |
|----|-----------|-------------|-----------|-------------|
| `easy_unlimited_liability` | Easy (1/5) | Liability | HIGH | No |
| `medium_auto_renewal` | Medium (2/5) | Term/Renewal | MODERATE | No |
| `hard_conflicting_obligations` | Hard (4/5) | Performance/Changes | HIGH | Yes |
| `easy_compliance_agreement` | Easy+ (2/5) | Compliance | LOW | No |
| `hard_intellectual_property` | Hard+ (5/5) | IP Ownership | HIGH | Yes |
| `medium_confidentiality_nda` | Medium+ (3/5) | Confidentiality | MODERATE | Yes |
| `hard_termination_convenience` | Hard++ (4/5) | Termination | HIGH | Yes |
| `expert_data_protection` | Expert (5/5) | Data Protection | HIGH | Yes |

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

**medium_confidentiality_nda** — An overbroad NDA with perpetual obligations and
no carve-outs for public information. The agent must narrow the scope, add a
time limit (3 years), and carve out publicly available and independently
developed information.

**hard_termination_convenience** — A one-sided termination clause allowing only
the Supplier to terminate at will with 5-day notice, while the Customer has no
termination rights and waives all remedies. The agent must establish mutual
termination, add a 30-day cure period, and include transition/wind-down
provisions.

**expert_data_protection** — A clause giving the Supplier blanket authority to
process personal data, transfer it to any jurisdiction, engage sub-processors
without notice, and waive data-subject rights. The agent must add DPA
requirements, 72-hour breach notification, sub-processor consent, data-subject
rights assistance, and data deletion obligations.

---

## Opponent Simulation

Each task includes **opponent responses** keyed by action type. When the agent
takes an action (e.g., `FLAG_RISK`, `EDIT_CLAUSE`), the counterparty replies
with a contextually appropriate pushback or counter-proposal, creating realistic
multi-turn negotiation dynamics:

```
agent   → FLAG_RISK
opponent → "Our legal team considers this standard. What specific cap do you propose?"
agent   → EDIT_CLAUSE (with cap at 12 months)
opponent → "We can accept a cap but consequential damages must remain."
agent   → PROPOSE_COUNTER (addressing consequential damages)
...
```

Opponent replies appear in the `negotiation_history` and in `info.opponent_reply`.

---

## Observation Space

Every call to `/reset` or `/step` returns an `Observation`:

```json
{
  "contract_text": "string — the current clause text (may be rewritten after EDIT/PROPOSE)",
  "clause_type": "string — e.g. liability, term_renewal, intellectual_property, confidentiality, termination, data_protection",
  "risk_level": "float ∈ (0, 1) — observed risk density (0=safe, 1=highly risky)",
  "step_count": "int — steps taken so far (0 = just reset)",
  "negotiation_history": [
    "opponent|[Counterparty] Unlimited indemnity is standard.",
    "agent|step=1 action=FLAG_RISK content_len=0",
    "opponent|[Counterparty] Our legal team considers this standard.",
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
reward = 0.35 × correctness
       + 0.25 × improvement
       + 0.25 × risk_alignment
       + 0.10 × semantic_similarity
       + 0.05 × completeness
```

| Component | What it measures |
|-----------|-----------------|
| **Correctness** (35%) | For EDIT/PROPOSE: how much risky language was *removed* from the original. For FLAG/REJECT/ACCEPT: how many risk keywords are identified in context. |
| **Improvement** (25%) | How well the proposed edit matches safe keywords and the expected safe rewrite. |
| **Risk Alignment** (25%) | Whether the chosen action is appropriate for the current risk level (e.g., editing a HIGH-risk clause scores 0.92×; accepting it scores 0.20×). |
| **Semantic Similarity** (10%) | Combined Jaccard + cosine similarity between the rewrite and the expected safe edit. |
| **Completeness** (5%) | Fraction of required legal elements present in the rewritten clause (e.g., liability cap, notice period, cure clause). |

### Task-specific adjustments

| Task | Adjustment |
|------|-----------|
| Easy | +8% bonus when safe edit matches well |
| Medium | −35% penalty for accepting risky auto-renewal terms |
| Hard | −50% penalty when hidden trap markers remain in the proposed text |
| Easy+ | +6% bonus for including breach-notification language |
| Hard+ | −45% penalty for unresolved IP traps; +7% bonus for explicit customer ownership |
| Medium+ | +8% bonus for well-scoped NDA; −30% for accepting overbroad terms |
| Hard++ | −45% penalty for unresolved one-sided termination; +9% for cure-period language |
| Expert | −50% penalty for missing data-protection safeguards; +10% for GDPR language (requires ≥2 indicators) |

Blocked accepts (accepting HIGH-risk text) are clamped to `0.001`.

### Episode score

The `[END]` line reports `score = mean(rewards over all steps)`.
An episode is considered successful if `score ≥ 0.50`.

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
python -m pytest contract_env/tests/ -v   # 51 tests
```

### Run the server

```bash
uvicorn contract_env.server.app:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
export HF_TOKEN="your-huggingface-token"
python inference.py --benchmark    # one episode per task (8 total)
python inference.py --episodes 3   # run 3 episodes cycling through tasks

# Against the Docker API server (for competition evaluation):
python inference.py --benchmark --mode api
```

### Docker

```bash
docker build -t contract-negotiation-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your-token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  contract-negotiation-env

# Then run inference against the Docker server:
python inference.py --benchmark --mode api
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace / LLM API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `BENCHMARK` | No | `contract_negotiation` | Benchmark name in [START] log line |
| `ENV_SERVER_URL` | No | `http://localhost:7860` | Docker server URL (for `--mode api`) |
| `PORT` | No | `7860` | Server port |

---

## Project Structure

```
contract_env/
├── env/
│   ├── environment.py   # ContractEnv — reset/step/state, 7-step episodes, opponent simulation
│   ├── graders.py       # evaluate_action() + 8 task-specific grader functions + semantic/completeness scoring
│   ├── models.py        # Pydantic v2 models: Action, Observation, Reward
│   └── tasks.py         # 8 NegotiationTask definitions with metadata + opponent responses
├── server/
│   └── app.py           # FastAPI server (port 7860)
├── tests/
│   ├── test_graders.py  # Grader unit tests covering all edge cases + new metrics
│   ├── test_api.py      # API endpoint tests
│   └── test_smoke.py    # Smoke tests including opponent simulation + opponent stance parsing
└── client.py            # HTTP client helper with from_docker_image() support
inference.py             # LLM-driven agent with opponent-aware multi-turn strategy + HTTP mode
openenv.yaml             # OpenEnv manifest (spec_version: 1, 8 graded tasks, action_space)
Dockerfile               # Python 3.10-slim container, port 7860
verify_graders.py        # Pre-submission grader validation script
```
