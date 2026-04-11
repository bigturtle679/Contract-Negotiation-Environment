---
title: Contract Negotiation Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Contract Negotiation Environment

An AI-powered OpenEnv environment for evaluating contract-negotiation agents
through hybrid rule-based and LLM-driven decision making.

## Overview

This project simulates real-world contract negotiation scenarios where an AI
agent must:

1. **Analyse** contract clauses to identify legal risks (unlimited liability,
   hidden traps, one-sided IP terms, etc.).
2. **Decide** on the best negotiation action: flag the risk, edit the clause,
   propose a counter-offer, reject, or accept.
3. **Generate** safer clause rewrites that protect the customer while keeping
   commercially reasonable terms.

Agents are scored on three dimensions:
- **Correctness** — how well the agent identifies risky language.
- **Improvement** — how much the proposed edits reduce risk.
- **Risk alignment** — whether the chosen action matches the actual risk level.

## Tasks

| ID | Difficulty | Clause Type | Industry |
|----|-----------|-------------|----------|
| `easy_unlimited_liability` | EASY | Liability | SaaS B2B |
| `medium_auto_renewal` | MEDIUM | Term/Renewal | SaaS B2B |
| `hard_conflicting_obligations` | HARD | Performance/Changes | Professional Services |
| `easy_compliance_agreement` | EASY+ | Compliance | SaaS B2B |
| `hard_intellectual_property` | HARD+ | IP Ownership | Professional Services |

Each task has a **dedicated grader** with difficulty-specific scoring adjustments
(e.g., harder tasks penalise unresolved hidden traps).

## Structure

```
contract_env/
├── env/
│   ├── environment.py   # ContractEnv — the main OpenEnv environment
│   ├── graders.py       # Task-specific grading functions
│   ├── models.py        # Pydantic models (Action, Reward, Observation)
│   └── tasks.py         # Task definitions and metadata
├── server/
│   └── app.py           # FastAPI server exposing /reset, /step, /state, /tasks
├── tests/               # Unit tests for API, graders, and environment
└── scripts/             # Helper scripts for local/Docker runs
inference.py             # LLM-driven inference agent
openenv.yaml             # OpenEnv manifest
Dockerfile               # Production container definition
```

## Quick Start

### Local development

```bash
pip install -e ".[dev]"
python -m pytest contract_env/tests/ -v
```

### Run the server

```bash
uvicorn contract_env.server.app:app --host 0.0.0.0 --port 7860
```

### Run inference

```bash
export HF_TOKEN="your-huggingface-token"
python inference.py --episodes 5
python inference.py --benchmark   # one episode per task
```

### Docker

```bash
docker build -t contract-negotiation-env .
docker run -p 7860:7860 contract-negotiation-env
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/state` | Current environment state |
| `POST` | `/reset` | Reset and get first observation |
| `POST` | `/step` | Submit an action, receive reward |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace API token |
| `PORT` | No | `7860` | Server port |