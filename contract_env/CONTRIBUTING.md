# Contributing

## Principles

- Keep **`ContractEnv`** deterministic: no hidden randomness in transitions unless explicitly seeded and documented.
- **Grader changes** affect both training signals and the reference agent—update `tests/` and `inference.py --benchmark` expectations together.
- Prefer **structured metrics** (`Reward.metrics`, `info`) over ad-hock string parsing.

## Checks before a PR

1. `python -m unittest discover -s tests -v`
2. `python inference.py --benchmark`
3. `docker build` from `contract_env/`

## Style

- Match existing type hints and Pydantic models.
- Avoid widening action/observation contracts without updating `openenv.yaml` and this README’s API table.
