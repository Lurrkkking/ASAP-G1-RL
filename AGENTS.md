# Repository Guidelines

## Project Structure & Module Organization
- `humanoidverse/`: core training and evaluation code (`train_agent.py`, `eval_agent.py`), agents, envs, simulator backends, and Hydra configs under `humanoidverse/config/`.
- `sim2real/`: deployment and runtime control utilities for sim-to-sim/sim-to-real, including ONNX policies in `sim2real/models/`.
- `isaac_utils/`: shared math/rotation helpers packaged separately.
- `scripts/`: one-off data processing and visualization helpers (for example `scripts/data_process/fit_smpl_motion.py`).
- `imgs/`: documentation media assets. Runtime outputs go to `logs/`, `logs_eval/`, and `runs/` (do not commit generated artifacts).

## Build, Test, and Development Commands
- `pip install -e . && pip install -e isaac_utils`: install the main package and local utility package in editable mode.
- `python humanoidverse/train_agent.py +simulator=isaacgym +exp=locomotion ...`: start training with Hydra overrides.
- `python humanoidverse/eval_agent.py +checkpoint=logs/<run>/model_<iter>.pt`: evaluate a saved checkpoint.
- `python sim2real/state_publisher.py` (or related `sim2real/*.py` entrypoints): run deployment-side components.

## Coding Style & Naming Conventions
- Follow existing Python style: 4-space indentation, snake_case for functions/variables/files, PascalCase for classes.
- Keep modules focused and config-driven; prefer adding behavior through Hydra config groups rather than hardcoding.
- Use descriptive config names matching existing patterns, e.g. `humanoidverse/config/exp/<task>.yaml` and `rewards/<domain>/<name>.yaml`.

## Testing Guidelines
- No formal `tests/` suite is currently enforced. Validate changes with targeted run checks:
  - Training smoke test: `num_envs=1 headless=False`.
  - Eval smoke test: run `eval_agent.py` with a known checkpoint.
- Treat `sim2real/utils/test_xbox.py` as a manual hardware-side check, not CI coverage.

## Commit & Pull Request Guidelines
- Match repository history: short, imperative subjects (for example `add delta_a config`, `fix training issue`, `update README`).
- Keep commits scoped to one logical change (code + config + docs together when tightly coupled).
- PRs should include:
  - What changed and why.
  - Exact repro/train/eval command(s) used.
  - Linked issue (if any) and result evidence (logs, plots, or GIFs for behavior changes).

## Security & Configuration Tips
- Never commit secrets or machine-specific paths.
- Keep large checkpoints, generated logs, and raw datasets out of Git; use external storage and reference paths via config.
