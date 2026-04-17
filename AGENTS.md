# Repository Guidelines

IMPORTANT: Before doing residual-gap experiments or reporting metrics from `genesis_simulation/residual_dataset/*`, read `CODEX_HANDOFF.md` in the repo root first.

## Core Algorithmic Principle
The most important conceptual distinction in this repo is the separation between the residual action model and the PPO main policy.

- The residual action model (`pi^Delta`, delta action model, action patch) is intended to solve a sim-gap / system-identification problem.
- Its job is: given current state `s_t` and original action `a_t`, output a correction `Delta a_t` so that executing `(a_t + Delta a_t)` in simulation produces physics closer to the desired / real physics outcome.
- This model is not supposed to optimize motion-tracking reward as its primary objective. Its purpose is environment correction / action compensation.

- The PPO main policy (`pi`) is a separate object with a different optimization target.
- After the residual action model is trained and frozen, PPO should treat it as a black-box patch inserted in front of the simulator / low-level action execution path.
- PPO does not need to "understand" the sim-gap directly. PPO's job is simply to maximize motion-tracking reward in the resulting patched environment, i.e. track the fixed reference trajectory as well as possible.

When reading or modifying code, do not collapse these two objectives into one:
- `pi^Delta` target: reduce sim-gap through action correction.
- PPO target: maximize tracking reward in the patched environment.

If an implementation or experiment mixes these roles, call that out explicitly instead of assuming the intended algorithmic separation is already preserved.

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
