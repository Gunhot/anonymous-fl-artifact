# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the training system. Use `src/main.py` as the primary entry point and `src/arguments.py` for all CLI/config flags and log-path generation. Federated orchestration lives in `src/node.py`, `src/server.py`, and `src/workers.py`. Dataset logic is under `src/dataLoader/`, and model backbones are provided by `../nn_models.py`. Utility scripts for repeatable experiments are in `src/scripts/`. Runtime artifacts are written to `save/` (gitignored).

## Build, Test, and Development Commands
There is no separate build step; run from `src/` so relative paths resolve correctly.

- `cd src && python main.py --help`: list all experiment options.
- `cd src && python main.py --round 1 --nodes 2 --n_procs 1 --model MobileNetV2 --dataset cifar100`: fast smoke run.
- `cd src && bash scripts/r50.sh`: run curated ResNet50 experiment variants.

## Coding Style & Naming Conventions
Use Python with 4-space indentation. Prefer `snake_case` for functions/variables/flags and `PascalCase` for classes/models (`Client`, `Server`, `ResNet50`). Keep new runtime flags centralized in `src/arguments.py`, and keep defaults reproducible (seed-aware changes should be explicit). Favor small, composable functions over long inlined blocks.

## Testing Guidelines
No formal test suite is currently configured. Validate changes with reduced-cost runs and verify logs are produced under `save/<dataset>/<log_name>/` (for example `server.txt` and final-round proxy logs). For data/model changes, confirm one full round completes without worker crashes or NaN/Inf regressions.

## Commit & Pull Request Guidelines
History shows short, experiment-oriented commit subjects (often model/dataset/hyperparameter focused, sometimes bilingual). Keep commits concise and scoped to one change. PRs should include:

- exact command(s) used to reproduce,
- dataset + hardware/GPU assumptions,
- key output path(s) under `save/`,
- before/after metrics.

## Data & Configuration Tips
Default dataset root is `artifact/data` (see `src/dataLoader/dataset.py`). `ours` uses centered perturbation-direction noise with a fixed clean-server scale base and pL+ update rule internally, and evaluation always runs on the final round. Ensure OOD/image assets exist in `src/freerider/` and `src/icons/` before running experiments that depend on them.
