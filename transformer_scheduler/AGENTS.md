# Repository Guidelines

## Project Structure & Module Organization
Core training code lives in `src/`. Use `src/main.py` as the experiment entrypoint; it coordinates federated rounds, multiprocessing workers, and logging. `src/server.py`, `src/node.py`, and `src/workers.py` implement aggregation, client training, and evaluation workers. Dataset loading and client partitioning are in `src/dataLoader/`. NLP runs use Transformers + PEFT in `workers.py`. Root-level `run_20news.sh` and `run_e2e.sh` provide the repeatable artifact experiments. Run artifacts are written under `save/` (for example `server.txt` and generation outputs).

## Build, Test, and Development Commands
Run commands from `src/`:

```bash
cd src
python main.py --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 1 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16
```

Use artifact run scripts for the full experiment sets:

```bash
sh run_20news.sh
sh run_e2e.sh
```

Use short smoke runs before pushing changes:

```bash
python main.py --dataset 20news --model distilbert --round 1
```

For generation metric checks:

```bash
python e2e-metrics/measure_scores.py <ref_file> <pred_file> -p
```

## Coding Style & Naming Conventions
Use Python with 4-space indentation. Follow existing naming: `snake_case` for functions/variables, `PascalCase` for classes (`Server`, `Client`). Keep CLI arguments centralized in `src/arguments.py` and preserve existing flag style (`--local_epoch`). No formatter/linter is enforced in-repo; keep imports grouped and changes focused.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with targeted smoke experiments and log checks in `save/<dataset>/...`. For classification tasks, confirm accuracy logging in `<type>.txt`. For generation tasks, confirm BLEU/NIST/METEOR/ROUGE-L/CIDEr logging and `measure_scores.py` execution.

## Commit & Pull Request Guidelines
Git history shows short, experiment-oriented commit subjects (often dataset/seed/strategy focused) rather than strict conventional commits. Keep subjects concise and scoped, e.g. `20news/distilbert: fix proxy evaluation logging`. In PRs, include: exact run command(s), key metric/log deltas, affected datasets/models, and any required local path assumptions.

## Configuration Tips
Runtime expects artifact-local data in `../data` and artifact-local models in `../model` from the repository root. The artifact run scripts pin `CUDA_VISIBLE_DEVICES=0`.
