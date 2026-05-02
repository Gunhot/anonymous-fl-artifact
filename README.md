# Artifact README

This artifact contains the dataset/model download scripts, the vision scheduler experiments, and the transformer scheduler experiments used for the reported results.

## Layout

- `download_datasets.py`: downloads all four datasets into `data/`.
- `download_models.py`: downloads transformer checkpoints into `model/`.
- `nn_models.py`: contains the local vision model definitions used by the artifact.
- `vision_scheduler/`: CIFAR-100 and Tiny-ImageNet experiments.
- `transformer_scheduler/`: 20News and E2E experiments.
- `data/`, `model/`, and `save/` are runtime outputs and are ignored by git.

## Prepare Data and Models

Run these commands from the artifact root:

```bash
python download_datasets.py
python download_models.py
```

`download_datasets.py` recreates `data/` from scratch each time. The final dataset layout is:

```text
data/20news
data/cifar100
data/e2e
data/tiny-imagenet
```

`download_models.py` recreates `model/` from scratch each time. The final model layout is:

```text
model/gpt2
model/distilbert-base-uncased
```

Vision models are defined locally in `nn_models.py`; they are not downloaded.

## Run Experiments

All run scripts pin `CUDA_VISIBLE_DEVICES=0`. They also use `PYTHON="${PYTHON:-python}"`, so a specific interpreter can be selected with `PYTHON=/path/to/python`.

Vision experiments:

```bash
cd vision_scheduler
sh run_cifar100.sh
sh run_tiny_imagenet.sh
```

Transformer experiments:

```bash
cd transformer_scheduler
sh run_20news.sh
sh run_e2e.sh
```

Each run script contains the exact experiment list used by the matching result-summary script.

## Summarize Results

Vision summaries:

```bash
cd vision_scheduler
python cifar100_results.py
python tiny-imagent_results.py
```

Transformer summaries:

```bash
cd transformer_scheduler
python 20news_results.py
python e2e_results.py
```

The summary scripts read logs from each scheduler's `save/` directory. Missing experiment directories are reported explicitly.

## Log Paths

Experiment logs are written under:

```text
vision_scheduler/save/<dataset>/<experiment-name>/
transformer_scheduler/save/<dataset>/<experiment-name>/
```

Important output files include:

- `server.txt`: final server evaluation.
- `proxy_*.txt`: proxy evaluation.
- `proxy_train_*.txt`: proxy train evaluation.
- `proxy_finetune_*.txt`: proxy finetune evaluation.
- `collusion_before_*.txt`: collusion evaluation before attack-side update.
- `collusion_after_*.txt`: collusion evaluation after attack-side update.

The artifact code evaluates the final round only.

## Notes

- CIFAR-100 uses `MobileNetV2`.
- Tiny-ImageNet uses `ResNet50`.
- 20News uses `distilbert`.
- E2E uses `gpt2s`.
- `--DP ours` uses the artifact-fixed CloakFL configuration internally.
- The old nested `src/scripts/` presets are not needed; the root `run_*.sh` files are the canonical artifact commands.
