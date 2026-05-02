import os
import re
import sys
from types import SimpleNamespace


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if TRANSFORMER_SRC_DIR not in sys.path:
    sys.path.insert(0, TRANSFORMER_SRC_DIR)

from arguments import generate_log_name


DEFAULT_ARGS = {
    "n_procs": 1,
    "nodes": 100,
    "fraction": 0.1,
    "round": 30,
    "ft_subset": 1.0,
    "DP": "none",
    "sigma": 0.0,
    "noise_update": 0,
    "p1": 0.0,
    "p2": 0.0,
    "omega": 0,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "use_pretrained": 1,
    "seed": 2,
    "max_len": 512,
    "model": "gpt2s",
    "dataset": "e2e",
    "batch_size": 4,
    "iid": 2,
    "beta": 0.1,
    "local_epoch": 1,
    "lr": 0.001,
    "lr_decay": 0.999,
    "opt": "adam",
    "eval_mode": "main",
}


def make_args(**overrides):
    args = DEFAULT_ARGS.copy()
    args.update(overrides)
    return args


experiments = [
    ("e2e FedAvg Cross-device Non-IID", make_args(DP="none", sigma=0.0)),
    ("e2e CloakFL Cross-device Non-IID rho=1.0", make_args(n_procs=2, DP="ours", sigma=10000.0)),
]


metric_pattern = re.compile(
    r"Round\s+(\d+)\s*\|\s*"
    r"BLEU:\s*([0-9]*\.?[0-9]+)\s*\|\s*"
    r"NIST:\s*([0-9]*\.?[0-9]+)\s*\|\s*"
    r"METEOR:\s*([0-9]*\.?[0-9]+)\s*\|\s*"
    r"ROUGE-L:\s*([0-9]*\.?[0-9]+)\s*\|\s*"
    r"CIDEr:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def read_bleu(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    matches = metric_pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1][1])


def mean_std(values):
    if not values:
        return 0.0, 0.0

    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, var ** 0.5


def collect_values(directory):
    buckets = {
        "server": [],
        "proxy": [],
        "proxy train": [],
        "proxy finetune": [],
        "collusion before": [],
        "collusion finetune": [],
        "collusion after": [],
    }

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        bleu = read_bleu(filepath)
        if bleu is None:
            continue

        if filename == "server.txt":
            buckets["server"].append(bleu)
        elif re.fullmatch(r"proxy_train_\d+\.txt", filename):
            buckets["proxy train"].append(bleu)
        elif re.fullmatch(r"proxy_finetune_\d+\.txt", filename):
            buckets["proxy finetune"].append(bleu)
        elif re.fullmatch(r"proxy_\d+\.txt", filename):
            buckets["proxy"].append(bleu)
        elif re.fullmatch(r"collusion_before_\d+\.txt", filename):
            buckets["collusion before"].append(bleu)
        elif re.fullmatch(r"collusion_finetune_\d+\.txt", filename):
            buckets["collusion finetune"].append(bleu)
        elif re.fullmatch(r"collusion_after_\d+\.txt", filename):
            buckets["collusion after"].append(bleu)

    return buckets


for title, args_dict in experiments:
    directory = generate_log_name(SimpleNamespace(**args_dict))
    print(f"[{title}]")

    if not os.path.isdir(directory):
        print(f"missing directory: {directory}")
        print()
        continue

    values = collect_values(directory)
    for label, vals in values.items():
        mean, _ = mean_std(vals)
        print(f"{label}: {mean:.4f}")

    print()
