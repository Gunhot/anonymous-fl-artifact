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
    "n_procs": 2,
    "nodes": 100,
    "fraction": 0.1,
    "round": 100,
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
    "max_len": 256,
    "model": "distilbert",
    "dataset": "20news",
    "batch_size": 16,
    "iid": 2,
    "beta": 0.1,
    "local_epoch": 1,
    "lr": 0.0001,
    "lr_decay": 0.999,
    "opt": "adam",
    "eval_mode": "main",
}


def make_args(**overrides):
    args = DEFAULT_ARGS.copy()
    args.update(overrides)
    return args


experiments = [
    ("20news FedAvg Cross-device Non-IID", make_args(DP="none", sigma=0.0)),
    ("20news Gaussian Cross-device Non-IID rho=1.3", make_args(n_procs=1, DP="gausg", sigma=13000.0)),
    ("20news CloakFL Cross-device Non-IID rho=0.7", make_args(DP="ours", sigma=7000.0)),
    ("20news CloakFL Cross-device Non-IID rho=0.8", make_args(DP="ours", sigma=8000.0)),
    ("20news CloakFL Cross-device Non-IID rho=0.9", make_args(DP="ours", sigma=9000.0)),
]


accuracy_pattern = re.compile(r"Accuracy:\s*([0-9.]+)")


def read_acc(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    matches = accuracy_pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


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

        acc = read_acc(filepath)
        if acc is None:
            continue

        if filename == "server.txt":
            buckets["server"].append(acc)
        elif re.fullmatch(r"proxy_train_\d+\.txt", filename):
            buckets["proxy train"].append(acc)
        elif re.fullmatch(r"proxy_finetune_\d+\.txt", filename):
            buckets["proxy finetune"].append(acc)
        elif re.fullmatch(r"proxy_\d+\.txt", filename):
            buckets["proxy"].append(acc)
        elif re.fullmatch(r"collusion_before_\d+\.txt", filename):
            buckets["collusion before"].append(acc)
        elif re.fullmatch(r"collusion_finetune_\d+\.txt", filename):
            buckets["collusion finetune"].append(acc)
        elif re.fullmatch(r"collusion_after_\d+\.txt", filename):
            buckets["collusion after"].append(acc)

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
        print(f"{label}: {(mean * 100):.4f}")

    print()
