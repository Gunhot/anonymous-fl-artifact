import os
import re
import sys
from types import SimpleNamespace


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if VISION_SRC_DIR not in sys.path:
    sys.path.insert(0, VISION_SRC_DIR)

from arguments import generate_log_name


DEFAULT_ARGS = {
    "n_procs": 2,
    "nodes": 100,
    "fraction": 0.1,
    "round": 100,
    "dataset": "cifar100",
    "model": "MobileNetV2",
    "batch_size": 64,
    "iid": 2,
    "beta": 0.1,
    "local_epoch": 5,
    "lr": 0.1,
    "ft_lr": 0.01,
    "opt": "sgd",
    "lr_decay": 0.999,
    "DP": "none",
    "sigma": 0,
    "p1": 0.0,
    "p2": 0.0,
    "omega": 0,
    "seed": 42,
    "qsn_fixed_mask": False,
    "noise_update": 0,
    "eval_mode": "main",
}


def make_args(**overrides):
    args = DEFAULT_ARGS.copy()
    args.update(overrides)
    return args


experiments = [
    ("cifar100 FedAvg Cross-device Non-IID", make_args(DP="none", sigma=0)),
    ("cifar100 CloakFL Cross-device Non-IID rho=0.5", make_args(DP="ours", sigma=5000)),
    ("cifar100 CloakFL Cross-device Non-IID rho=0.6", make_args(DP="ours", sigma=6000)),
    ("cifar100 CloakFL Cross-device Non-IID rho=0.7", make_args(DP="ours", sigma=7000)),
    ("cifar100 Gaussian Cross-device Non-IID rho=1.0", make_args(DP="gausg", sigma=10000)),
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
        if not filename.endswith(".txt") or filename == "dataset_analysis.txt":
            continue

        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        acc = read_acc(filepath)
        if acc is None:
            continue

        if filename == "server.txt":
            buckets["server"].append(acc)
        elif filename.startswith("proxy_train_"):
            buckets["proxy train"].append(acc)
        elif filename.startswith("proxy_finetune_"):
            buckets["proxy finetune"].append(acc)
        elif filename.startswith("proxy_"):
            buckets["proxy"].append(acc)
        elif filename.startswith("collusion_before_"):
            buckets["collusion before"].append(acc)
        elif filename.startswith("collusion_finetune_"):
            buckets["collusion finetune"].append(acc)
        elif filename.startswith("collusion_after_"):
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
