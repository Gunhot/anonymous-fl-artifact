import os
import re
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if TRANSFORMER_SRC_DIR not in sys.path:
    sys.path.insert(0, TRANSFORMER_SRC_DIR)

from arguments import generate_log_name

base_dir = os.path.join(SCRIPT_DIR, "save")


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
}


def make_args(**overrides):
    args = DEFAULT_ARGS.copy()
    args.update(overrides)
    return args


def relative_log_name(args):
    return os.path.relpath(generate_log_name(SimpleNamespace(**args)), base_dir)


fedavg_args = make_args(DP="none", sigma=0.0)
experiments = [
    ("FedAvg", fedavg_args),
    ("ρ=1.3", make_args(n_procs=1, DP="gausg", sigma=13000.0)),
    ("FedAvg", fedavg_args),
    ("ρ=0.8", make_args(DP="ours", sigma=8000.0)),

    ("FedAvg", fedavg_args),
    ("ρ=0.7", make_args(DP="ours", sigma=7000.0)),
    ("FedAvg", fedavg_args),
    ("ρ=0.8", make_args(DP="ours", sigma=8000.0)),
    ("FedAvg", fedavg_args),
    ("ρ=0.9", make_args(DP="ours", sigma=9000.0)),

    ("FedAvg", fedavg_args),
    ("ρ=0.8", make_args(DP="ours", sigma=8000.0)),
]

titles = [title for title, _ in experiments]
file_names = [relative_log_name(args) for _, args in experiments]


# Tune the figure for compact PNG-style inspection.
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
})


tracked_clients = [0, 1, 2]
keys = ["server"]
for client_id in tracked_clients:
    keys.extend([
        f"proxy_{client_id}",
        f"proxy_train_{client_id}",
        f"proxy_finetune_{client_id}",
    ])

group_defs = {
    "proxy": [f"proxy_{client_id}" for client_id in tracked_clients],
}

accuracy_pattern = re.compile(r"Round\s+(\d+)\s*\|\s*Accuracy:\s*([0-9.]+)")


def read_accuracy_series(filepath):
    """Read one accuracy log and return rounds plus percentages."""
    if not os.path.exists(filepath):
        return None

    rounds = [0]
    accs = [0.0]

    with open(filepath, "r") as f:
        for line in f:
            match = accuracy_pattern.search(line)
            if match is None:
                continue

            rounds.append(int(match.group(1)))
            accs.append(float(match.group(2)) * 100.0)

    if len(rounds) == 1:
        return None
    return rounds, accs


def densify(rounds, accs, max_round):
    """Forward-fill sparse tracking points so client averages share one x-axis."""
    round_to_acc = dict(zip(rounds, accs))
    dense_accs = []
    last_acc = 0.0

    for round_idx in range(max_round + 1):
        if round_idx in round_to_acc:
            last_acc = round_to_acc[round_idx]
        dense_accs.append(last_acc)

    return dense_accs


def plot_comparison(compare_idx):
    """Plot one comparison entry against the FedAvg entry immediately before it."""
    file_name = file_names[compare_idx]
    title = titles[compare_idx]
    print(title)

    exp_dir = os.path.join(base_dir, file_name)
    if not os.path.isdir(exp_dir):
        print(f"missing directory: {exp_dir}")
        return

    series = {}
    max_round = 0
    for key in keys:
        parsed = read_accuracy_series(os.path.join(exp_dir, f"{key}.txt"))
        if parsed is None:
            continue

        rounds, accs = parsed
        series[key] = (rounds, accs)
        max_round = max(max_round, max(rounds))

    fedavg_series = None
    baseline_max = 0.0
    fedavg_idx = compare_idx - 1
    if fedavg_idx >= 0:
        fedavg_dir = os.path.join(base_dir, file_names[fedavg_idx])
        fedavg_series = read_accuracy_series(os.path.join(fedavg_dir, "server.txt"))
        if fedavg_series is not None:
            max_round = max(max_round, max(fedavg_series[0]))
            baseline_max = max(fedavg_series[1])

    if max_round == 0:
        print(f"no plottable accuracy logs: {exp_dir}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.6), dpi=180)
    y_max_candidates = [baseline_max]

    # Plot the previous FedAvg server curve as the baseline.
    if fedavg_series is not None:
        ref_rounds, ref_accs = fedavg_series
        y_max_candidates.append(max(ref_accs))
        ax.plot(
            ref_rounds,
            ref_accs,
            color="black",
            linestyle="--",
            lw=2.0,
            zorder=1,
            label=f"FedAvg ({max(ref_accs):.1f}%)",
        )

    # Plot the current experiment server curve when the log exists.
    if "server" in series:
        rounds_vals, accs_vals = series["server"]
        y_max_candidates.append(max(accs_vals))
        markevery = max(1, len(rounds_vals) // 12)
        ax.plot(
            rounds_vals,
            accs_vals,
            label=f"server ({max(accs_vals):.1f}%)",
            lw=2.5,
            linestyle="-",
            marker="o",
            markersize=3,
            markevery=markevery,
            zorder=3,
        )

    # Average tracked clients after forward-filling their sparse evaluations.
    for group_name, members in group_defs.items():
        y_list = []
        for key in members:
            if key not in series:
                continue

            rounds_vals, accs_vals = series[key]
            y_list.append(densify(rounds_vals, accs_vals, max_round))

        if not y_list:
            continue

        ys = np.array(y_list)
        x = list(range(max_round + 1))
        y_mean = ys.mean(axis=0)
        y_max_candidates.append(float(y_mean.max()))

        ax.plot(
            x,
            y_mean,
            lw=2,
            linestyle=":",
            label=f"{group_name} ({y_mean.max():.1f}%)",
            zorder=2,
        )

    # Draw a thin horizontal reference at the FedAvg best accuracy.
    if baseline_max > 0:
        ax.axhline(
            y=baseline_max,
            linestyle="--",
            color="gray",
            lw=0.8,
            alpha=0.5,
            zorder=0,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(y_max_candidates) + 5)
    ax.set_xlim(0, max_round)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=13, frameon=False, loc="lower right")

    plt.tight_layout()
    plt.show()


if len(file_names) != len(titles):
    raise ValueError("file_names and titles must have the same length.")

# Compare entries at 1, 3, 5, ... against the FedAvg entry right before them.
compare_indices = [idx for idx in range(len(file_names)) if idx % 2 == 1]

for idx in compare_indices:
    plot_comparison(idx)
