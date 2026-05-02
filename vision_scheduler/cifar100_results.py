import os
import re
import sys
from types import SimpleNamespace


VISION_SCHEDULER_DIR = os.path.dirname(__file__)
VISION_SRC_DIR = os.path.join(VISION_SCHEDULER_DIR, "src")
sys.path.insert(0, VISION_SRC_DIR)

from arguments import generate_log_name


experiments = [
    # title, nodes, fraction, round, iid, beta, DP, sigma, p2, omega
    # Cross-silo / Non-IID
    ('cifar100 CloakFL Cross-silo Non-IID (rho=0.6)', 20, 0.5, 30, 2, 0.1, 'ours', 6000, 0.0, 0),
    ('cifar100 FedQSN Cross-silo Non-IID (8-bit, 2%)', 20, 0.5, 30, 2, 0.1, 'qsn', 0, 0.02, 8),
    ('cifar100 FedQSN Cross-silo Non-IID (omega=4, p1=p2=0)', 20, 0.5, 30, 2, 0.1, 'qsn', 0, 0.0, 4),
    ('cifar100 FedQSN Cross-silo Non-IID (omega=8, p1=p2=0)', 20, 0.5, 30, 2, 0.1, 'qsn', 0, 0.0, 8),
    ('cifar100 FedAvg Cross-silo Non-IID', 20, 0.5, 30, 2, 0.1, 'none', 0, 0.0, 0),

    # Cross-silo / IID
    ('cifar100 CloakFL Cross-silo IID (rho=0.7)', 20, 0.5, 30, 1, 0.0, 'ours', 7000, 0.0, 0),
    ('cifar100 FedQSN Cross-silo IID (8-bit, 3%)', 20, 0.5, 30, 1, 0.0, 'qsn', 0, 0.03, 8),
    ('cifar100 FedQSN Cross-silo IID (omega=4, p1=p2=0)', 20, 0.5, 30, 1, 0.0, 'qsn', 0, 0.0, 4),
    ('cifar100 FedQSN Cross-silo IID (omega=8, p1=p2=0)', 20, 0.5, 30, 1, 0.0, 'qsn', 0, 0.0, 8),
    ('cifar100 FedAvg Cross-silo IID', 20, 0.5, 30, 1, 0.0, 'none', 0, 0.0, 0),

    # Cross-device / Non-IID
    ('cifar100 CloakFL Cross-device Non-IID (rho=0.6)', 100, 0.1, 100, 2, 0.1, 'ours', 6000, 0.0, 0),
    ('cifar100 FedQSN Cross-device Non-IID (8-bit, 1%)', 100, 0.1, 100, 2, 0.1, 'qsn', 0, 0.01, 8),
    ('cifar100 FedQSN Cross-device Non-IID (omega=4, p1=p2=0)', 100, 0.1, 100, 2, 0.1, 'qsn', 0, 0.0, 4),
    ('cifar100 FedQSN Cross-device Non-IID (omega=8, p1=p2=0)', 100, 0.1, 100, 2, 0.1, 'qsn', 0, 0.0, 8),
    ('cifar100 FedAvg Cross-device Non-IID', 100, 0.1, 100, 2, 0.1, 'none', 0, 0.0, 0),

    # Cross-device / IID
    ('cifar100 CloakFL Cross-device IID (rho=0.7)', 100, 0.1, 100, 1, 0.0, 'ours', 7000, 0.0, 0),
    ('cifar100 FedQSN Cross-device IID (8-bit, 2%)', 100, 0.1, 100, 1, 0.0, 'qsn', 0, 0.02, 8),
    ('cifar100 FedQSN Cross-device IID (omega=4, p1=p2=0)', 100, 0.1, 100, 1, 0.0, 'qsn', 0, 0.0, 4),
    ('cifar100 FedQSN Cross-device IID (omega=8, p1=p2=0)', 100, 0.1, 100, 1, 0.0, 'qsn', 0, 0.0, 8),
    ('cifar100 FedAvg Cross-device IID', 100, 0.1, 100, 1, 0.0, 'none', 0, 0.0, 0),
]


def result_dir(nodes, fraction, rounds, iid, beta, dp, sigma, p2, omega):
    args = SimpleNamespace(
        n_procs=2,
        nodes=nodes,
        fraction=fraction,
        round=rounds,
        dataset='cifar100',
        model='MobileNetV2',
        batch_size=64,
        iid=iid,
        beta=beta,
        local_epoch=5,
        lr=0.1,
        ft_lr=0.01,
        opt='sgd',
        lr_decay=0.999,
        DP=dp,
        sigma=sigma,
        p1=0.0,
        p2=p2,
        omega=omega,
        seed=42,
        qsn_fixed_mask=False,
        noise_update=0,
    )
    return generate_log_name(args)


def read_acc(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    matches = re.findall(r"Accuracy:\s*([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def mean_std(values):
    if not values:
        return 0.0, 0.0

    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = var ** 0.5
    return mean, std


for title, nodes, fraction, rounds, iid, beta, dp, sigma, p2, omega in experiments:
    directory = result_dir(nodes, fraction, rounds, iid, beta, dp, sigma, p2, omega)

    if not os.path.isdir(directory):
        print(f"[{title}]")
        print(f"missing directory: {directory}")
        print()
        continue

    proxy_values = []
    train_values = []
    finetune_values = []
    collusion_before_values = []
    collusion_after_values = []
    server_values = []

    for filename in os.listdir(directory):
        if filename == "dataset_analysis.txt":
            continue
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        acc = read_acc(filepath)
        if acc is None:
            continue

        if filename == "server.txt":
            server_values.append(acc)
        elif filename.startswith("proxy_train_"):
            train_values.append(acc)
        elif filename.startswith("proxy_finetune_"):
            finetune_values.append(acc)
        elif filename.startswith("proxy_"):
            proxy_values.append(acc)
        elif filename.startswith("collusion_before_"):
            collusion_before_values.append(acc)
        elif filename.startswith("collusion_after_"):
            collusion_after_values.append(acc)

    print(f"[{title}]")

    server_mean, server_std = mean_std(server_values)
    print(f"server: {(server_mean * 100):.4f}")

    proxy_mean, proxy_std = mean_std(proxy_values)
    print(f"proxy: {(proxy_mean * 100):.4f}")

    train_mean, train_std = mean_std(train_values)
    print(f"proxy train: {(train_mean * 100):.4f}")

    finetune_mean, finetune_std = mean_std(finetune_values)
    print(f"proxy finetune: {(finetune_mean * 100):.4f}")

    collusion_before_mean, collusion_before_std = mean_std(collusion_before_values)
    print(f"collusion before: {(collusion_before_mean * 100):.4f}")

    collusion_after_mean, collusion_after_std = mean_std(collusion_after_values)
    print(f"collusion after: {(collusion_after_mean * 100):.4f}")

    print()
