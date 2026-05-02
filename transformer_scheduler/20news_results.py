import os
import re
import sys
from types import SimpleNamespace


TRANSFORMER_SCHEDULER_DIR = os.path.dirname(__file__)
TRANSFORMER_SRC_DIR = os.path.join(TRANSFORMER_SCHEDULER_DIR, "src")
sys.path.insert(0, TRANSFORMER_SRC_DIR)

from arguments import generate_log_name


experiments = [
    # title, nodes, fraction, round, iid, beta, DP, sigma, p2, omega
    # Cross-device / IID
    ('distilbert Cross-device iid FedAvg', 100, 0.1, 100, 1, 0.0, 'none', 0, 0.0, 0),
    ('distilbert Cross-device iid CloakFL rho=0.9', 100, 0.1, 100, 1, 0.0, 'ours', 9000.0, 0.0, 0),
    ('distilbert Cross-device iid QSN 8bit 0.5%', 100, 0.1, 100, 1, 0.0, 'qsn', 0, 0.005, 8),
    ('distilbert Cross-device iid QSN 4bit 0%', 100, 0.1, 100, 1, 0.0, 'qsn', 0, 0.0, 4),
    ('distilbert Cross-device iid QSN 8bit 0%', 100, 0.1, 100, 1, 0.0, 'qsn', 0, 0.0, 8),

    # Cross-device / non-IID
    ('distilbert Cross-device non-iid FedAvg', 100, 0.1, 100, 2, 0.1, 'none', 0, 0.0, 0),
    ('distilbert Cross-device non-iid CloakFL rho=0.8', 100, 0.1, 100, 2, 0.1, 'ours', 8000.0, 0.0, 0),
    ('distilbert Cross-device non-iid QSN 8bit 0.01%', 100, 0.1, 100, 2, 0.1, 'qsn', 0, 0.0001, 8),
    ('distilbert Cross-device non-iid QSN 4bit 0%', 100, 0.1, 100, 2, 0.1, 'qsn', 0, 0.0, 4),
    ('distilbert Cross-device non-iid QSN 8bit 0%', 100, 0.1, 100, 2, 0.1, 'qsn', 0, 0.0, 8),

    # Cross-silo / IID
    ('distilbert Cross-silo iid FedAvg', 20, 0.5, 30, 1, 0.0, 'none', 0, 0.0, 0),
    ('distilbert Cross-silo iid CloakFL rho=0.9', 20, 0.5, 30, 1, 0.0, 'ours', 9000.0, 0.0, 0),
    ('distilbert Cross-silo iid QSN 8bit 5%', 20, 0.5, 30, 1, 0.0, 'qsn', 0, 0.05, 8),
    ('distilbert Cross-silo iid QSN 4bit 0%', 20, 0.5, 30, 1, 0.0, 'qsn', 0, 0.0, 4),
    ('distilbert Cross-silo iid QSN 8bit 0%', 20, 0.5, 30, 1, 0.0, 'qsn', 0, 0.0, 8),

    # Cross-silo / non-IID
    ('distilbert Cross-silo non-iid FedAvg', 20, 0.5, 30, 2, 0.1, 'none', 0, 0.0, 0),
    ('distilbert Cross-silo non-iid CloakFL rho=0.7', 20, 0.5, 30, 2, 0.1, 'ours', 7000.0, 0.0, 0),
    ('distilbert Cross-silo non-iid QSN 8bit 0.5%', 20, 0.5, 30, 2, 0.1, 'qsn', 0, 0.005, 8),
    ('distilbert Cross-silo non-iid QSN 4bit 0%', 20, 0.5, 30, 2, 0.1, 'qsn', 0, 0.0, 4),
    ('distilbert Cross-silo non-iid QSN 8bit 0%', 20, 0.5, 30, 2, 0.1, 'qsn', 0, 0.0, 8),
]


def result_dir(nodes, fraction, rounds, iid, beta, dp, sigma, p2, omega):
    args = SimpleNamespace(
        n_procs=1,
        nodes=nodes,
        fraction=fraction,
        round=rounds,
        ft_subset=1.0,
        DP=dp,
        sigma=sigma,
        noise_update=0,
        p1=0.0,
        p2=p2,
        omega=omega,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_pretrained=1,
        seed=2,
        max_len=256,
        model='distilbert',
        dataset='20news',
        batch_size=16,
        iid=iid,
        beta=beta,
        local_epoch=1,
        lr=0.0001,
        lr_decay=0.999,
        opt='adam',
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
        elif re.fullmatch(r"proxy_train_\d+\.txt", filename):
            train_values.append(acc)
        elif re.fullmatch(r"proxy_finetune_\d+\.txt", filename):
            finetune_values.append(acc)
        elif re.fullmatch(r"proxy_\d+\.txt", filename):
            proxy_values.append(acc)
        elif re.fullmatch(r"collusion_before_\d+\.txt", filename):
            collusion_before_values.append(acc)
        elif re.fullmatch(r"collusion_after_\d+\.txt", filename):
            collusion_after_values.append(acc)

    print(f"[{title}]")

    mean, std = mean_std(server_values)
    print(f"server: {(mean * 100):.4f}")

    mean, std = mean_std(proxy_values)
    print(f"proxy: {(mean * 100):.4f}")

    mean, std = mean_std(train_values)
    print(f"train: {(mean * 100):.4f}")

    mean, std = mean_std(finetune_values)
    print(f"finetune: {(mean * 100):.4f}")

    mean, std = mean_std(collusion_before_values)
    print(f"collusion_before: {(mean * 100):.4f}")

    mean, std = mean_std(collusion_after_values)
    print(f"collusion_after: {(mean * 100):.4f}")

    print()
