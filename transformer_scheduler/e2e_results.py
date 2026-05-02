import os
import re
import sys
from types import SimpleNamespace


TRANSFORMER_SCHEDULER_DIR = os.path.dirname(__file__)
TRANSFORMER_SRC_DIR = os.path.join(TRANSFORMER_SCHEDULER_DIR, "src")
sys.path.insert(0, TRANSFORMER_SRC_DIR)

from arguments import generate_log_name


experiments = [
    # title, nodes, fraction, round, iid, beta, DP, sigma, p2, omega, seed
    # Cross-device / IID
    ('gpt2s Cross-device iid FedAvg', 100, 0.1, 30, 1, 0.0, 'none', 0, 0.0, 0, 2),
    ('gpt2s Cross-device iid CloakFL rho=1.0', 100, 0.1, 30, 1, 0.0, 'ours', 10000.0, 0.0, 0, 2),
    ('gpt2s Cross-device iid QSN 4bit 20%', 100, 0.1, 30, 1, 0.0, 'qsn', 0, 0.2, 4, 4),
    ('gpt2s Cross-device iid QSN 4bit 0%', 100, 0.1, 30, 1, 0.0, 'qsn', 0, 0.0, 4, 2),
    ('gpt2s Cross-device iid QSN 2bit 0%', 100, 0.1, 30, 1, 0.0, 'qsn', 0, 0.0, 2, 2),

    # Cross-device / non-IID
    ('gpt2s Cross-device non-iid FedAvg', 100, 0.1, 30, 2, 0.1, 'none', 0, 0.0, 0, 2),
    ('gpt2s Cross-device non-iid CloakFL rho=1.0', 100, 0.1, 30, 2, 0.1, 'ours', 10000.0, 0.0, 0, 2),
    ('gpt2s Cross-device non-iid QSN 4bit 20%', 100, 0.1, 30, 2, 0.1, 'qsn', 0, 0.2, 4, 2),
    ('gpt2s Cross-device non-iid QSN 4bit 0%', 100, 0.1, 30, 2, 0.1, 'qsn', 0, 0.0, 4, 2),
    ('gpt2s Cross-device non-iid QSN 2bit 0%', 100, 0.1, 30, 2, 0.1, 'qsn', 0, 0.0, 2, 2),

    # Cross-silo / IID
    ('gpt2s Cross-silo iid FedAvg', 20, 0.5, 10, 1, 0.0, 'none', 0, 0.0, 0, 2),
    ('gpt2s Cross-silo iid CloakFL rho=0.9', 20, 0.5, 10, 1, 0.0, 'ours', 9000.0, 0.0, 0, 2),
    ('gpt2s Cross-silo iid QSN 2bit 20%', 20, 0.5, 10, 1, 0.0, 'qsn', 0, 0.2, 2, 2),
    ('gpt2s Cross-silo iid QSN 1bit 0%', 20, 0.5, 10, 1, 0.0, 'qsn', 0, 0.0, 1, 2),
    ('gpt2s Cross-silo iid QSN 2bit 0%', 20, 0.5, 10, 1, 0.0, 'qsn', 0, 0.0, 2, 2),

    # Cross-silo / non-IID
    ('gpt2s Cross-silo non-iid FedAvg', 20, 0.5, 10, 2, 0.1, 'none', 0, 0.0, 0, 2),
    ('gpt2s Cross-silo non-iid CloakFL rho=1.0', 20, 0.5, 10, 2, 0.1, 'ours', 10000.0, 0.0, 0, 2),
    ('gpt2s Cross-silo non-iid QSN 2bit 20%', 20, 0.5, 10, 2, 0.1, 'qsn', 0, 0.2, 2, 2),
    ('gpt2s Cross-silo non-iid QSN 1bit 0%', 20, 0.5, 10, 2, 0.1, 'qsn', 0, 0.0, 1, 2),
    ('gpt2s Cross-silo non-iid QSN 2bit 0%', 20, 0.5, 10, 2, 0.1, 'qsn', 0, 0.0, 2, 2),
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


def result_dir(nodes, fraction, rounds, iid, beta, dp, sigma, p2, omega, seed):
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
        seed=seed,
        max_len=512,
        model='gpt2s',
        dataset='e2e',
        batch_size=4,
        iid=iid,
        beta=beta,
        local_epoch=1,
        lr=0.001,
        lr_decay=0.999,
        opt='adam',
    )
    return generate_log_name(args)


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
    std = var ** 0.5
    return mean, std


for title, nodes, fraction, rounds, iid, beta, dp, sigma, p2, omega, seed in experiments:
    directory = result_dir(nodes, fraction, rounds, iid, beta, dp, sigma, p2, omega, seed)

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
        filepath = os.path.join(directory, filename)

        if not os.path.isfile(filepath):
            continue
        if not filename.endswith(".txt"):
            continue

        bleu = read_bleu(filepath)
        if bleu is None:
            continue

        if filename == "server.txt":
            server_values.append(bleu)
        elif re.fullmatch(r"proxy_train_\d+\.txt", filename):
            train_values.append(bleu)
        elif re.fullmatch(r"proxy_finetune_\d+\.txt", filename):
            finetune_values.append(bleu)
        elif re.fullmatch(r"proxy_\d+\.txt", filename):
            proxy_values.append(bleu)
        elif re.fullmatch(r"collusion_before_\d+\.txt", filename):
            collusion_before_values.append(bleu)
        elif re.fullmatch(r"collusion_after_\d+\.txt", filename):
            collusion_after_values.append(bleu)

    print(f"[{title}]")

    mean, std = mean_std(server_values)
    print(f"server: {mean:.4f}")

    mean, std = mean_std(proxy_values)
    print(f"proxy: {mean:.4f}")

    mean, std = mean_std(train_values)
    print(f"train: {mean:.4f}")

    mean, std = mean_std(finetune_values)
    print(f"finetune: {mean:.4f}")

    mean, std = mean_std(collusion_before_values)
    print(f"collusion_before: {mean:.4f}")

    mean, std = mean_std(collusion_after_values)
    print(f"collusion_after: {mean:.4f}")

    print()
