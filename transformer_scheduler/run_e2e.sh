#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=0

cd "$SCRIPT_DIR/src"

run_exp() {
    echo
    echo "### $1"
    shift
    "$PYTHON" main.py "$@"
}

COMMON_ARGS="--n_procs 1 --dataset e2e --model gpt2s --batch_size 4 --local_epoch 1 --opt adam --lr 0.001 --lr_decay 0.999 --max_len 512 --use_pretrained 1 --seed 2 --noise_update 0"

# Cross-device / IID
run_exp "gpt2s Cross-device iid FedAvg" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 1 --beta 0.0 \
    --DP none

run_exp "gpt2s Cross-device iid CloakFL rho=1.0" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 1 --beta 0.0 \
    --DP ours --sigma 10000

run_exp "gpt2s Cross-device iid QSN 4bit 20%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.2 --omega 4

run_exp "gpt2s Cross-device iid QSN 4bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "gpt2s Cross-device iid QSN 2bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 2

# Cross-device / non-IID
run_exp "gpt2s Cross-device non-iid FedAvg" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 2 --beta 0.1 \
    --DP none

run_exp "gpt2s Cross-device non-iid CloakFL rho=1.0" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 2 --beta 0.1 \
    --DP ours --sigma 10000

run_exp "gpt2s Cross-device non-iid QSN 4bit 20%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.2 --omega 4

run_exp "gpt2s Cross-device non-iid QSN 4bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "gpt2s Cross-device non-iid QSN 2bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 30 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 2

# Cross-silo / IID
run_exp "gpt2s Cross-silo iid FedAvg" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 1 --beta 0.0 \
    --DP none

run_exp "gpt2s Cross-silo iid CloakFL rho=0.9" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 1 --beta 0.0 \
    --DP ours --sigma 9000

run_exp "gpt2s Cross-silo iid QSN 2bit 20%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.2 --omega 2

run_exp "gpt2s Cross-silo iid QSN 1bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 1

run_exp "gpt2s Cross-silo iid QSN 2bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 2

# Cross-silo / non-IID
run_exp "gpt2s Cross-silo non-iid FedAvg" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 2 --beta 0.1 \
    --DP none

run_exp "gpt2s Cross-silo non-iid CloakFL rho=1.0" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 2 --beta 0.1 \
    --DP ours --sigma 10000

run_exp "gpt2s Cross-silo non-iid QSN 2bit 20%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.2 --omega 2

run_exp "gpt2s Cross-silo non-iid QSN 1bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 1

run_exp "gpt2s Cross-silo non-iid QSN 2bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 10 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 2
