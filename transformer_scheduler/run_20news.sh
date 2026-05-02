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

COMMON_ARGS="--n_procs 1 --dataset 20news --model distilbert --batch_size 16 --local_epoch 1 --opt adam --lr 1e-4 --lr_decay 0.999 --max_len 256 --use_pretrained 1 --seed 2 --noise_update 0"

# Cross-device / IID
run_exp "distilbert Cross-device iid FedAvg" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 1 --beta 0.0 \
    --DP none

run_exp "distilbert Cross-device iid CloakFL rho=0.9" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 1 --beta 0.0 \
    --DP ours --sigma 9000

run_exp "distilbert Cross-device iid QSN 8bit 0.5%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.005 --omega 8

run_exp "distilbert Cross-device iid QSN 4bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "distilbert Cross-device iid QSN 8bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8

# Cross-device / non-IID
run_exp "distilbert Cross-device non-iid FedAvg" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP none

run_exp "distilbert Cross-device non-iid CloakFL rho=0.8" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP ours --sigma 8000

run_exp "distilbert Cross-device non-iid QSN 8bit 0.01%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0001 --omega 8

run_exp "distilbert Cross-device non-iid QSN 4bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "distilbert Cross-device non-iid QSN 8bit 0%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8

# Cross-silo / IID
run_exp "distilbert Cross-silo iid FedAvg" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 1 --beta 0.0 \
    --DP none

run_exp "distilbert Cross-silo iid CloakFL rho=0.9" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 1 --beta 0.0 \
    --DP ours --sigma 9000

run_exp "distilbert Cross-silo iid QSN 8bit 5%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.05 --omega 8

run_exp "distilbert Cross-silo iid QSN 4bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "distilbert Cross-silo iid QSN 8bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8

# Cross-silo / non-IID
run_exp "distilbert Cross-silo non-iid FedAvg" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 2 --beta 0.1 \
    --DP none

run_exp "distilbert Cross-silo non-iid CloakFL rho=0.7" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 2 --beta 0.1 \
    --DP ours --sigma 7000

run_exp "distilbert Cross-silo non-iid QSN 8bit 0.5%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.005 --omega 8

run_exp "distilbert Cross-silo non-iid QSN 4bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "distilbert Cross-silo non-iid QSN 8bit 0%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 30 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8
