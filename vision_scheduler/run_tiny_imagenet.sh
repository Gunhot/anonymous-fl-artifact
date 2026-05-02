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

COMMON_ARGS="--n_procs 2 --dataset tiny-imagenet --model ResNet50 --batch_size 64 --local_epoch 5 --lr 0.1 --ft_lr 0.01 --opt sgd --lr_decay 0.999 --seed 42 --noise_update 0"

# Cross-silo / Non-IID
run_exp "tiny-imagenet CloakFL Cross-silo Non-IID rho=0.5" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 2 --beta 0.1 \
    --DP ours --sigma 5000

run_exp "tiny-imagenet FedQSN Cross-silo Non-IID 8bit 2%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.02 --omega 8

run_exp "tiny-imagenet FedQSN Cross-silo Non-IID omega=4 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "tiny-imagenet FedQSN Cross-silo Non-IID omega=8 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8

run_exp "tiny-imagenet FedAvg Cross-silo Non-IID" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 2 --beta 0.1 \
    --DP none

# Cross-silo / IID
run_exp "tiny-imagenet CloakFL Cross-silo IID rho=0.7" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 1 --beta 0.0 \
    --DP ours --sigma 7000

run_exp "tiny-imagenet FedQSN Cross-silo IID 8bit 5%" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.05 --omega 8

run_exp "tiny-imagenet FedQSN Cross-silo IID omega=4 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "tiny-imagenet FedQSN Cross-silo IID omega=8 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8

run_exp "tiny-imagenet FedAvg Cross-silo IID" \
    $COMMON_ARGS \
    --nodes 20 --fraction 0.5 --round 50 --iid 1 --beta 0.0 \
    --DP none

# Cross-device / Non-IID
run_exp "tiny-imagenet CloakFL Cross-device Non-IID rho=0.5" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 2 --beta 0.1 \
    --DP ours --sigma 5000

run_exp "tiny-imagenet FedQSN Cross-device Non-IID 8bit 1%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.01 --omega 8

run_exp "tiny-imagenet FedQSN Cross-device Non-IID omega=4 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "tiny-imagenet FedQSN Cross-device Non-IID omega=8 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 2 --beta 0.1 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8

run_exp "tiny-imagenet FedAvg Cross-device Non-IID" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 2 --beta 0.1 \
    --DP none

# Cross-device / IID
run_exp "tiny-imagenet CloakFL Cross-device IID rho=0.7" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 1 --beta 0.0 \
    --DP ours --sigma 7000

run_exp "tiny-imagenet FedQSN Cross-device IID 8bit 3%" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.03 --omega 8

run_exp "tiny-imagenet FedQSN Cross-device IID omega=4 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 4

run_exp "tiny-imagenet FedQSN Cross-device IID omega=8 p1=p2=0" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 1 --beta 0.0 \
    --DP qsn --p1 0.0 --p2 0.0 --omega 8

run_exp "tiny-imagenet FedAvg Cross-device IID" \
    $COMMON_ARGS \
    --nodes 100 --fraction 0.1 --round 150 --iid 1 --beta 0.0 \
    --DP none
