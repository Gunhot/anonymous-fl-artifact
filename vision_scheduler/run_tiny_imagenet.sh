#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=0

cd "$SCRIPT_DIR/src"

run_exp() {
    title=$1
    shift

    echo
    echo "### $title"
    "$PYTHON" main.py $COMMON_ARGS "$@"
}

COMMON_ARGS="--n_procs 2 --dataset tiny-imagenet --model ResNet50 --batch_size 64 --local_epoch 5 --lr 0.1 --ft_lr 0.01 --opt sgd --lr_decay 0.999 --seed 42 --noise_update 0"

run_exp "tiny-imagenet FedAvg Cross-device Non-IID" \
    --nodes 100 --fraction 0.1 --round 150 --iid 2 --beta 0.1 \
    --DP none --sigma 0

run_exp "tiny-imagenet CloakFL Cross-device Non-IID rho=0.5" \
    --nodes 100 --fraction 0.1 --round 150 --iid 2 --beta 0.1 \
    --DP ours --sigma 5000
