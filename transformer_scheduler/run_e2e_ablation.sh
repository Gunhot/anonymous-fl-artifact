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

COMMON_ARGS="--eval_mode ablation --dataset e2e --model gpt2s --batch_size 4 --local_epoch 1 --opt adam --lr 0.001 --lr_decay 0.999 --max_len 512 --use_pretrained 1 --seed 2 --noise_update 0"

run_exp "e2e FedAvg Cross-device Non-IID" \
    --n_procs 1 --nodes 100 --fraction 0.1 --round 30 --iid 2 --beta 0.1 \
    --DP none --sigma 0

run_exp "e2e CloakFL Cross-device Non-IID rho=1.0" \
    --n_procs 2 --nodes 100 --fraction 0.1 --round 30 --iid 2 --beta 0.1 \
    --DP ours --sigma 10000
