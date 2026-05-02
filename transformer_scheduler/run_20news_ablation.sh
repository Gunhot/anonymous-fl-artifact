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

COMMON_ARGS="--eval_mode ablation --dataset 20news --model distilbert --batch_size 16 --local_epoch 1 --opt adam --lr 0.0001 --lr_decay 0.999 --max_len 256 --use_pretrained 1 --seed 2 --noise_update 0"

run_exp "20news FedAvg Cross-device Non-IID" \
    --n_procs 2 --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP none --sigma 0

run_exp "20news Gaussian Cross-device Non-IID rho=1.3" \
    --n_procs 1 --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP gausg --sigma 13000

run_exp "20news CloakFL center Cross-device Non-IID rho=0.7" \
    --n_procs 2 --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP ours --sigma 7000

run_exp "20news CloakFL center Cross-device Non-IID rho=0.8" \
    --n_procs 2 --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP ours --sigma 8000

run_exp "20news CloakFL center Cross-device Non-IID rho=0.9" \
    --n_procs 2 --nodes 100 --fraction 0.1 --round 100 --iid 2 --beta 0.1 \
    --DP ours --sigma 9000
