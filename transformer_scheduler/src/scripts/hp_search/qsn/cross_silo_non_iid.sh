#!/bin/bash
# set -e
# Run from src/

# 20news
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# e2e
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2
