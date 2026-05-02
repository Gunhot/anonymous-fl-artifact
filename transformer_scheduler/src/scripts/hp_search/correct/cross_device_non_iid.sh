#!/bin/bash
# set -e
# Run from src/

# 20news
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 12000

# # e2e
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 11000


# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000
