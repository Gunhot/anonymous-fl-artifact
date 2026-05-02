#!/bin/bash
# set -e
# Run from src/

# Cross-device
# iid

# 20news
# baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 12000



# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP lpp --omega 4

# # qsn
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# # gausg
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 20000

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 12000

# # # e2e
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP lpp --omega 4

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 20000

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 11000



#!/bin/bash
# Run from src/

# Cross-device
# non-iid beta 0.1

# 20news
# baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP lpp --omega 4

# # qsn
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 20000

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 12000

# e2e
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP lpp --omega 4

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 20000

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 11000




#!/bin/bash
# Run from src/

# Cross-silo
# iid

# e2e
# baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2

# lpp
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 20000

# ours
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 11000

# 20news
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP lpp --omega 4

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP gausg --sigma 20000

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 12000



#!/bin/bash
# Run from src/

# Cross-silo
# non-iid beta 0.1

# 20news
# baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 20000

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 30 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 12000

# e2e
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP lpp --omega 4

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --omega 0 --p1 0.0 --p2 0.2

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 15000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP gausg --sigma 20000

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 30 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 11000
