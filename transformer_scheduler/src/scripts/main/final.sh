# #!/bin/bash
# # Run from src/

# ### 20news / Cross-device IID
# # # FedAvg
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --noise_update 0 

# # # # ours 10000
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000 --noise_update 0 

# # # # qsn 8bit masking 5%
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.005 --noise_update 0 
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.003 --noise_update 0 


# # # ### 20news / Cross-device non IID
# # # # FedAvg
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --noise_update 0 

# # # # ours 8000
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 8000 --noise_update 0 

# # # # qsn 8bit maksing 1%
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.0001 --noise_update 0 
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 0 --p2 0.002 --noise_update 0 


# # # ### 20news / Cross-silo IID
# # # # FedAvg
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --noise_update 0 

# # # # ours 10000
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000 --noise_update 0 

# # # # gausg 11000

# # # # qsn 8bit masking 5%
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.05 --noise_update 0 


# # # ### 20news / Cross-silo non IID
# # # # FedAvg
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --noise_update 0 

# # # # ours 8000
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 7000 --noise_update 0 

# # # # qsn 8bit masking 1%
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.005 --noise_update 0 


# # ### e2e / Cross-device IID
# # # FedAvg
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --noise_update 0 

# # # # ours 100%
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 0 

# # 4bit 20%
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 4 --DP qsn --p1 0.0 --omega 2 --p2 0.1 --noise_update 0 

# # ### e2e / Cross-device non-IID
# # # FedAvg
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --noise_update 0 

# # # ours 100%
# # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 0 

# # # 2bit 20%
# #CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 4 --p2 0.2 --noise_update 0 


# # ### e2e / Cross-silo IID
# # # FedAvg
# #CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --noise_update 0 

# # # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 11000 --noise_update 0 
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 0 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 9000 --noise_update 0 
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 8000 --noise_update 0 

# # # 2bit 20%
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 2 --p2 0.2 --noise_update 0 

# # ### e2e / Cross-silo non-IID
# # # FedAvg
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --noise_update 0 

# # # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 0 

# # # 2bit 20%
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 2 --p2 0.2 --noise_update 0 


### qsn final runs

# ### 20news / Cross-device IID
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 4 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.0 --noise_update 0

# ### 20news / Cross-device non IID
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 4 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.0 --noise_update 0

# ### 20news / Cross-silo IID
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 4 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.0 --noise_update 0

# ### 20news / Cross-silo non IID
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 4 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 8 --p2 0.0 --noise_update 0

# e2e / Cross-device IID
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 2 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 4 --p2 0.0 --noise_update 0

# ### e2e / Cross-device non-IID
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 2 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2,3 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 4 --p2 0.0 --noise_update 0

### e2e / Cross-silo IID
# CUDA_VISIBLE_DEVICES=0,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 1 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP qsn --p1 0.0 --omega 2 --p2 0.0 --noise_update 0

# ### e2e / Cross-silo non-IID
# CUDA_VISIBLE_DEVICES=0,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 1 --p2 0.0 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,2 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP qsn --p1 0.0 --omega 2 --p2 0.0 --noise_update 0
