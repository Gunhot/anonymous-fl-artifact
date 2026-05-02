# #!/bin/bash
# # Run from src/

### 20news / Cross-device IID
# ours 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 0
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 11000 --noise_update 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 12000 --noise_update 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 13000 --noise_update 0

# ### 20news / Cross-device non IID
# # ours 8000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 8000 --noise_update 0

# ### 20news / Cross-silo IID
# # ours 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 0

# ### 20news / Cross-silo non IID
# # ours 8000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 8000 --noise_update 0

# ### e2e / Cross-device IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 0

# ### e2e / Cross-device non-IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 0

# ### e2e / Cross-silo IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 0

# ### e2e / Cross-silo non-IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 0

# #!/bin/bash
# # Run from src/

### 20news / Cross-device IID
# ours 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 11000 --noise_update 1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 12000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 13000 --noise_update 1

# ### 20news / Cross-device non IID
# # ours 8000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 8000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 11000 --noise_update 1

# ### 20news / Cross-silo IID
# # ours 10000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 11000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 12000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 13000 --noise_update 1

# ### 20news / Cross-silo non IID
# # ours 8000
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 8000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 9000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset 20news --model distilbert --nodes 20 --fraction 0.5 --round 30 --local_epoch 1 --opt adam --lr 1e-4 --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 11000 --noise_update 1

# ### e2e / Cross-device IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 1

# ### e2e / Cross-device non-IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 100 --fraction 0.1 --round 30 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 1

# ### e2e / Cross-silo IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 1 --beta 0.0 --seed 2 --DP ours --sigma 10000 --noise_update 1

# ### e2e / Cross-silo non-IID
# # ours 100%
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --n_procs 1 --dataset e2e --model gpt2s --nodes 20 --fraction 0.5 --round 10 --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4 --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000 --noise_update 1
