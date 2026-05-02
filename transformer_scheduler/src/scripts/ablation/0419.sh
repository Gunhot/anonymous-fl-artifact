# source of perturbation
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 2 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4  --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP gausg --noise_update 0 --sigma 8000


# magnitude calibration
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 2 --dataset 20news --model distilbert --nodes 100 --fraction 0.1 --round 100 --local_epoch 1 --opt adam --lr 1e-4  --max_len 256 --batch_size 16 --iid 2 --beta 0.1 --seed 2 --DP ours --noise_update 0 --sigma 7000


# Reference Model for Update Computation
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 2 --dataset e2e    --model gpt2s      --nodes 100 --fraction 0.1 --round 30  --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4  --iid 2 --beta 0.1 --seed 2 
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 2 --dataset e2e    --model gpt2s      --nodes 100 --fraction 0.1 --round 30  --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4  --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --n_procs 2 --dataset e2e    --model gpt2s      --nodes 100 --fraction 0.1 --round 30  --local_epoch 1 --opt adam --lr 0.001 --max_len 512 --batch_size 4  --iid 2 --beta 0.1 --seed 2 --DP ours --sigma 10000
