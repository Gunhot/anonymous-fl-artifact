
#!/bin/bash
# Run from src/

### Cross-device / IID / MobileNetV2
# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0
# # 8bit 3%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 9000 --noise_update 0
# # 80%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

# ### Cross-device / non-iid / MobileNetV2
# # FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0
# # 8bit 1%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 9000 --noise_update 0
# # 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

# ### Cross-silo / IID / MobileNetV2
# # FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0
# # 8bit 3%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 9000 --noise_update 0
# # 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

# ### Cross-silo / non-iid / MobileNetV2
# # FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0
# # 8bit 3%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 9000 --noise_update 0
# # 60%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

# ### Cross-device / IID / ResNet50
# # FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0
# # 8bit 5%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 9000 --noise_update 0
# # 80%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

# ### Cross-device / non-iid / ResNet50
# # FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0
# # 8bit 2%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 9000 --noise_update 0
# # 60%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0

# ### Cross-silo / IID / ResNet50
# # FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0
# # 8bit 5%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000 --noise_update 0
# # 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

# ### Cross-silo / non-IID / ResNet50
# # FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0
# # 8bit 3%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03 --noise_update 0
# # gausg 100%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000 --noise_update 0
# # 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
