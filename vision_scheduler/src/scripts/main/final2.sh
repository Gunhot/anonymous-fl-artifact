#!/bin/bash
# Run from src/
#
# Grouping guide
# - Cross-device: nodes=100, fraction=0.1
# - Cross-silo:   nodes=20,  fraction=0.5
# - IID:     --iid 1 --beta 0.0
# - non-IID: --iid 2 --beta 0.1
# - ACTIVE blocks are intentionally left uncommented.

################################################################################
# Tiny-ImageNet / ResNet50
################################################################################

### Cross-device / IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0

# Ours center pL+ / sigma 7000 / target 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

### Cross-device / non-IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,2,3" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --lr_decay 1.0 --noise_update 0

# Ours center pL+ / sigma 5000 / target 60%
# CUDA_VISIBLE_DEVICES="0,2,3" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --lr_decay 1.0 --DP ours --sigma 5000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,2,3" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --lr_decay 1.0 --DP ours --sigma 5000 --noise_update 0

# ACTIVE: QSN ablation / per-client mask vs fixed shared client mask
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --qsn_fixed_mask --noise_update 0

### Cross-silo / IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0

# Ours center pL+ / sigma 7000 / target 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0
# Stats run
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

### Cross-silo / non-IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,2,3" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0

# Ours center pL+ / sigma 5000
# CUDA_VISIBLE_DEVICES="0,2,3" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,2,3" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0

################################################################################
# CIFAR100 / MobileNetV2
################################################################################

### Cross-device / IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0

# Ours center pL+ / sigma 7000 / target 80%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0
# Stats run
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

### Cross-device / non-IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0

# Ours center pL+ / sigma 6000 / target 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
# Stats run
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

# ACTIVE: QSN ablation / per-client mask vs fixed shared client mask
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --qsn_fixed_mask --noise_update 0

### Cross-silo / IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --noise_update 0

# Ours center pL+ / sigma 7000 / target 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0
# Stats run
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

### Cross-silo / non-IID
# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0

# Ours center pL+ / sigma 6000 / target 60%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
# Stats run
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
