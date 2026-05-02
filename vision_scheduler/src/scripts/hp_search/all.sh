#!/bin/bash
# Run from src/

# Cross-device
# iid

# tiny-imagenet / ResNet50
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 8000

# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 6000

# cifar100 / MobileNetV2
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 15000

# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 6000



#!/bin/bash
# Run from src/

# Cross-device
# non-iid beta 0.1

# tiny-imagenet / ResNet50
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 8000

# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000

# cifar100 / MobileNetV2
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000

# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000




#!/bin/bash
# Run from src/

# Cross-silo
# iid

# tiny-imagenet / ResNet50
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000

# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 6000

# cifar100 / MobileNetV2
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP gausg --sigma 10000

# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 6000






#!/bin/bash
# Run from src/

# Cross-silo
# non-iid beta 0.1

# tiny-imagenet / ResNet50
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000
# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 3000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000

# cifar100 / MobileNetV2
# baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42

# lpp
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP lpp --omega 4

# qsn
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.05
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.1
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.2

# gausg
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 6000
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 8000

# ours
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 3000
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 8000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000
