#!/bin/bash
# Run from src/

### Cross-device / IID / MobileNetV2
# 80%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000 --noise_update 0

### Cross-device / non-iid / MobileNetV2
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 0

### Cross-silo / IID / MobileNetV2
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

### Cross-silo / non-iid / MobileNetV2
# 60%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

### Cross-device / IID / ResNet50
# 80%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000 --noise_update 0

### Cross-device / non-iid / ResNet50
# 60%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

### Cross-silo / IID / ResNet50
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 0

### Cross-silo / non-IID / ResNet50
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 0

#!/bin/bash
# Run from src/

### Cross-device / IID / MobileNetV2
# 80%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000 --noise_update 1

### Cross-device / non-iid / MobileNetV2
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 1

### Cross-silo / IID / MobileNetV2
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 1

### Cross-silo / non-iid / MobileNetV2
# 60%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 1

### Cross-device / IID / ResNet50
# 80%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 8000 --noise_update 1

### Cross-device / non-iid / ResNet50
# 60%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 1

### Cross-silo / IID / ResNet50
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --DP ours --sigma 7000 --noise_update 1

### Cross-silo / non-IID / ResNet50
# 70%
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 1
