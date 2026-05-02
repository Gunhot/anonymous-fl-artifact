#### Step Decay

# # Cross-silo / non-IID / ResNet50
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --lr_decay 1.0 --DP ours --sigma 5000 --noise_update 0

# # # Cross-silo / IID / ResNet50
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 20 --fraction 0.5 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --lr_decay 1.0 --DP ours --sigma 7000 --noise_update 0

# Cross-silo / non-iid / MobileNetV2
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --lr_decay 1.0 --DP ours --sigma 6000 --noise_update 0

# Cross-silo / IID / MobileNetV2
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 20 --fraction 0.5 --round 30 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --lr_decay 1.0 --DP ours --sigma 7000 --noise_update 0


# # Cross-device / IID / MobileNetV2
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --lr_decay 1.0 --DP ours --sigma 6000 --noise_update 0

# # Cross-device / NoN-IID / MobileNetV2
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --lr_decay 1.0  --DP ours --sigma 7000 --noise_update 0


# # Cross-silo / non-IID / ResNet50
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --lr_decay 1.0  --lr_decay 1.0 --DP ours --sigma 5000 --noise_update 0

# # # Cross-silo / IID / ResNet50
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 1 --beta 0.0 --seed 42 --lr_decay 1.0  --lr_decay 1.0 --DP ours --sigma 7000 --noise_update 0



##### Noise Norm


# FedAvg
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --noise_update 0

# Ours center pL+ / sigma 6000 / target 70%
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
# Stats run
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0



# # ### Cross-device / non-iid / MobileNetV2

# # Quantization
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 2 --p1 0.0 --p2 0.00 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 4 --p1 0.0 --p2 0.00 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.00 --noise_update 0

# # 8bit 1%
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.04 --noise_update 0

# # CloakFL
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 4000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 8000 --noise_update 0

# # Gaussian
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 5000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 9000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 11000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 15000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 50 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 20000 --noise_update 0


# # ### Cross-device / non-iid / MobileNetV2

# Quantization
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 2 --p1 0.0 --p2 0.00 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 4 --p1 0.0 --p2 0.00 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.00 --noise_update 0

# QSN
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.005 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03 --noise_update 0

# CloakFL
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 4000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 0


# Cross-silo / non-IID / ResNet50
# Quantization
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 2 --p1 0.0 --p2 0.0 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 4 --p1 0.0 --p2 0.0 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.0 --noise_update 0

# FedQSN
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.005 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.01  --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.02  --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP qsn --omega 8 --p1 0.0 --p2 0.03  --noise_update 0

# CloakFL
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 4000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
CUDA_VISIBLE_DEVICES="0,1,2,3,4" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 0
