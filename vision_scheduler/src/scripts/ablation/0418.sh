# # Centering Preservers convergence
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

# # Magnitude Calibaration
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 4000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0


# # Stats
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0

# # Source of Perturbation
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 6000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP gausg --sigma 10000 --noise_update 0


# # magnitude calibration
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 7000 --noise_update 0
# CUDA_VISIBLE_DEVICES="0,1,2" python main.py --n_procs 2 --dataset cifar100 --model MobileNetV2 --nodes 100 --fraction 0.1 --round 100 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 5000 --noise_update 0


CUDA_VISIBLE_DEVICES="0,2,3" python main.py --n_procs 2 --dataset tiny-imagenet --model ResNet50 --nodes 100 --fraction 0.1 --round 150 --local_epoch 5 --opt sgd --lr 0.1 --ft_lr 0.01 --batch_size 64 --iid 2 --beta 0.1 --seed 42 --DP ours --sigma 0 --noise_update 0
