import argparse
import os


def generate_log_name(args):
    log_name = (
        # 1. System Configuration
        f"NP{args.n_procs}"
        # 2. Federated Learning Configuration
        f"N{args.nodes}FR{args.fraction}R{args.round}"
        # 3. Dataset Configuration
        f"DS{args.dataset}MD{args.model}BS{args.batch_size}I{args.iid}BT{args.beta}"
        # 4. Learning Configuration
        f"LE{args.local_epoch}LR{args.lr}FT{args.ft_lr}OP{args.opt}DC{args.lr_decay}STEP{args.step_decay}EP{args.step_epoch}"
        # 5. Free Rider
        f"DP{args.DP}DI{args.DP_i}DN{args.DP_n}DS{args.DP_s}SG{args.sigma}.ST{args.stats}"
        f"BB{args.bank_beta}p1{args.p1}p2{args.p2}OM{args.omega}SD{args.seed}"
        f"{'QF1' if args.qsn_fixed_mask else ''}"
        f"NU{args.noise_update}"
        f"FINAL{int(args.final)}"
        f"NORM{int(args.norm)}"
    )
    log_path = os.path.join("..", "save", args.dataset, log_name)
    return log_path

def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    # Free Rider
    parser.add_argument('--DP', type=str, default='none',
                        help='distortion type: none=None, ours=ClientDrift(use DP_n/DP_s), gausg=Gaussian(layer grad norm), quan=Quantization, prev=PrevGrad, lapl=Laplace')
    parser.add_argument('--sigma', type=int, default=0,
                        help='sigma ratio in percent (>0: noise). *g: layer grad-norm ratio, *G: global average-matching ratio')
    parser.add_argument('--stats', type=int, default=0,
                        help='sigma for BN running stats (>0: noise)')
    parser.add_argument('--noise_update', type=int, default=0, choices=[0, 1],
                        help='If 1, subtract the noise added in get_model() before update_node_info bookkeeping')
    parser.add_argument('--DP_i', type=str, default='my',
                        choices=['wrong', 'random', 'my', 'correct', 'center', 'center_0', 'center_m', 'center_M', 'global'],
                        help='Node index strategy for ours: random=Use any random ID with available proxy_gradient, wrong=Use (ID+1) mod N, my=Use own ID, correct=center alias, center=Mean-centered sampled proxy_gradient, center_0=Reuse one centered direction for all selected nodes, center_m=Reuse the least aligned centered direction, center_M=Reuse the most aligned centered direction, global=Use server global proxy_gradient bank.')
    parser.add_argument('--DP_n', type=str, default='p',
                        help='Base model for scale update in update_node_info: c=clean server model, p=proxy model.')
    parser.add_argument('--DP_s', type=str, default='p',
                        help='Proxy-gradient bank update rule: include c/p for base model, +/- for sign, and optional L/M/E for overwrite/momentum/EMA.')
    parser.add_argument('--bank_beta', type=float, default=0.9,
                        help='EMA beta for ours bank update')
    
    # FedQSN Specific
    parser.add_argument('--p1', type=float, default=0.0, help='Server-side mask probability')
    parser.add_argument('--p2', type=float, default=0.0, help='Client-side mask probability')
    parser.add_argument('--omega', type=int, default=0, help='Quantization bit-width')
    parser.add_argument('--qsn_fixed_mask', action='store_true',
                        help='For DP=qsn, reuse one client-side mask across all clients and rounds so every client receives the same masked proxy model.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed used across main/client/server/data sampling')
    
    # Learning Configuration
    parser.add_argument('--local_epoch',  type=int, default=5,
                        help='number of local_epoch')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--ft_lr', type=float, default=0.01,
                        help='finetuning learning rate')
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='local optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='0.992, 0.998')    
    parser.add_argument('--step_decay', type=int, default=0,
                        help='Round index for one-time LR drop by 10x. 0 disables step decay.')
    parser.add_argument('--step_epoch', type=int, default=3,
                        help='Round index for one-time LR drop by 10x. 0 disables step decay.')
    
    # Dataset Configuration                
    parser.add_argument('--dataset',  type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet'],
                        help='type of dataset')
    parser.add_argument('--model', type=str,
                        help='model backbone: ResNet20, ResNet18, ResNet50, MobileNetV2')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='size of batch')
    parser.add_argument('--iid', type=int, default=0,
                        help='iid')
    parser.add_argument('--beta', type=float, default= 0.1,
                        help='beta for non iid dirichlet dist')

    # Federated Learning Configuration
    parser.add_argument('--nodes', type=int, default=100,
                        help='total number of nodes')
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='ratio of participating node')
    parser.add_argument('--round', type=int, default=500,
                        help='number of rounds')    

    # System Configuration
    parser.add_argument('--n_procs', type=int, default=5,
                        help='number of processes per GPU')

    # Experiment bookkeeping
    parser.add_argument('--log_name', type=str, default=None,
                        help='Optional override for log directory path')
    parser.add_argument('--final', action='store_true',
                        help='Evaluate only the final round with final-round proxy/collusion/finetuning flow')
    parser.add_argument('--norm', action='store_true',
                        help='Run clean FedAvg for all rounds, then create final-round proxies only for perturbation norms and proxy evaluation')
    args = parser.parse_args()

    if args.final and args.norm:
        parser.error('--final and --norm are mutually exclusive.')

    if args.step_decay < 0:
        parser.error('--step_decay must be >= 0.')
    if args.step_decay > args.round:
        parser.error(f'--step_decay ({args.step_decay}) must be <= --round ({args.round}), or 0 to disable it.')

    if args.DP_i == 'correct':
        args.DP_i = 'center'

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'tiny-imagenet':
        args.num_classes = 200
    else:
        args.num_classes = 100

    if args.log_name is None:
        args.log_name = generate_log_name(args)
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
