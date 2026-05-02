import argparse
import os


VISION_SCHEDULER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generate_log_name(args):
    log_name = (
        # 1. System Configuration
        f"NP{args.n_procs}"
        # 2. Federated Learning Configuration
        f"N{args.nodes}FR{args.fraction}R{args.round}"
        # 3. Dataset Configuration
        f"DS{args.dataset}MD{args.model}BS{args.batch_size}I{args.iid}BT{args.beta}"
        # 4. Learning Configuration
        f"LE{args.local_epoch}LR{args.lr}FT{args.ft_lr}OP{args.opt}DC{args.lr_decay}"
        # 5. Free Rider
        f"DP{args.DP}SG{args.sigma}"
        f"p1{args.p1}p2{args.p2}OM{args.omega}SD{args.seed}"
        f"{'QF1' if args.qsn_fixed_mask else ''}"
        f"NU{args.noise_update}"
        f"ablation"
    )
    log_path = os.path.join(VISION_SCHEDULER_DIR, "save", args.dataset, log_name)
    return log_path

def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')
    # Free Rider
    parser.add_argument('--DP', type=str, default='none',
                        help='distortion type: none=None, ours=ClientDrift, gausg=Gaussian(layer grad norm), quan=Quantization, prev=PrevGrad, lapl=Laplace')
    parser.add_argument('--sigma', type=int, default=0,
                        help='sigma ratio in percent (>0: noise). *g: layer grad-norm ratio, *G: global average-matching ratio')
    parser.add_argument('--noise_update', type=int, default=0, choices=[0, 1],
                        help='If 1, subtract the noise added in get_model() before update_node_info bookkeeping')
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

    # Dataset Configuration                
    parser.add_argument('--dataset',  type=str, default='cifar100',
                        choices=['cifar100', 'tiny-imagenet'],
                        help='type of dataset')
    parser.add_argument('--model', type=str,
                        choices=['ResNet50', 'RResNet50', 'MobileNetV2'],
                        help='model backbone')
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

    parser.add_argument('--log_name', type=str, default=None,
                        help='Optional override for log directory path')
    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = generate_log_name(args)
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
