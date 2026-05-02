import argparse
import os


TRANSFORMER_SCHEDULER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generate_log_name(args):
    log_name = (
        f"NP{args.n_procs}" #1
        f"N{args.nodes}FR{args.fraction}R{args.round}FTS{args.ft_subset}" #4
        f"DP{args.DP}SG{args.sigma}NU{args.noise_update}"#4
        f"p1{args.p1}p2{args.p2}o{args.omega}"
        f"LOR{args.lora_r}LOA{args.lora_alpha}LOD{args.lora_dropout}"#3
        f"UP{args.use_pretrained}"#1
        f"SD{args.seed}ML{args.max_len}"#2
        f"MD{args.model}"#1
        f"DS{args.dataset}BS{args.batch_size}I{args.iid}BT{args.beta}"#5
        f"LE{args.local_epoch}LR{args.lr}DC{args.lr_decay}OPT{args.opt}"#3
    )
    log_path = os.path.join(TRANSFORMER_SCHEDULER_DIR, "save", args.dataset, log_name)
    return log_path


def parser():
    parser = argparse.ArgumentParser(
        description='FedAvg LoRA hyperparameters with trainable-parameter state sync'
    )

    # Learning Configuration
    parser.add_argument('--local_epoch',  type=int, default=3,
                        help='number of local epochs per client update')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='client optimizer learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='round-wise decay applied after each aggregation')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='client optimizer (adam or sgd)')


    # Dataset / model configuration
    parser.add_argument('--model',  type=str, default='distilbert',
                        choices=['gpt2s', 'gpt2slora', 'distilbert', 'distilbertlora'],
                        help='model to federate')
    parser.add_argument('--use_pretrained', type=int, default=1, choices=[0, 1],
                        help='1: load pretrained weights, 0: initialize model from config')
    parser.add_argument('--dataset',  type=str, default='20news',
                        choices=['20news', 'e2e'],
                        help='text dataset to federate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size for client updates')
    parser.add_argument('--iid', type=int, default=1,
                        help='1: IID, 0: existing non-IID, 2: equal counts with label/source-based non-IID')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Dirichlet concentration for non-IID splits')
    
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducibility')
    parser.add_argument('--max_len', type=int, default=512,
                        help='maximum sequence length for tokenization')

    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout probability')

    # DP / Noise Configuration
    parser.add_argument('--DP', type=str, default="none",
                        help='gausg, qsn, lpp, or ours')
    parser.add_argument('--sigma', type=float, default=0, help='sigma for noise')
    parser.add_argument('--noise_update', type=int, default=0, choices=[0, 1],
                        help='if 1, subtract noise before update_node_info')

    # FedQSN Configuration
    parser.add_argument('--p1', type=float, default=0.0, help='FedQSN: Server Mask Ratio')
    parser.add_argument('--p2', type=float, default=0.0, help='FedQSN: Client Mask Ratio')
    parser.add_argument('--omega', type=int, default=0, help='FedQSN: Quantization bit-width')

    # Federated Learning Configuration
    parser.add_argument('--nodes', type=int, default=50,
                        help='total number of clients')
    parser.add_argument('--fraction', type=float, default=0.1,
                        help='fraction of clients per round')
    parser.add_argument('--round', type=int, default=30,
                        help='number of communication rounds')
    parser.add_argument('--ft_subset', type=float, default=1.0,
                        help='fraction of samples to use for client 0 finetuning branch (0.0-1.0)')


    # System Configuration
    parser.add_argument('--n_procs', type=int, default=1,
                        help='number of worker processes (kept for compatibility)')

    parser.add_argument('--log_name', type=str, default=None,
                        help='Optional override for log directory path')

    args = parser.parse_args()
    if args.log_name is None:
        args.log_name = generate_log_name(args)
    return args


if __name__ == "__main__":
    args = parser()
    print(args)
