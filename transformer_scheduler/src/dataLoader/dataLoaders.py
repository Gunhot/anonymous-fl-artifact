from .sampling import (
    iid,
    noniid,
    noniid_e2e,
    noniid_label_iid_count,
    noniid_source_iid_count,
)
from .dataset import get_dataset


def getNodeIndicies(nodeIDs, n_normal, args):
    train_dataset, test_dataset, _ = get_dataset(args)

    node_indices = {}
    n_nodes = len(nodeIDs)

    if args.iid == 1:
        dict_nodes = iid(train_dataset, n_nodes, seed=args.seed)
    elif args.iid == 2:
        if args.dataset == '20news':
            dict_nodes = noniid_label_iid_count(train_dataset, n_nodes, args.beta, seed=args.seed)
        elif args.dataset == 'e2e':
            dict_nodes = noniid_source_iid_count(train_dataset, n_nodes, args.beta, seed=args.seed)
        else:
            dict_nodes = iid(train_dataset, n_nodes, seed=args.seed)
    else:
        if args.dataset == '20news':
            dict_nodes = noniid(train_dataset, n_nodes, args.beta, seed=args.seed)
        elif args.dataset == 'e2e':
            dict_nodes = noniid_e2e(train_dataset, n_nodes, args.beta, seed=args.seed)
        else:
            dict_nodes = iid(train_dataset, n_nodes, seed=args.seed)

    for i, nodeID in enumerate(nodeIDs):
        node_indices[nodeID] = dict_nodes[i]

    return node_indices
