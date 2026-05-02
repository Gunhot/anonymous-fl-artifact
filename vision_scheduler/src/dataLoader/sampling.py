import numpy as np
from torch.utils.data import Subset

def _resolve_rng(seed=None):
    if seed is None:
        return np.random
    return np.random.default_rng(int(seed))


def iid(dataset, num_nodes, seed=None):
    rng = _resolve_rng(seed)
    num_sample = int(len(dataset) / num_nodes)
    dict_nodes = {}
    index = np.arange(len(dataset))
    for i in range(num_nodes):
        chosen = rng.choice(index, num_sample, replace=False)
        dict_nodes[i] = np.asarray(chosen)
        index = np.setdiff1d(index, dict_nodes[i], assume_unique=False)
    return dict_nodes


def iid_waterbirds(dataset_1, dataset_2, num_nodes, seed=None):
    base_seed = None if seed is None else int(seed)
    rng_1 = _resolve_rng(base_seed)
    rng_2 = _resolve_rng(None if base_seed is None else base_seed + 1)
    num_sample_1 = int(len(dataset_1) / (num_nodes // 2))
    num_sample_2 = int(len(dataset_2) / (num_nodes // 2))

    dict_nodes = {}
    index_1 = np.arange(len(dataset_1))
    for i in range(num_nodes // 2):
        chosen = rng_1.choice(index_1, num_sample_1, replace=False)
        dict_nodes[i] = np.asarray(chosen)
        index_1 = np.setdiff1d(index_1, dict_nodes[i], assume_unique=False)

    index_2 = np.arange(len(dataset_2))
    for i in range(num_nodes // 2, num_nodes):
        chosen = rng_2.choice(index_2, num_sample_2, replace=False)
        dict_nodes[i] = np.asarray(chosen)
        index_2 = np.setdiff1d(index_2, dict_nodes[i], assume_unique=False)

    return dict_nodes


def noniid_num(dataset, num_nodes, beta, seed=None):
    rng = _resolve_rng(seed)
    proportions = len(dataset) * rng.dirichlet(np.repeat(beta, num_nodes))
    dict_nodes = {}
    total_sum = 0

    index = np.arange(len(dataset))
    for i in range(num_nodes):
        n_take = int(proportions[i])
        if n_take <= 0:
            dict_nodes[i] = np.array([], dtype=np.int64)
            continue
        n_take = min(n_take, len(index))
        chosen = rng.choice(index, n_take, replace=False)
        dict_nodes[i] = np.asarray(chosen)
        index = np.setdiff1d(index, dict_nodes[i], assume_unique=False)

        total_sum += len(dict_nodes[i])
        # print(len(dict_nodes[i]))

    #print(total_sum)
    #exit()

    return dict_nodes


def split_dataset(dataset):
    labels = np.array([label for _, label in dataset])
    K = np.max(labels) + 1
    idx_1 = []
    idx_2 = []

    for k in range(K // 2):
        idx_k = np.where(labels == k)[0]
        idx_1.extend(idx_k)

    for k in range(K // 2, K):
        idx_k = np.where(labels == k)[0]
        idx_2.extend(idx_k)

    dataset_1 = Subset(dataset,idx_1)
    dataset_2 = Subset(dataset,idx_2)

    return dataset_1, dataset_2

def noniid_catastrophic(dataset, num_nodes, seed=None):
    dataset_1, dataset_2 = split_dataset(dataset)

    if seed is None:
        dict_1 = iid(dataset_1, num_nodes // 2)
        dict_2 = iid(dataset_2, num_nodes // 2)
    else:
        base_seed = int(seed)
        dict_1 = iid(dataset_1, num_nodes // 2, seed=base_seed)
        dict_2 = iid(dataset_2, num_nodes // 2, seed=base_seed + 1)

    return dict_1, dict_2


def noniid(dataset, num_nodes, beta, seed=None):
    rng = _resolve_rng(seed)
    labels = np.array([label for _, label in dataset])
    min_size = 0
    K = np.max(labels) + 1
    N = labels.shape[0]
    net_dataidx_map = {}
    n_nets = num_nodes

    while min_size < 10:

        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(beta, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    data_min = 100000
    data_max = 0

    for j in range(n_nets):
        rng.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
        data_min = min(data_min, len(net_dataidx_map[j]))
        data_max = max(data_max, len(net_dataidx_map[j]))
        # print(len(net_dataidx_map[j]))

    # print(data_min)
    # print(data_max)
    # exit()

    return net_dataidx_map


def noniid_nlp(dataset, num_nodes, beta, seed=None):
    rng = _resolve_rng(seed)

    min_size = 0
    N = len(dataset)
    net_dataidx_map = {}
    n_nets = num_nodes

    while min_size < 640:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset

        idx_k = np.arange(N)
        rng.shuffle(idx_k)
        proportions = rng.dirichlet(np.repeat(beta, n_nets))
        ## Balance
        proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
        proportions = proportions/proportions.sum()
        proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        # print(min_size)


    for j in range(n_nets):
        rng.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])

    return net_dataidx_map
