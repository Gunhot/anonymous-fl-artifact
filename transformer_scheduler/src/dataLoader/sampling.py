import numpy as np
from torch.utils.data import Subset

def _equal_quota(num_samples, num_nodes):
    quota = np.full(num_nodes, num_samples // num_nodes, dtype=int)
    quota[: num_samples % num_nodes] += 1
    return quota

def iid(dataset, num_nodes, seed=1):
    np.random.seed(seed)
    num_sample = int(len(dataset)/(num_nodes))
    # num_sample = int(len(dataset)/(num_nodes)) // 10

    dict_nodes = {}
    index = [i for i in range(len(dataset))]
    for i in range(num_nodes):
        dict_nodes[i] = np.random.choice(index, num_sample,
                                         replace=False)
        index = sorted(set(index) - set(dict_nodes[i]))
    return dict_nodes


def iid_waterbirds(dataset_1, dataset_2, num_nodes, seed=1):

    np.random.seed(seed)
    num_sample_1 = int(len(dataset_1)/(num_nodes // 2))
    num_sample_2 = int(len(dataset_2)/(num_nodes // 2))

    dict_nodes = {}
    index_1 = [i for i in range(len(dataset_1))]
    for i in range(num_nodes // 2):
        dict_nodes[i] = np.random.choice(index_1, num_sample_1,
                                         replace=False)
        index_1 = sorted(set(index_1) - set(dict_nodes[i]))
    
    index_2 = [i for i in range(len(dataset_2))]
    for i in range(num_nodes // 2, num_nodes):
        dict_nodes[i] = np.random.choice(index_2, num_sample_2,
                                         replace=False)
        index_2 = sorted(set(index_2) - set(dict_nodes[i]))

    return dict_nodes

def noniid_num(dataset, num_nodes, beta, seed=1):
    np.random.seed(seed)
    # num_sample = int(len(dataset)/(num_nodes))
    proportions = len(dataset) * np.random.dirichlet(np.repeat(beta, num_nodes))
    dict_nodes = {}
    total_sum = 0

    index = [i for i in range(len(dataset))]
    for i in range(num_nodes):
        dict_nodes[i] = np.random.choice(index, int(proportions[i]),
                                         replace=False)
        index = sorted(set(index) - set(dict_nodes[i]))

        total_sum += len(dict_nodes[i])
        # print(len(dict_nodes[i]))

    #print(total_sum)
    #exit()

    return dict_nodes


def split_dataset(dataset, seed=1):
    labels = np.array([label for _, label in dataset])
    K = np.max(labels) + 1
    np.random.seed(seed)
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

def noniid_catastrophic(dataset, num_nodes, seed=1):

    dataset_1, dataset_2 = split_dataset(dataset, seed=seed)
    
    dict_1 = iid(dataset_1, num_nodes // 2, seed=seed)
    dict_2 = iid(dataset_2, num_nodes // 2, seed=seed)

    return dict_1, dict_2

def noniid(dataset, num_nodes, beta, seed=1):
    # Support HuggingFace datasets by checking for 'labels' column
    if hasattr(dataset, 'column_names') and 'labels' in dataset.column_names:
        labels = np.array(dataset['labels'])
    else:
        labels = np.array([label for _, label in dataset])
    
    min_size = 0
    K = np.max(labels) + 1
    N = labels.shape[0]
    net_dataidx_map = {}
    n_nets = num_nodes
    np.random.seed(seed)

    while min_size < 10:

        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    data_min = 100000
    data_max = 0

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
        data_min = min(data_min, len(net_dataidx_map[j]))
        data_max = max(data_max, len(net_dataidx_map[j]))
        # print(len(net_dataidx_map[j]))

    # print(data_min)
    # print(data_max)
    # exit()

    return net_dataidx_map


def noniid_label_iid_count(dataset, num_nodes, beta, seed=1):
    if hasattr(dataset, 'column_names') and 'labels' in dataset.column_names:
        labels = np.array(dataset['labels'])
    else:
        labels = np.array([label for _, label in dataset])

    K = np.max(labels) + 1
    np.random.seed(seed)

    class_indices = {}
    for k in range(K):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        class_indices[k] = idx_k.tolist()

    quota = _equal_quota(len(dataset), num_nodes)
    class_pref = np.random.dirichlet(np.repeat(beta, K), size=num_nodes)
    net_dataidx_map = {i: [] for i in range(num_nodes)}

    for node_id in range(num_nodes):
        while len(net_dataidx_map[node_id]) < quota[node_id]:
            available = [k for k in range(K) if len(class_indices[k]) > 0]
            probs = class_pref[node_id][available]
            probs = probs / probs.sum()
            picked_class = np.random.choice(available, p=probs)
            net_dataidx_map[node_id].append(class_indices[picked_class].pop())

    for node_id in range(num_nodes):
        np.random.shuffle(net_dataidx_map[node_id])
        net_dataidx_map[node_id] = np.array(net_dataidx_map[node_id])

    return net_dataidx_map


def noniid_nlp(dataset, num_nodes, beta, seed=1):

    min_size = 0
    N = len(dataset)
    net_dataidx_map = {}
    n_nets = num_nodes
    np.random.seed(seed)

    while min_size < 640:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
    
        idx_k = np.arange(N)
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(beta, n_nets))
        ## Balance
        proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
        proportions = proportions/proportions.sum()
        proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        # print(min_size)
        

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
    
    return net_dataidx_map


def noniid_e2e(dataset, num_nodes, beta, seed=1):
    np.random.seed(seed)
    source_to_indices = {}

    for idx, source in enumerate(dataset['source']):
        if source not in source_to_indices:
            source_to_indices[source] = []
        source_to_indices[source].append(idx)

    grouped_indices = list(source_to_indices.values())
    np.random.shuffle(grouped_indices)

    net_dataidx_map = {i: [] for i in range(num_nodes)}

    # Assign at least one source group to each node first.
    for i in range(num_nodes):
        net_dataidx_map[i].extend(grouped_indices[i])

    # Assign remaining source groups to nodes using Dirichlet probabilities.
    proportions = np.random.dirichlet(np.repeat(beta, num_nodes))
    for indices in grouped_indices[num_nodes:]:
        node_id = np.random.choice(np.arange(num_nodes), p=proportions)
        net_dataidx_map[node_id].extend(indices)

    for i in range(num_nodes):
        np.random.shuffle(net_dataidx_map[i])
        net_dataidx_map[i] = np.array(net_dataidx_map[i])

    return net_dataidx_map


def noniid_source_iid_count(dataset, num_nodes, beta, seed=1):
    sources = np.array(dataset['source'])
    _, source_ids = np.unique(sources, return_inverse=True)
    num_sources = int(source_ids.max()) + 1
    np.random.seed(seed)

    source_indices = {}
    for source_id in range(num_sources):
        idx_s = np.where(source_ids == source_id)[0]
        np.random.shuffle(idx_s)
        source_indices[source_id] = idx_s.tolist()

    quota = _equal_quota(len(dataset), num_nodes)
    source_pref = np.random.dirichlet(np.repeat(beta, num_sources), size=num_nodes)
    net_dataidx_map = {i: [] for i in range(num_nodes)}

    for node_id in range(num_nodes):
        while len(net_dataidx_map[node_id]) < quota[node_id]:
            available = [source_id for source_id in range(num_sources) if len(source_indices[source_id]) > 0]
            probs = source_pref[node_id][available]
            probs = probs / probs.sum()
            picked_source = np.random.choice(available, p=probs)
            net_dataidx_map[node_id].append(source_indices[picked_source].pop())

    for node_id in range(num_nodes):
        np.random.shuffle(net_dataidx_map[node_id])
        net_dataidx_map[node_id] = np.array(net_dataidx_map[node_id])

    return net_dataidx_map
