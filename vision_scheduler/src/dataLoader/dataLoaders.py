from .sampling import iid, noniid, noniid_nlp, noniid_num, iid_waterbirds
from .dataset import get_dataset
from torch.utils.data import DataLoader

def getNodeIndicies(nodeIDs, n_normal,  args):

	train_dataset, test_dataset = get_dataset(args)

	node_indices = {}
	n_nodes = len(nodeIDs)
	seed = args.seed

	if args.iid == 1:
		dict_nodes = iid(train_dataset, n_nodes, seed=seed)

	else:
		dict_nodes = noniid(train_dataset, n_nodes, args.beta, seed=seed)

	for i, nodeID in enumerate(nodeIDs):
		node_indices[nodeID] = dict_nodes[i]


	return node_indices
