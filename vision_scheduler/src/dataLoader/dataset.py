import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import random
# import torchtext
# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.data.utils import get_tokenizer
from .sampling import noniid_nlp
import numpy as np
from PIL import Image
import pandas as pd
from .ImageFolder import ImageFolder

BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))


def _sync_seed_from_args(args):
    if not hasattr(args, 'seed') or args.seed is None:
        return
    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length) -> None:
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[0].size(0)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        seq_length = min(self.seq_length, self.S - index)
        return self.dataset[:, self.idx[index]:self.idx[index]+seq_length]

def get_dataset(args):
    _sync_seed_from_args(args)

    if args.dataset == 'mnist':
        apply_transform_train = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Resize((32,32)),
             transforms.Normalize((0.1307),
                                  (0.3081)),
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((32,32)),
             transforms.Normalize((0.1307),
                                  (0.3081)),
                                  ]
        )
        dir = os.path.join(BASE_DATA_DIR, 'mnist')
        train_dataset = datasets.MNIST(dir, train=True, download=True,
                                         transform=apply_transform_train)
        test_dataset = datasets.MNIST(dir, train=False, download=True,
                                        transform=apply_transform_test)
        return train_dataset, test_dataset

    if args.dataset == 'cifar100':
        apply_transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
                                  ]
        )
        dir = os.path.join(BASE_DATA_DIR, 'cifar100')
        train_dataset = datasets.CIFAR100(dir, train=True, download=False,
                                         transform=apply_transform_train)
        test_dataset = datasets.CIFAR100(dir, train=False, download=False,
                                        transform=apply_transform_test)

        return train_dataset, test_dataset

    if args.dataset == 'tiny-imagenet':
        apply_transform_train = transforms.Compose(
            [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                                  ]
        )
        dir = os.path.join(BASE_DATA_DIR, 'tiny-imagenet')
        train_dataset = datasets.ImageFolder(os.path.join(dir, 'train'), transform=apply_transform_train)
        test_dataset = datasets.ImageFolder(os.path.join(dir, 'val'), transform=apply_transform_test)

        return train_dataset, test_dataset

    if args.dataset == 'femnist':

        apply_transform_train = transforms.Compose(
            [
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
           transforms.Normalize((0.1307),
                                  (0.3081)),
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((32,32)),
            transforms.Normalize((0.1307),
                                  (0.3081)),
                                  ]
        )

        train_datasets = list()

        if args.nodes == 8:
            dir = os.path.join(BASE_DATA_DIR, 'femnist_silo')

            for i in range(8):
                train_datasets.append(ImageFolder(os.path.join(dir, f'train_{i}'), transform=apply_transform_train))

            test_dataset = ImageFolder(os.path.join(dir, 'test'), transform=apply_transform_test)

        else:
            dir = os.path.join(BASE_DATA_DIR, 'femnist')

            for i in range(128):
                train_datasets.append(ImageFolder(os.path.join(dir, f'train_{i}'), transform=apply_transform_train))

            test_dataset = ImageFolder(os.path.join(dir, 'test'), transform=apply_transform_test)

        return train_datasets, test_dataset

    if args.dataset == 'imagenet':
        apply_transform_train = transforms.Compose(
            [transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                                  ]
        )
        apply_transform_test = transforms.Compose(
            [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                                  ]
        )
        dir = os.path.join(BASE_DATA_DIR, 'imagenet')
        train_dataset = datasets.ImageNet(dir, split='train',
                                         transform=apply_transform_train)
        test_dataset = datasets.ImageNet(dir, split='val',
                                        transform=apply_transform_test)

        return train_dataset, test_dataset

    if args.dataset == 'waterbirds':
        subsets = prepare_confounder_data()
        train_dataset_0 = subsets['train_0']
        train_dataset_1 = subsets['train_1']
        val_dataset = subsets['val']
        test_dataset = subsets['test']

        return train_dataset_0, train_dataset_1, val_dataset


    if args.dataset == 'wikitext-2':
        dir = os.path.join(BASE_DATA_DIR, 'wikitext-2')
        train_iter = torchtext.datasets.WikiText2(dir, split='train')
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(map(tokenizer,train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        # print(len(vocab))

        train_iter, test_iter = torchtext.datasets.WikiText2(root=dir, split=('train', 'test'))
        def data_process(raw_text_iter):
            data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) \
                for item in raw_text_iter]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

        def split(dataset: torch.Tensor):
            dataset = dataset[:(len(dataset) // args.nodes) * args.nodes]
            return dataset.reshape(args.nodes, -1)

        def batchify(dataset : torch.Tensor, batch_size):
            num_batch = len(dataset) // batch_size
            dataset = dataset.narrow(0, 0, num_batch * batch_size)
            dataset = dataset.reshape(batch_size, -1)
            return dataset

        train_dataset = data_process(train_iter)
        test_dataset = data_process(test_iter)
        train_datasets = split(train_dataset)
        train_datasets = [batchify(dataset, args.batch_size) for dataset in train_datasets]
        test_dataset = batchify(test_dataset, args.batch_size)

        return train_datasets, test_dataset

if __name__ == "__main__":

    import torch

    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224))]
    )
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    train_datasets = []
    for domain in domains:
        dir = '~/scheduler/data/' + domain + '_train'
        train_dataset = datasets.ImageFolder(dir, transform=apply_transform)
        train_datasets.append(train_dataset)

    dataset = torch.utils.data.ConcatDataset(train_datasets)
    TrainLoader = DataLoader(dataset, shuffle=False, num_workers=os.cpu_count(), batch_size=1)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for i, (inputs, labels) in enumerate(TrainLoader):
        if i % 10000 == 0:
            print(i)

        for j in range(3):
            mean[j] += inputs[:,j,:,:].mean()
            std[j] += inputs[:,j,:,:].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean, std)
