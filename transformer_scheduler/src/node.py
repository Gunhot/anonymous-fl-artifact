import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import random
from workers import get_model_sd, set_model_sd, build_model
import time
class Client:
    def __init__(self, nodeID, args, node_indices=None):
        self.nodeID = nodeID
        self.__node_indices = node_indices
        self.__args = args

    def _run_training_epochs(self, model, loader, optimizer, epochs, device):
        model.train()
        for epoch_idx in range(epochs): 
            t0 = time.time()
            optimizer.zero_grad() 
            for step, batch in enumerate(loader):
                batch = {
                    k: v.to(device) 
                    for k, v in batch.items() 
                    if isinstance(v, torch.Tensor) and k not in ['source', 'target']
                }
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()


    def _build_train_loader(self, dataset, seed):
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=self.__args.batch_size,
            shuffle=True,
            generator=generator
        )

    def _reset_rng(self, seed):
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def _get_optimizer(self, model, lr):
        if self.__args.opt == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
        elif self.__args.opt == 'adam':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    def train(self, device, lr, model, train_dataset, round, finetune=False, virtual_sd=None):
        base_seed = (self.__args.seed * 1000000) + (round * 1000) + self.nodeID
        main_seed = base_seed * 10 + 1
        finetune_seed = base_seed * 10 + 2
        
        dataset = Subset(train_dataset, self.__node_indices)
        train_subset = dataset
        # ---------------------------------------------------------------------
        # 1. Finetuning branch
        # ---------------------------------------------------------------------
        finetune_weight = None
        if finetune:
            f_model, _ = build_model(self.__args)
            set_model_sd(f_model, get_model_sd(model))
            
            f_model.to(device)

            num_ft_samples = int(len(train_subset) * self.__args.ft_subset)
            ft_subset = Subset(train_subset, list(range(num_ft_samples)))
            ft_loader = self._build_train_loader(ft_subset, seed=finetune_seed)

            ft_optimizer = self._get_optimizer(f_model, lr * 0.1)
            self._reset_rng(finetune_seed)
            
            self._run_training_epochs(
                model=f_model,
                loader=ft_loader,
                optimizer=ft_optimizer,
                epochs=1,
                device=device
            )
            
            f_model.to('cpu')
            finetune_weight = get_model_sd(f_model)
            del f_model

        # ---------------------------------------------------------------------
        # 2. Main Model Training (Standard)
        # ---------------------------------------------------------------------
        model.to(device)
        main_loader = self._build_train_loader(train_subset, seed=main_seed)
        optimizer = self._get_optimizer(model, lr)
        self._reset_rng(main_seed)
        self._run_training_epochs(
            model=model,
            loader=main_loader,
            optimizer=optimizer,
            epochs=self.__args.local_epoch,
            device=device
        )
        model.to('cpu')
        weight = get_model_sd(model)

        # ---------------------------------------------------------------------
        # 3. Virtual Model Training (Clean Execution Tracking)
        # ---------------------------------------------------------------------
        virtual_weight = None
        if virtual_sd is not None:
            v_model, _ = build_model(self.__args)
            set_model_sd(v_model, virtual_sd)
            v_model.to(device)
            v_loader = self._build_train_loader(train_subset, seed=main_seed)
            v_optimizer = self._get_optimizer(v_model, lr)
            self._reset_rng(main_seed)
            
            self._run_training_epochs(
                model=v_model,
                loader=v_loader,
                optimizer=v_optimizer,
                epochs=self.__args.local_epoch,
                device=device
            )
            
            v_model.to('cpu')
            virtual_weight = get_model_sd(v_model)
            del v_model

        return weight, finetune_weight, virtual_weight
