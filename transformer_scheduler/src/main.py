import os
import random
import shutil
import time

import numpy as np
import torch
import torch.multiprocessing as mp

import arguments
from dataLoader.dataLoaders import getNodeIndicies
from node import Client
from server import Server
from workers import (
    cleanup_ipc_path,
    get_model_sd,
    gpu_test_worker,
    gpu_train_worker,
    load_ipc_state_dict,
    save_ipc_state_dict,
)


def set_global_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_tracking(roundIdx, tracking_ids, proxy_weights, proxy_trains, proxy_finetunes, test_q, args, server_weight):
    test_count = 0

    server_path = save_ipc_state_dict(server_weight, args, f"test_r{roundIdx}_server")
    test_q.put({
        'round': roundIdx,
        'weight': server_path,
        'worker_type': 'server',
        'cleanup_weight': True
    })
    test_count += 1

    for node_id in tracking_ids:
        if node_id not in proxy_weights:
            continue

        proxy_path = save_ipc_state_dict(proxy_weights[node_id], args, f"test_r{roundIdx}_proxy_{node_id}")
        test_q.put({
            'round': roundIdx,
            'weight': proxy_path,
            'worker_type': f'proxy_{node_id}',
            'cleanup_weight': True
        })
        test_count += 1

        train_sd = proxy_trains.get(node_id)
        if train_sd is not None:
            train_path = save_ipc_state_dict(train_sd, args, f"test_r{roundIdx}_proxy_train_{node_id}")
            test_q.put({
                'round': roundIdx,
                'weight': train_path,
                'worker_type': f'proxy_train_{node_id}',
                'cleanup_weight': True
            })
            test_count += 1

        finetune_sd = proxy_finetunes.get(node_id)
        if finetune_sd is not None:
            finetune_path = save_ipc_state_dict(finetune_sd, args, f"test_r{roundIdx}_proxy_finetune_{node_id}")
            test_q.put({
                'round': roundIdx,
                'weight': finetune_path,
                'worker_type': f'proxy_finetune_{node_id}',
                'cleanup_weight': True
            })
            test_count += 1

    return test_count


if __name__ == "__main__":
    args = arguments.parser()
    set_global_seed(args.seed)
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    n_devices = torch.cuda.device_count()
    devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
    os.environ["OMP_NUM_THREADS"] = "1"
    print("> Setting:", args)

    n_train_processes = n_devices * args.n_procs
    trainQ = mp.Queue()
    train_ack_q = mp.Queue()
    test_q = mp.Queue()
    test_ack_q = mp.Queue()

    test_procs = []
    for i in range(n_devices):
        p = mp.Process(target=gpu_test_worker, args=(test_q, test_ack_q, devices[i], args))
        p.start()
        test_procs.append(p)

    train_procs = []
    for i in range(n_train_processes):
        p = mp.Process(target=gpu_train_worker, args=(trainQ, train_ack_q, devices[i % n_devices], args))
        p.start()
        train_procs.append(p)
        time.sleep(0.1)

    server = Server(args)
    server.set_initial_model()
    nodeIDs = [i for i in range(args.nodes)]
    nodeindices = getNodeIndicies(nodeIDs, args.nodes, args)
    nodes = []
    for nodeID in nodeIDs:
        nodes.append(Client(nodeID, args, nodeindices[nodeID]))

    experiment_dir = args.log_name
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir)

    lr = args.lr
    tracking_ids = [0, 1, 2]
    for roundIdx in range(1, args.round + 1):
        train_count = 0
        test_count = 0
        cur_time = time.time()
        lr *= args.lr_decay

        print(f"Round {roundIdx}", end=', ')

        n_trainees = int(len(nodeIDs) * args.fraction)
        selected_ids = random.sample(nodeIDs, n_trainees)
        trainees = [nodes[i] for i in selected_ids]
        tracking_selected_ids = [
            node_id for node_id in tracking_ids
            if node_id in selected_ids
        ]

        clean_weight = get_model_sd(server.model)

        centered_noise = None
        if args.DP == 'ours' and roundIdx != 1:
            centered_noise = server.center_noise(roundIdx, selected_ids)

        proxy_weights = {}
        for node in trainees:
            proxy_weights[node.nodeID] = server.get_model(
                round_idx=roundIdx,
                client_id=node.nodeID,
                centered_noise=centered_noise
            )

        for node in trainees:
            proxy_path = save_ipc_state_dict(proxy_weights[node.nodeID], args, f"train_in_r{roundIdx}_n{node.nodeID}_proxy")
            trainQ.put({
                'type': 'train',
                'node': node,
                'lr': lr,
                'weight': proxy_path,
                'round': roundIdx,
                'finetune': node.nodeID in tracking_selected_ids,
                'virtual_weight': None,
                'cleanup_weight': True
            })
            train_count += 1

        if args.DP == 'ours':
            server.reset_perturbation_direction()

        tracking_finetunes = {}
        tracking_proxy_trains = {}
        for _ in range(train_count):
            msg = train_ack_q.get()
            node_id = msg['id']
            proxy_update = load_ipc_state_dict(msg['weight'])
            proxy_finetune = load_ipc_state_dict(msg['finetune_weight'])
            cleanup_ipc_path(msg['weight'])
            cleanup_ipc_path(msg['finetune_weight'])

            server.update_node_info(proxy_update, proxy_weights[node_id], clean_weight, node_id)
            if node_id in tracking_selected_ids:
                tracking_finetunes[node_id] = proxy_finetune
                tracking_proxy_trains[node_id] = proxy_update
            del msg

        time.sleep(2.0)

        should_evaluate_round = bool(tracking_selected_ids)
        if should_evaluate_round:
            time.sleep(1.0)
            test_count += evaluate_tracking(
                roundIdx=roundIdx,
                tracking_ids=tracking_selected_ids,
                proxy_weights=proxy_weights,
                proxy_trains=tracking_proxy_trains,
                proxy_finetunes=tracking_finetunes,
                test_q=test_q,
                args=args,
                server_weight=clean_weight
            )

        server.avg_parameters(roundIdx)
        print(f"Elapsed Time : {time.time() - cur_time:.1f}")

        time.sleep(1.0)
        for _ in range(test_count):
            test_ack_q.get()
        time.sleep(2.0)

    for _ in range(n_train_processes):
        trainQ.put('kill')
    for p in train_procs:
        p.join()

    for _ in range(n_devices):
        test_q.put('kill')
    for p in test_procs:
        p.join()

    print(f"\n> Experiment finished. Results saved in: {args.log_name}")
