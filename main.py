import torch
import torch.nn.functional as F
import numpy as np
import time
import arguments
import random
import os
import torch.multiprocessing as mp
import shutil
from node import Client
from server import Server
from workers import *
from dataLoader.dataLoaders import getNodeIndicies
from dataLoader.dataset import get_dataset
from collections import Counter

def set_global_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cpu_clone_sd(sd):
    if sd is None:
        return None
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in sd.items()
    }


def evaluate_final(roundIdx, selected_ids, proxy_weights, proxy_trains, proxy_finetunes, test_q):
    test_count = 0

    for node_id in selected_ids:
        test_q.put({
            'round': roundIdx,
            'weight': proxy_weights[node_id],
            'worker_type': f'proxy_{node_id}'
        })
        test_count += 1

        train_sd = proxy_trains.get(node_id)
        if train_sd is not None:
            test_q.put({
                'round': roundIdx,
                'weight': train_sd,
                'worker_type': f'proxy_train_{node_id}'
            })
            test_count += 1

        finetune_sd = proxy_finetunes.get(node_id)
        if finetune_sd is not None:
            test_q.put({
                'round': roundIdx,
                'weight': finetune_sd,
                'worker_type': f'proxy_finetune_{node_id}'
            })
            test_count += 1

    if len(selected_ids) >= 3:
        for collusion_idx in range(5):
            collusion_before_sd = {}
            collusion_finetune_sd = {}
            collusion_after_sd = {}
            collusion_before_ids = random.sample(selected_ids, 3)
            collusion_finetune_ids = random.sample(selected_ids, 3)
            collusion_after_ids = random.sample(selected_ids, 3)
            for key in proxy_weights[collusion_before_ids[0]].keys():
                before0 = proxy_weights[collusion_before_ids[0]][key]
                before1 = proxy_weights[collusion_before_ids[1]][key]
                before2 = proxy_weights[collusion_before_ids[2]][key]
                collusion_before_sd[key] = ((before0 + before1 + before2) / 3.0).clone()

                finetune0 = proxy_finetunes[collusion_finetune_ids[0]][key]
                finetune1 = proxy_finetunes[collusion_finetune_ids[1]][key]
                finetune2 = proxy_finetunes[collusion_finetune_ids[2]][key]
                collusion_finetune_sd[key] = ((finetune0 + finetune1 + finetune2) / 3.0).clone()

                after0 = proxy_trains[collusion_after_ids[0]][key]
                after1 = proxy_trains[collusion_after_ids[1]][key]
                after2 = proxy_trains[collusion_after_ids[2]][key]
                collusion_after_sd[key] = ((after0 + after1 + after2) / 3.0).clone()

            test_q.put({
                'round': roundIdx,
                'weight': collusion_before_sd,
                'worker_type': f'collusion_before_{collusion_idx}'
            })
            test_count += 1
            test_q.put({
                'round': roundIdx,
                'weight': collusion_finetune_sd,
                'worker_type': f'collusion_finetune_{collusion_idx}'
            })
            test_count += 1
            test_q.put({
                'round': roundIdx,
                'weight': collusion_after_sd,
                'worker_type': f'collusion_after_{collusion_idx}'
            })
            test_count += 1

    return test_count


def state_dict_diff_norm(reference_sd, target_sd):
    diff_norm_sq = 0.0
    reference_norm_sq = 0.0
    tensor_count = 0

    for key, reference_tensor in reference_sd.items():
        if key not in target_sd:
            continue
        if not torch.is_floating_point(reference_tensor):
            continue

        target_tensor = target_sd[key]
        diff = target_tensor.detach().float().cpu() - reference_tensor.detach().float().cpu()
        reference = reference_tensor.detach().float().cpu()
        diff_norm_sq += torch.sum(diff * diff).item()
        reference_norm_sq += torch.sum(reference * reference).item()
        tensor_count += 1

    diff_norm = diff_norm_sq ** 0.5
    reference_norm = reference_norm_sq ** 0.5
    ratio = diff_norm / (reference_norm + 1e-8)
    return diff_norm, reference_norm, ratio, tensor_count


def proxy_norm_log_line(roundIdx, node_id, clean_weight, proxy_weight):
    proxy_norm, model_norm, ratio, tensor_count = state_dict_diff_norm(
        clean_weight,
        proxy_weight
    )
    return (
        f"Round: {roundIdx} | Proxy_ID: {node_id} | "
        f"Perturbation_Norm: {proxy_norm:.6e} | "
        f"Model_Norm: {model_norm:.6e} | "
        f"Ratio: {ratio:.6e} | "
        f"Tensor_Count: {tensor_count}\n"
    )


def write_proxy_norms(proxy_norm_txt_path, roundIdx, selected_ids, clean_weight, proxy_weights):
    with open(proxy_norm_txt_path, "a") as f:
        for node_id in sorted(selected_ids):
            f.write(proxy_norm_log_line(roundIdx, node_id, clean_weight, proxy_weights[node_id]))


def queue_norm_proxy_evaluations(
    roundIdx,
    proxy_ids,
    clean_weight,
    server,
    centered_noise,
    proxy_norm_txt_path,
    test_q,
    test_ack_q,
    max_pending
):
    pending_tests = 0
    max_pending = max(1, int(max_pending))

    def enqueue_eval(weight, worker_type):
        nonlocal pending_tests
        test_q.put({
            'round': roundIdx,
            'weight': weight,
            'worker_type': worker_type
        })
        pending_tests += 1
        if pending_tests >= max_pending:
            test_ack_q.get()
            pending_tests -= 1

    enqueue_eval(clean_weight, 'server')

    with open(proxy_norm_txt_path, "a") as f:
        for node_id in sorted(proxy_ids):
            proxy_weight = server.get_model(
                round_idx=roundIdx,
                client_id=node_id,
                tracking_ids=[],
                centered_noise=centered_noise,
                force_proxy=True
            )
            f.write(proxy_norm_log_line(roundIdx, node_id, clean_weight, proxy_weight))
            enqueue_eval(proxy_weight, f'proxy_{node_id}')
            del proxy_weight

    return pending_tests


def analyze(roundIdx, node_id, clean_weight, proxy_weights, proxy_updates, clean_updates, proxy_finetune, proxy_gradient):
    all_keys = [k for k in clean_weight.keys() if 'weight' in k or 'bias' in k]
    all_keys.sort()
    analysis_txt_path = os.path.join(experiment_dir, f"analysis_{node_id}.txt")
    with open(analysis_txt_path, "a") as f:
        def safe_cosine(vec1, vec2):
            if vec1.norm() == 0 or vec2.norm() == 0:
                return 0.0
            return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

        for key in clean_weight.keys():
            if 'weight' not in key and 'bias' not in key:
                continue
            # Load Tensors
            server = clean_weight[key].detach().float().cpu().flatten()                            # Clean Server
            proxy = proxy_weights[node_id][key].detach().float().cpu().flatten()                    # Node proxy (Start)
            proxy_update_weight = proxy_updates[node_id][key].detach().float().cpu().flatten()         # Proxy-trained
            clean_update_weight = clean_updates[node_id][key].detach().float().cpu().flatten()         # Clean-server-trained
            proxy_finetune_weight = proxy_finetune[key].detach().float().cpu().flatten()     # Proxy finetuned
            proxy_gradient_vec = proxy_gradient[key].detach().float().cpu().flatten()

            proxy_update_stack = torch.stack(
                [proxy_updates[other_id][key].detach().float().cpu().flatten() for other_id in proxy_updates.keys()],
                dim=0
            )
            clean_update_stack = torch.stack(
                [clean_updates[other_id][key].detach().float().cpu().flatten() for other_id in clean_updates.keys()],
                dim=0
            )
            proxy_server_weight = torch.mean(proxy_update_stack, dim=0)
            clean_server_weight = torch.mean(clean_update_stack, dim=0)

            # 1. Noise Vector (Clean Server -> Proxy)
            noise = proxy - server
            induced_error = (proxy_update_weight - clean_update_weight) - noise

            # 1-1. Peer proxy/noise stats (exclude node_id)
            peer_noise_sum = torch.zeros_like(proxy)
            peer_count = 0
            noise_cos_values = []
            induced_error_cos_values = []
            induced_error_stack = []
            for other_id, other_proxy_sd in proxy_weights.items():
                other_proxy = other_proxy_sd[key].detach().float().cpu().flatten()
                other_noise = other_proxy - server
                other_proxy_update_weight = proxy_updates[other_id][key].detach().float().cpu().flatten()
                other_clean_update_weight = clean_updates[other_id][key].detach().float().cpu().flatten()
                other_induced_error = (other_proxy_update_weight - other_clean_update_weight) - other_noise
                induced_error_stack.append(other_induced_error)
                if other_id == node_id:
                    continue
                peer_noise_sum += other_noise
                noise_cos_values.append(safe_cosine(noise, other_noise))
                induced_error_cos_values.append(safe_cosine(induced_error, other_induced_error))
                peer_count += 1

            mean_noise_norm = torch.norm(peer_noise_sum / peer_count).item() if peer_count > 0 else 0.0
            noise_cos_max = max(noise_cos_values) if noise_cos_values else 0.0
            noise_cos_min = min(noise_cos_values) if noise_cos_values else 0.0
            noise_cos_mean = float(np.mean(noise_cos_values)) if noise_cos_values else 0.0
            induced_error_cos_max = max(induced_error_cos_values) if induced_error_cos_values else 0.0
            induced_error_cos_min = min(induced_error_cos_values) if induced_error_cos_values else 0.0
            induced_error_cos_mean = float(np.mean(induced_error_cos_values)) if induced_error_cos_values else 0.0

            # 2. Proxy/Clean update vectors
            proxy_update = proxy_update_weight - proxy
            clean_update = clean_update_weight - server
            proxy_server_update = proxy_server_weight - server
            clean_server_update = clean_server_weight - server
            server_induced_error = torch.mean(torch.stack(induced_error_stack, dim=0), dim=0)

            # 3. Finetune update vector
            finetune_update = proxy_finetune_weight - proxy

            # Metrics
            server_norm = torch.norm(server).item()
            noise_norm = torch.norm(noise).item()
            proxy_gradient_norm = torch.norm(proxy_gradient_vec).item()
            proxy_update_norm = torch.norm(proxy_update).item()
            clean_update_norm = torch.norm(clean_update).item()
            proxy_server_update_norm = torch.norm(proxy_server_update).item()
            clean_server_update_norm = torch.norm(clean_server_update).item()
            induced_error_norm = torch.norm(induced_error).item()
            server_induced_error_norm = torch.norm(server_induced_error).item()
            finetune_update_norm = torch.norm(finetune_update).item()
            node_update_diff_norm = torch.norm(proxy_update_weight - clean_update_weight).item()
            mean_update_diff_norm = torch.norm(proxy_server_weight - clean_server_weight).item()

            # Pairwise cosine similarities among
            # server_induced_error, induced_error, noise, proxy_update,
            # clean_update, proxy_server_update, clean_server_update (7C2 = 21)
            cos_server_induced_induced = safe_cosine(server_induced_error, induced_error)
            cos_server_induced_noise = safe_cosine(server_induced_error, noise)
            cos_server_induced_proxy = safe_cosine(server_induced_error, proxy_update)
            cos_server_induced_clean = safe_cosine(server_induced_error, clean_update)
            cos_server_induced_proxy_server = safe_cosine(server_induced_error, proxy_server_update)
            cos_server_induced_clean_server = safe_cosine(server_induced_error, clean_server_update)
            cos_proxy_gradient_proxy = safe_cosine(proxy_gradient_vec, proxy_update)
            cos_proxy_gradient_clean = safe_cosine(proxy_gradient_vec, clean_update)
            cos_induced_noise = safe_cosine(induced_error, noise)
            cos_induced_proxy = safe_cosine(induced_error, proxy_update)
            cos_induced_clean = safe_cosine(induced_error, clean_update)
            cos_induced_proxy_server = safe_cosine(induced_error, proxy_server_update)
            cos_induced_clean_server = safe_cosine(induced_error, clean_server_update)
            cos_clean_proxy = safe_cosine(clean_update, proxy_update)
            cos_clean_noise = safe_cosine(clean_update, noise)
            cos_clean_proxy_server = safe_cosine(clean_update, proxy_server_update)
            cos_clean_clean_server = safe_cosine(clean_update, clean_server_update)
            cos_proxy_noise = safe_cosine(proxy_update, noise)
            cos_proxy_proxy_server = safe_cosine(proxy_update, proxy_server_update)
            cos_proxy_clean_server = safe_cosine(proxy_update, clean_server_update)
            cos_noise_proxy_server = safe_cosine(noise, proxy_server_update)
            cos_noise_clean_server = safe_cosine(noise, clean_server_update)
            cos_proxy_server_clean_server = safe_cosine(proxy_server_update, clean_server_update)

            log_line = (
                f"Round: {roundIdx} | Layer: {key} | "
                f"Server_Norm: {server_norm:10.6f} | "
                f"Noise_Norm: {noise_norm:10.6f} | "
                f"Mean_Noise_Norm: {mean_noise_norm:10.6f} | "


                f"Noise_Cos_Max: {noise_cos_max:.6f} | "
                f"Noise_Cos_Min: {noise_cos_min:.6f} | "
                f"Noise_Cos_Mean: {noise_cos_mean:.6f} | "

                f"InducedError_Cos_Max: {induced_error_cos_max:.6f} | "
                f"InducedError_Cos_Min: {induced_error_cos_min:.6f} | "
                f"InducedError_Cos_Mean: {induced_error_cos_mean:.6f} | "


                f"Proxy_Update_Norm: {proxy_update_norm:10.6f} | "
                f"Clean_Update_Norm: {clean_update_norm:10.6f} | "
                f"Proxy_Gradient_Norm: {proxy_gradient_norm:10.6f} | "
                f"Proxy_Server_Update_Norm: {proxy_server_update_norm:10.6f} | "
                f"Clean_Server_Update_Norm: {clean_server_update_norm:10.6f} | "
                f"Finetune_Update_Norm: {finetune_update_norm:10.6f} | "


                f"Induced_Error_Norm: {induced_error_norm:10.6f} | "
                f"Server_Induced_Error_Norm: {server_induced_error_norm:10.6f} | "
                f"Node_Update_Diff_Norm: {node_update_diff_norm:10.6f} | "
                f"Mean_Update_Diff_Norm: {mean_update_diff_norm:10.6f} | "


                f"Cos_ServerInducedError_InducedError: {cos_server_induced_induced:.6f} | "
                f"Cos_ServerInducedError_Noise: {cos_server_induced_noise:.6f} | "
                f"Cos_ServerInducedError_ProxyUpdate: {cos_server_induced_proxy:.6f} | "
                f"Cos_ServerInducedError_CleanUpdate: {cos_server_induced_clean:.6f} | "
                f"Cos_ServerInducedError_ProxyServerUpdate: {cos_server_induced_proxy_server:.6f} | "
                f"Cos_ServerInducedError_CleanServerUpdate: {cos_server_induced_clean_server:.6f} | "

                f"Cos_InducedError_Noise: {cos_induced_noise:.6f} | "
                f"Cos_InducedError_ProxyUpdate: {cos_induced_proxy:.6f} | "
                f"Cos_InducedError_CleanUpdate: {cos_induced_clean:.6f} | "
                f"Cos_InducedError_ProxyServerUpdate: {cos_induced_proxy_server:.6f} | "
                f"Cos_InducedError_CleanServerUpdate: {cos_induced_clean_server:.6f} | "

                f"Cos_ProxyGradient_ProxyUpdate: {cos_proxy_gradient_proxy:.6f} | "
                f"Cos_ProxyGradient_CleanUpdate: {cos_proxy_gradient_clean:.6f} | "

                f"Cos_CleanUpdate_ProxyUpdate: {cos_clean_proxy:.6f} | "
                f"Cos_CleanUpdate_Noise: {cos_clean_noise:.6f} | "
                f"Cos_CleanUpdate_ProxyServerUpdate: {cos_clean_proxy_server:.6f} | "
                f"Cos_CleanUpdate_CleanServerUpdate: {cos_clean_clean_server:.6f} | "

                f"Cos_ProxyUpdate_Noise: {cos_proxy_noise:.6f} | "
                f"Cos_ProxyUpdate_ProxyServerUpdate: {cos_proxy_proxy_server:.6f} | "
                f"Cos_ProxyUpdate_CleanServerUpdate: {cos_proxy_clean_server:.6f} | "

                f"Cos_Noise_ProxyServerUpdate: {cos_noise_proxy_server:.6f} | "
                f"Cos_Noise_CleanServerUpdate: {cos_noise_clean_server:.6f} | "

                f"Cos_ProxyServerUpdate_CleanServerUpdate: {cos_proxy_server_clean_server:.6f}\n"
            )
            f.write(log_line)




def write_dataset_analyze(dataset_analyze_txt_path, nodeindices, tracking_ids):
    train_dataset, _ = get_dataset(args)
    label_source = train_dataset.targets

    with open(dataset_analyze_txt_path, "w") as f:
        f.write(f"Tracking_IDs: {','.join(str(tracking_id) for tracking_id in tracking_ids)}\n")

        for node_id in sorted(nodeindices.keys()):
            counter = Counter()
            for idx in nodeindices[node_id]:
                label = int(label_source[int(idx)])
                counter[label] += 1

            label_summary = ", ".join(f"{label}:{counter[label]}" for label in sorted(counter))
            if not label_summary:
                label_summary = "unavailable"

            prefix = "Tracking Client" if node_id in tracking_ids else "Client"
            f.write(
                f"[{prefix} {node_id}] total={len(nodeindices[node_id])} | "
                f"labels={label_summary}\n"
            )

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
        p = mp.Process(target=gpu_train_worker, args=(trainQ, train_ack_q, devices[i%n_devices], args))
        p.start()
        train_procs.append(p)
        time.sleep(0.1)






    server = Server(args)
    server.set_initial_model()
    nodeIDs = [i for i in range(args.nodes)]
    nodeindices = getNodeIndicies(nodeIDs, args.nodes, args)
    nodes = []
    for i, nodeID in enumerate(nodeIDs):
        nodes.append(Client(nodeID, args, nodeindices[nodeID]))





    experiment_dir = args.log_name
    if os.path.exists(experiment_dir):
        shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir)
    dataset_analyze_txt_path = os.path.join(experiment_dir, "dataset_analysis.txt")
    tracking_ids = [] if (args.final or args.norm) else [0,1,2]
    write_dataset_analyze(dataset_analyze_txt_path, nodeindices, tracking_ids)
    with open(os.path.join(experiment_dir, f"server.txt"), "w") as f: pass        
    if args.norm:
        with open(os.path.join(experiment_dir, "proxy_norm.txt"), "w") as f: pass
    for tracking_id in tracking_ids:
        with open(os.path.join(experiment_dir, f"analysis_{tracking_id}.txt"), "w") as f: pass        
        with open(os.path.join(experiment_dir, f"distortion_{tracking_id}.txt"), "w") as f: pass        
        with open(os.path.join(experiment_dir, f"proxy_{tracking_id}.txt"), "w") as f: pass        
        with open(os.path.join(experiment_dir, f"proxy_finetune_{tracking_id}.txt"), "w") as f: pass        
        with open(os.path.join(experiment_dir, f"proxy_collusion_before_{tracking_id}.txt"), "w") as f: pass        
        with open(os.path.join(experiment_dir, f"proxy_collusion_finetune_{tracking_id}.txt"), "w") as f: pass        
        with open(os.path.join(experiment_dir, f"proxy_collusion_after_{tracking_id}.txt"), "w") as f: pass        


    lr = args.lr
    local_epoch = args.local_epoch
    for roundIdx in range(1, args.round+1):
        train_count = 0
        test_count = 0
        cur_time = time.time()
        lr *= args.lr_decay
        if args.step_decay > 0 and roundIdx == args.step_decay:
            lr /= 10.0
            local_epoch = args.step_epoch
            print(f"[step_decay] round={roundIdx}, lr={lr:.6f}, local_epoch={local_epoch}")

        print(f"Round {roundIdx}", end=', ')

        n_trainees = int(len(nodeIDs) * args.fraction)
        selected_ids = random.sample(nodeIDs, n_trainees)
        trainees = [nodes[i] for i in selected_ids]

        if args.norm and roundIdx == args.round:
            clean_weight = get_model_sd(server.model)
            centered_noise = None
            if args.DP == 'ours' and args.sigma > 0 and args.DP_i in ('center', 'center_0', 'center_m', 'center_M'):
                centered_noise = server.center_noise(roundIdx, selected_ids, mode=args.DP_i)
            test_count += queue_norm_proxy_evaluations(
                roundIdx=roundIdx,
                proxy_ids=selected_ids,
                clean_weight=clean_weight,
                server=server,
                centered_noise=centered_noise,
                proxy_norm_txt_path=os.path.join(experiment_dir, "proxy_norm.txt"),
                test_q=test_q,
                test_ack_q=test_ack_q,
                max_pending=max(1, n_devices * 2)
            )
            print(f"Elapsed Time : {time.time()-cur_time:.1f}")
            time.sleep(1.0)
            for _ in range(test_count):
                test_ack_q.get()
            time.sleep(2.0)
            continue

        is_final_round = args.final and (roundIdx == args.round)
        is_tracking_round = (not args.final) and (not args.norm) and any(node.nodeID in tracking_ids for node in trainees)
        
        proxy_weights = {}
        extra_finetune_ids = []
        clean_weight = get_model_sd(server.model)
        if args.final and is_final_round:
            test_q.put({
                'round': roundIdx,
                'weight': clean_weight,
                'worker_type': 'server'
            })
            test_count += 1
        centered_noise = None
        if (not args.norm) and args.DP == 'ours' and args.DP_i in ('center', 'center_0', 'center_m', 'center_M') and roundIdx != 1:
            centered_noise = server.center_noise(roundIdx, selected_ids, mode=args.DP_i)

        if args.norm:
            for node in trainees:
                proxy_weights[node.nodeID] = clean_weight
        else:
            for node in trainees:
                if args.final:
                    current_tracking_ids = []
                else:
                    current_tracking_ids = tracking_ids

                proxy_sd = server.get_model(roundIdx, node.nodeID, current_tracking_ids, centered_noise)
                proxy_weights[node.nodeID] = proxy_sd

        if is_tracking_round:
            candidate_ids = []
            for node_id in selected_ids:
                if node_id not in tracking_ids:
                    candidate_ids.append(node_id)
            if len(candidate_ids) >= 2:
                extra_finetune_ids = random.sample(candidate_ids, 2)
            else:
                extra_finetune_ids = candidate_ids

        if (not args.final) and is_tracking_round and len(proxy_weights) >= 3:
            for node in trainees:
                if node.nodeID in tracking_ids:
                    other_ids = [other_id for other_id in proxy_weights.keys() if other_id != node.nodeID]
                    if len(other_ids) < 2:
                        continue
                    collusion_before_sd = {}
                    collusion_before_ids = [node.nodeID] + random.sample(other_ids, 2)
                    for key in proxy_weights[node.nodeID].keys():
                        weight0 = proxy_weights[collusion_before_ids[0]][key]
                        weight1 = proxy_weights[collusion_before_ids[1]][key]
                        weight2 = proxy_weights[collusion_before_ids[2]][key]
                        collusion_before_sd[key] = ((weight0 + weight1 + weight2) / 3.0).clone()
                    test_q.put({
                        'round': roundIdx,
                        'weight': collusion_before_sd,
                        'worker_type': f'proxy_collusion_before_{node.nodeID}'
                    })
                    test_count += 1

        for node in trainees:
            if args.final:
                should_finetune = is_final_round
                virtual_weight = None
            elif args.norm:
                should_finetune = False
                virtual_weight = None
            else:
                should_finetune = (node.nodeID in tracking_ids) or (node.nodeID in extra_finetune_ids)
                virtual_weight = clean_weight if is_tracking_round else None

            trainQ.put({
                'type': 'train', 
                'node': node, 
                'lr': lr, 
                'local_epoch': local_epoch,
                'weight': proxy_weights[node.nodeID], 
                'round': roundIdx, 
                'finetune': should_finetune,
                'virtual_weight': virtual_weight
            })
            train_count += 1

        if 'L' in args.DP_s:
            server.reset_proxy_gradient()

        proxy_updates = {}
        clean_updates = {}
        tracking_finetunes = {}
        tracking_results = []
        final_finetunes = {}
        final_proxy_trains = {}
        for _ in range(train_count):
            msg = train_ack_q.get()
            node_id = msg['id']
            proxy_update = cpu_clone_sd(msg['weight'])
            clean_update = cpu_clone_sd(msg['virtual_weight'])
            proxy_finetune = cpu_clone_sd(msg['finetune_weight'])

            if is_tracking_round:
                proxy_updates[node_id] = proxy_update
                clean_updates[node_id] = clean_update
                if proxy_finetune is not None:
                    tracking_finetunes[node_id] = proxy_finetune

            server.update_node_info(proxy_update, proxy_weights[node_id], clean_weight, node_id)
            if is_tracking_round and node_id in tracking_ids:
                proxy_gradient = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in server.proxy_gradient[node_id].items()
                }
                tracking_results.append((node_id, proxy_update, clean_update, proxy_finetune, proxy_gradient))
            if is_final_round:
                final_finetunes[node_id] = proxy_finetune
                final_proxy_trains[node_id] = proxy_update
            del msg

        time.sleep(2.0)

        if args.final:
            if is_final_round:
                time.sleep(1.0)
                test_count += evaluate_final(
                    roundIdx=roundIdx,
                    selected_ids=selected_ids,
                    proxy_weights=proxy_weights,
                    proxy_trains=final_proxy_trains,
                    proxy_finetunes=final_finetunes,
                    test_q=test_q
                )
        elif is_tracking_round:
            time.sleep(1.0)
            for node_id, proxy_update, clean_update, proxy_finetune, proxy_gradient in tracking_results:
                test_q.put({'round': roundIdx, 'weight': proxy_weights[node_id], 'worker_type': f'proxy_{node_id}'})
                test_count += 1
                test_q.put({'round': roundIdx, 'weight': proxy_update, 'worker_type': f'proxy_train_{node_id}'})
                test_count += 1
                test_q.put({'round': roundIdx, 'weight': proxy_finetune, 'worker_type': f'proxy_finetune_{node_id}'})
                test_count += 1

                analyze(roundIdx, node_id,
                    clean_weight=clean_weight,
                    proxy_weights=proxy_weights,
                    proxy_updates=proxy_updates,
                    clean_updates=clean_updates,
                    proxy_finetune=proxy_finetune,
                    proxy_gradient=proxy_gradient)

                if len(tracking_finetunes) >= 3:
                    collusion_finetune_sd = {}
                    collusion_finetune_ids = random.sample(list(tracking_finetunes.keys()), 3)
                    for key in proxy_weights[node_id].keys():
                        weight0 = tracking_finetunes[collusion_finetune_ids[0]][key]
                        weight1 = tracking_finetunes[collusion_finetune_ids[1]][key]
                        weight2 = tracking_finetunes[collusion_finetune_ids[2]][key]
                        collusion_finetune_sd[key] = ((weight0 + weight1 + weight2) / 3.0).clone()
                    test_q.put({
                        'round': roundIdx,
                        'weight': collusion_finetune_sd,
                        'worker_type': f'proxy_collusion_finetune_{node_id}'
                    })
                    test_count += 1

                if len(proxy_updates) >= 3:
                    collusion_after_sd = {}
                    collusion_after_ids = random.sample(list(proxy_updates.keys()), 3)
                    for key in proxy_weights[node_id].keys():
                        weight0 = proxy_updates[collusion_after_ids[0]][key]
                        weight1 = proxy_updates[collusion_after_ids[1]][key]
                        weight2 = proxy_updates[collusion_after_ids[2]][key]
                        collusion_after_sd[key] = ((weight0 + weight1 + weight2) / 3.0).clone()
                    test_q.put({
                        'round': roundIdx,
                        'weight': collusion_after_sd,
                        'worker_type': f'proxy_collusion_after_{node_id}'
                    })
                    test_count += 1

        server.avg_parameters(roundIdx)
        print(f"Elapsed Time : {time.time()-cur_time:.1f}")
        interval = 5
        if args.round < 30:
            interval = 2
        if (not args.final) and (not args.norm) and roundIdx % interval == 0:
            time.sleep(1.0)
            test_q.put({
                'round': roundIdx, 
                'weight': get_model_sd(server.model), 
                'worker_type': 'server'
            })
            test_count += 1

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
