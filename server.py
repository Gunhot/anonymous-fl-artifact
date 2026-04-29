import hashlib
import os

import numpy as np
import torch

from workers import build_model, get_model_sd, set_model_sd


def model_avg(param_sum, param_count, args, h, origin, round_idx=None):
    # Use dictionary comprehension for faster cloning of the state dict tensors
    w_avg = {k: v.clone() if v is not None else None for k, v in param_sum.items()}

    with torch.no_grad():
        for k in w_avg.keys():
            if param_count[k] == 0:
                w_avg[k] = origin[k].clone()
                continue

            w_avg[k] = torch.div(w_avg[k], param_count[k])

    return w_avg


class Server:
    def __init__(self, args):
        self.model = None
        self.param_sum = dict()
        self.param_count = dict()
        self.args = args
        self.proxy_gradient = [dict() for _ in range(args.nodes)]
        self.global_proxy_gradient = dict()
        self.original_model_sd = None
        self.global_init_sd = None

    def _apply_fedqsn_logic(self, model_sd, round_idx, client_id, tracking_ids=None):
        tracking_ids = set(tracking_ids or [])
        proxy_sd = {name: param.detach().clone() for name, param in model_sd.items()}
        with torch.no_grad():
            for name, param in proxy_sd.items():
                original_param = param.float()
                target_param = original_param.clone()
                device = target_param.device

                p1 = float(self.args.p1)
                p2 = float(self.args.p2)
                omega_val = int(self.args.omega)
                if param.dim() < 2 or 'weight' not in name:
                    p1 = 0.0
                    p2 = 0.0
                    omega_val = 0
                p1 = min(max(p1, 0.0), 0.999999)
                p2 = min(max(p2, 0.0), 0.999999)

                if p1 > 0 and target_param.numel() > 0:
                    flat_view = target_param.reshape(target_param.size(0), -1)
                    if name not in self.server_masks or self.server_masks[name].shape[1] != flat_view.size(1):
                        h = hashlib.md5(name.encode()).hexdigest()
                        name_hash = int(h[:8], 16)
                        layer_unique_seed = (self.args.seed * 1000003 + name_hash) % (2**31 - 1)

                        server_rng = torch.Generator(device=device)
                        server_rng.manual_seed(layer_unique_seed)

                        mask_col = torch.bernoulli(
                            torch.full(
                                (1, flat_view.size(1)),
                                1.0 - p1,
                                device=device,
                                dtype=target_param.dtype
                            ),
                            generator=server_rng
                        )
                        self.server_masks[name] = mask_col

                    server_mask = self.server_masks[name].to(
                        device=device,
                        dtype=target_param.dtype
                    ).expand_as(flat_view).reshape_as(target_param)
                    target_param = target_param * server_mask

                if p2 > 0 and target_param.numel() > 0:
                    r_idx = round_idx if round_idx is not None else 0
                    c_id = client_id if client_id is not None else 0

                    h = hashlib.md5(name.encode()).hexdigest()
                    name_hash = int(h[:8], 16)
                    if self.args.qsn_fixed_mask:
                        client_seed = (self.args.seed * 1000003 + 7919 + name_hash) % (2**31 - 1)
                    else:
                        client_seed = (self.args.seed * 1000003 + r_idx * 100003 + c_id + name_hash) % (2**31 - 1)

                    client_rng = torch.Generator(device=device)
                    client_rng.manual_seed(client_seed)

                    flat_view = target_param.reshape(target_param.size(0), -1)
                    mask_col = torch.bernoulli(
                        torch.full(
                            (1, flat_view.size(1)),
                            1.0 - p2,
                            device=device,
                            dtype=target_param.dtype
                        ),
                        generator=client_rng
                    )
                    client_mask = mask_col.expand_as(flat_view).reshape_as(target_param)
                    target_param = target_param * client_mask

                if omega_val > 0:
                    block_size = 256
                    flat_param = target_param.flatten()

                    pad_len = (block_size - (flat_param.numel() % block_size)) % block_size
                    if pad_len > 0:
                        flat_param = torch.nn.functional.pad(flat_param, (0, pad_len))

                    blocks = flat_param.view(-1, block_size)
                    absmax = torch.clamp(
                        blocks.abs().max(dim=1, keepdim=True).values,
                        min=1e-8
                    )

                    if omega_val == 1:
                        quantized = torch.sign(blocks) * absmax
                    else:
                        quant_range = (2 ** (omega_val - 1)) - 1
                        scale = quant_range / absmax
                        q_int = torch.clamp(
                            torch.round(blocks * scale),
                            -quant_range,
                            quant_range
                        )
                        quantized = q_int / scale

                    quantized_flat = quantized.view(-1)
                    if pad_len > 0:
                        quantized_flat = quantized_flat[:-pad_len]

                    proxy_final = quantized_flat.view(original_param.size())
                else:
                    proxy_final = target_param

                proxy_sd[name] = proxy_final.to(param.dtype)

                if client_id in tracking_ids:
                    addition = proxy_final - original_param
                    w_norm = torch.norm(original_param).item()
                    addition_norm = torch.norm(addition).item()
                    distortion_log_path = os.path.join(self.args.log_name, f"distortion_{client_id}.txt")
                    with open(distortion_log_path, "a") as f:
                        w_ratio = addition_norm / (w_norm + 1e-8)
                        g_norm = 0.0
                        if hasattr(self, 'scale') and client_id is not None and 0 <= client_id < len(self.scale):
                            if name in self.scale[client_id]:
                                g_norm = torch.norm(self.scale[client_id][name].float()).item()
                        if g_norm > 1e-8:
                            g_ratio = addition_norm / (g_norm + 1e-8)
                        else:
                            g_ratio = -1.0
                        f.write(
                            f"(Round: {round_idx})[Layer: {name}] "
                            f"W_Ratio: {w_ratio:.6f} | G_Ratio: {g_ratio:.6f} | "
                            f"Noise_Norm: {addition_norm:.4e} | Weight_Norm: {w_norm:.4e} | Grad_Norm: {g_norm:.4e}\n"
                        )

        return proxy_sd

    def _apply_fedlpp_logic(self, model_sd, round_idx, client_id):
        proxy_sd = {name: param.detach().clone() for name, param in model_sd.items()}
        with torch.no_grad():
            for name, param in proxy_sd.items():
                if self.args.omega <= 0:
                    continue

                original_param = param.float()
                # [Filter] Scope Check - Override omega for small dimensions
                w = int(self.args.omega) if param.dim() >= 2 else 0

                if w > 0:
                    # -------------------------------------------------------
                    # A. Define Standard Values (V)
                    # -------------------------------------------------------
                    if w == 2:
                        V = torch.tensor([-1.00, 0.00, 0.33, 1.00], device=original_param.device)
                    elif w == 1:
                        V = torch.tensor([-1.00, 0.00, 1.00], device=original_param.device)
                    elif w == 3:
                        V = torch.tensor([-1.00, -0.47, -0.21, 0.00, 0.16, 0.33, 0.56, 1.00], device=original_param.device)
                    elif w == 4:
                        # QLoRA NF4 codebook (16 levels)
                        V = torch.tensor([
                            -1.0, -0.6961928009986877, -0.5250730514526367,
                            -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                            -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                            0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
                            0.5626170039176941, 0.7229568362236023, 1.0
                        ], device=original_param.device)

                    # -------------------------------------------------------
                    # B. Block-wise Nearest Neighbor
                    # -------------------------------------------------------
                    block_size = 256
                    flat_param = original_param.flatten()  # LPP는 마스킹 없이 원본 사용

                    pad_len = (block_size - (flat_param.numel() % block_size)) % block_size
                    if pad_len > 0:
                        flat_param = torch.nn.functional.pad(flat_param, (0, pad_len))

                    blocks = flat_param.view(-1, block_size)

                    # Normalize
                    z_l = torch.where(
                        blocks.abs().max(dim=1, keepdim=True).values == 0,
                        torch.ones(1, device=blocks.device),
                        blocks.abs().max(dim=1, keepdim=True).values
                    )
                    normalized = blocks / z_l

                    # Find Nearest
                    diff = normalized.unsqueeze(-1) - V
                    min_indices = diff.abs().argmin(dim=-1)
                    quantized_blocks = V[min_indices] * z_l

                    # Restore Shape
                    quantized_flat = quantized_blocks.view(-1)
                    if pad_len > 0:
                        quantized_flat = quantized_flat[:-pad_len]

                    proxy_final = quantized_flat.view(original_param.size())

                    # Update
                    proxy_sd[name] = proxy_final.to(param.dtype)
                else:
                    proxy_final = original_param

                if client_id == 0:
                    addition = proxy_final.to(param.dtype) - original_param.to(param.dtype)
                    w_norm = torch.norm(original_param).item()
                    addition_norm = torch.norm(addition).item()
                    distortion_log_path = os.path.join(self.args.log_name, "distortion.txt")
                    with open(distortion_log_path, "a") as f:
                        ratio = addition_norm / (w_norm + 1e-8)
                        w_ratio = addition_norm / (w_norm + 1e-8)
                        g_norm = 0.0
                        if hasattr(self, 'scale') and client_id is not None and 0 <= client_id < len(self.scale):
                            if name in self.scale[client_id]:
                                g_norm = torch.norm(self.scale[client_id][name]).item()
                        if g_norm > 1e-8:
                            g_ratio = addition_norm / (g_norm + 1e-8)
                        else:
                            g_ratio = -1.0
                        f.write(f"(Round: {round_idx})[Layer: {name}] W_Ratio: {w_ratio:.6f} | G_Ratio: {g_ratio:.6f} | Noise_Norm: {addition_norm:.4e} | Weight_Norm: {w_norm:.4e} | Grad_Norm: {g_norm:.4e}\n")

        return proxy_sd

    def _build_get_model_rngs(self, round_idx, client_id):
        round_idx = 0 if round_idx is None else int(round_idx)
        client_id = 0 if client_id is None else int(client_id)

        base_seed = int((self.args.seed * 1000003 + round_idx * 100003 + client_id) % (2**31 - 1))
        rng_param = torch.Generator()
        rng_param.manual_seed(base_seed)

        np_rng = np.random.default_rng(base_seed)
        return round_idx, client_id, rng_param, np_rng

    def _has_nonzero(self, bank, client_id):
        for _, tensor in bank[client_id].items():
            if torch.count_nonzero(tensor).item() != 0:
                return True
        return False

    def _get_nonzero_ids(self, bank):
        nonzero_ids = []
        for client_id in range(self.args.nodes):
            if self._has_nonzero(bank, client_id):
                nonzero_ids.append(client_id)
        return nonzero_ids

    def reset_proxy_gradient(self):
        for client_proxy_gradient in self.proxy_gradient:
            for name in client_proxy_gradient.keys():
                client_proxy_gradient[name].zero_()

    def center_noise(self, round_idx, target_client_ids, mode='center'):
        raw_noise = {}
        proxy_gradient_source_ids = {}
        mean_noise = {}
        valid_proxy_gradient_ids = self._get_nonzero_ids(self.proxy_gradient)

        for client_id in target_client_ids:
            _, _, _, np_rng = self._build_get_model_rngs(round_idx, client_id)

            proxy_gradient_source_id = int(np_rng.choice(valid_proxy_gradient_ids))
            proxy_gradient_source_ids[client_id] = proxy_gradient_source_id

            raw_noise[client_id] = {}
            for name, drift_tensor in self.proxy_gradient[proxy_gradient_source_id].items():
                raw_noise[client_id][name] = drift_tensor.detach().clone().float()

        ref_client_id = target_client_ids[0]
        for name in raw_noise[ref_client_id].keys():
            mean_tensor = torch.zeros_like(raw_noise[ref_client_id][name])
            for client_id in target_client_ids:
                mean_tensor += raw_noise[client_id][name]
            mean_noise[name] = mean_tensor / len(target_client_ids)

        centered_noise = {}
        for client_id in target_client_ids:
            centered_noise[client_id] = {
                'proxy_gradient_source_id': proxy_gradient_source_ids[client_id],
                'noise': {}
            }
            for name, drift_tensor in raw_noise[client_id].items():
                centered_noise[client_id]['noise'][name] = drift_tensor - mean_noise[name]

        if mode in ('center_0', 'center_m', 'center_M'):
            if mode == 'center_0':
                common_client_id = target_client_ids[0]
            else:
                cosine_scores = {}
                for client_id in target_client_ids:
                    dot = 0.0
                    client_norm_sq = 0.0
                    mean_norm_sq = 0.0
                    for name, client_tensor in raw_noise[client_id].items():
                        mean_tensor = mean_noise[name]
                        dot += torch.sum(client_tensor * mean_tensor).item()
                        client_norm_sq += torch.sum(client_tensor * client_tensor).item()
                        mean_norm_sq += torch.sum(mean_tensor * mean_tensor).item()
                    denom = (client_norm_sq ** 0.5) * (mean_norm_sq ** 0.5)
                    cosine_scores[client_id] = 0.0 if denom <= 1e-12 else dot / denom

                chooser = min if mode == 'center_m' else max
                common_client_id = chooser(
                    target_client_ids,
                    key=lambda client_id: cosine_scores[client_id]
                )

            common_source_id = proxy_gradient_source_ids[common_client_id]
            common_direction = centered_noise[common_client_id]['noise']
            for client_id in target_client_ids:
                centered_noise[client_id]['proxy_gradient_source_id'] = common_source_id
                centered_noise[client_id]['noise'] = {
                    name: tensor.detach().clone()
                    for name, tensor in common_direction.items()
                }

        return centered_noise

    def get_model(self, round_idx=None, client_id=None, tracking_ids=None, centered_noise=None, force_proxy=False):
        tracking_ids = set(tracking_ids or [])
        model_sd = get_model_sd(self.model)
        round_idx, client_id, rng_param, np_rng = self._build_get_model_rngs(round_idx, client_id)

        if self.args.DP == 'qsn':
            return self._apply_fedqsn_logic(model_sd, round_idx, client_id, tracking_ids)
        if self.args.DP == 'lpp':
            return self._apply_fedlpp_logic(model_sd, round_idx, client_id)

        if round_idx == 1 and not force_proxy:
            return model_sd

        sigma = self.args.sigma
        if sigma <= 0 and self.args.stats <= 0:
            return model_sd

        proxy_sd = {name: param.detach().clone() for name, param in model_sd.items()}
        with torch.no_grad():
            if sigma > 0:
                gausg_scale_reference_id = client_id
                if self.args.DP == 'gausg' and not self._has_nonzero(self.scale, client_id):
                    valid_scale_ids = self._get_nonzero_ids(self.scale)
                    if len(valid_scale_ids) > 0:
                        gausg_scale_reference_id = int(np_rng.choice(valid_scale_ids))

                if self.args.DP == 'ours':
                    dp_i = self.args.DP_i
                    ours_proxy_gradient_id = None
                    ours_scale_reference_id = None

                    if dp_i == 'my':
                        ours_proxy_gradient_id = client_id
                        ours_scale_reference_id = client_id

                    elif dp_i == 'wrong':
                        ours_proxy_gradient_id = (client_id + 1) % self.args.nodes
                        ours_scale_reference_id = client_id

                    elif dp_i == 'random':
                        valid_proxy_gradient_ids = self._get_nonzero_ids(self.proxy_gradient)
                        ours_proxy_gradient_id = int(np_rng.choice(valid_proxy_gradient_ids))
                        ours_scale_reference_id = client_id

                    elif dp_i in ('center', 'center_0', 'center_m', 'center_M'):
                        ours_proxy_gradient_id = centered_noise[client_id]['proxy_gradient_source_id']
                        ours_scale_reference_id = client_id

                    elif dp_i == 'global':
                        ours_proxy_gradient_id = 'global'
                        ours_scale_reference_id = client_id

                    if not self._has_nonzero(self.scale, client_id):
                        if dp_i == 'global':
                            valid_scale_ids = self._get_nonzero_ids(self.scale)
                            if len(valid_scale_ids) > 0:
                                ours_scale_reference_id = int(np_rng.choice(valid_scale_ids))
                        else:
                            ours_scale_reference_id = ours_proxy_gradient_id

                for name, param in model_sd.items():
                    if 'weight' not in name and 'bias' not in name:
                        continue

                    original_param = param.float()
                    applied_scale = 0.0

                    if self.args.DP == 'gausg':
                        proxy_gradient_source_id = None

                        noise = torch.randn(
                            original_param.size(),
                            device=original_param.device,
                            dtype=torch.float32
                        )

                        scale_tensor = self.scale[gausg_scale_reference_id][name].to(original_param.device).float()
                        scale_norm = torch.norm(scale_tensor).item()
                        target_norm = scale_norm * (sigma / 10000.0)
                        noise_norm = torch.norm(noise)
                        applied_scale = (target_norm / (noise_norm + 1e-8)).item()
                        addition = noise * applied_scale

                    elif self.args.DP == 'ours':
                        proxy_gradient_source_id = ours_proxy_gradient_id
                        gausg_scale_reference_id = ours_scale_reference_id

                        if self.args.DP_i in ('center', 'center_0', 'center_m', 'center_M'):
                            proxy_vector = centered_noise[client_id]['noise'][name].to(original_param.device).float()
                        elif self.args.DP_i == 'global':
                            proxy_vector = self.global_proxy_gradient[name].to(original_param.device).float()
                        else:
                            drift_tensor = self.proxy_gradient[proxy_gradient_source_id][name]
                            proxy_vector = drift_tensor.to(original_param.device).float()

                        drift_norm = torch.norm(proxy_vector).item()

                        scale_tensor = self.scale[gausg_scale_reference_id][name].to(original_param.device).float()
                        scale_norm = torch.norm(scale_tensor).item()

                        applied_scale = sigma / 10000.0 * scale_norm / (drift_norm + 1e-8)
                        addition = proxy_vector * applied_scale

                    proxy_sd[name] = (original_param + addition).to(param.dtype)

                    if client_id in tracking_ids:
                        addition_norm = torch.norm(addition).item()
                        w_norm = torch.norm(original_param).item()
                        w_ratio = addition_norm / (w_norm + 1e-8)
                        g_norm = torch.norm(scale_tensor).item()
                        g_ratio = addition_norm / (g_norm + 1e-8)
                        distortion_log_path = os.path.join(self.args.log_name, f"distortion_{client_id}.txt")
                        with open(distortion_log_path, "a") as f:
                            f.write(
                                f"(Round: {round_idx})[Layer: {name}] "
                                f"DP_Mode: {self.args.DP} | "
                                f"Proxy_Gradient_Source_Id: {proxy_gradient_source_id} | "
                                f"Scale_Source_Id: {gausg_scale_reference_id} | "
                                f"Applied_Scale: {applied_scale:.4e} | "
                                f"W_Ratio: {w_ratio:.6f} | G_Ratio: {g_ratio:.6f} | "
                                f"Noise_Norm: {addition_norm:.4e} | Weight_Norm: {w_norm:.4e} | Grad_Norm: {g_norm:.4e}\n"
                            )

            if self.args.stats > 0:
                stats_sigma = self.args.stats
                rng_buf = torch.Generator()
                rng_buf.manual_seed(
                    int((self.args.seed * 1000003 + round_idx * 100003 + client_id + 1) % (2**31 - 1))
                )

                for name, buffer in model_sd.items():
                    if 'weight' in name or 'bias' in name or 'num_batches_tracked' in name:
                        continue

                    original_buffer = buffer.float()
                    distribution = torch.randn(
                        original_buffer.size(),
                        generator=rng_buf,
                        device=original_buffer.device,
                        dtype=torch.float32
                    )
                    std_dev = torch.abs(original_buffer) * stats_sigma / 100.0
                    addition = distribution * std_dev
                    proxy_sd[name] = (original_buffer + addition).to(buffer.dtype)
        return proxy_sd

    # ---------- 업로드 수집 ----------
    def update_node_info(self, weight, proxy_weight, clean_weight, node_id):
        scale_base_weight = clean_weight if 'c' in self.args.DP_n else proxy_weight
        proxy_gradient_base_weight = clean_weight if 'c' in self.args.DP_s else proxy_weight

        for k in weight.keys():
            w_k = weight[k].clone().detach()
            if self.args.noise_update == 1:
                w_k -= (proxy_weight[k] - clean_weight[k])

            if self.param_sum[k] is None:
                self.param_sum[k] = w_k
            else:
                self.param_sum[k] += w_k
            self.param_count[k] += 1

            if 'weight' in k or 'bias' in k:
                scale_update_tensor = (w_k - scale_base_weight[k]).clone()
                proxy_gradient_update_tensor = (w_k - proxy_gradient_base_weight[k]).clone()
                signed_update_tensor = (
                    proxy_gradient_update_tensor if '+' in self.args.DP_s else -proxy_gradient_update_tensor
                )
                self.scale[node_id][k] = scale_update_tensor
                if 'L' in self.args.DP_s:
                    self.proxy_gradient[node_id][k] = signed_update_tensor
                    continue

                if 'M' in self.args.DP_s:
                    beta = self.args.bank_beta
                    self.proxy_gradient[node_id][k] = beta * self.proxy_gradient[node_id][k] + signed_update_tensor
                elif 'E' in self.args.DP_s:
                    beta = self.args.bank_beta
                    self.proxy_gradient[node_id][k] = (
                        beta * self.proxy_gradient[node_id][k] + (1.0 - beta) * signed_update_tensor
                    )
                else:
                    self.proxy_gradient[node_id][k] += signed_update_tensor

    # ---------- 평균/업데이트 ----------
    def avg_parameters(self, round_idx):
        model_state = get_model_sd(self.model)
        h = {k: torch.zeros_like(v) for k, v in model_state.items()}

        avg_parameters = model_avg(self.param_sum, self.param_count, self.args, h, model_state, round_idx)
        if self.args.DP == 'qsn' and len(self.server_masks) > 0 and self.global_init_sd is not None:
            with torch.no_grad():
                for name, mask in self.server_masks.items():
                    if name not in avg_parameters:
                        continue
                    device = avg_parameters[name].device
                    dtype = avg_parameters[name].dtype
                    avg_flat = avg_parameters[name].reshape(avg_parameters[name].size(0), -1)
                    init_flat = self.global_init_sd[name].to(device=device, dtype=dtype).reshape_as(avg_flat)
                    expanded_mask = mask.to(device=device, dtype=dtype).expand_as(avg_flat)
                    avg_parameters[name] = (
                        avg_flat * expanded_mask +
                        init_flat * (1 - expanded_mask)
                    ).reshape_as(avg_parameters[name])

        if self.args.DP == 'ours' and self.args.DP_i == 'global':
            for k in avg_parameters.keys():
                if 'weight' not in k and 'bias' not in k:
                    continue

                server_update_tensor = (avg_parameters[k] - model_state[k]).clone()
                signed_update_tensor = (
                    server_update_tensor if '+' in self.args.DP_s else -server_update_tensor
                )

                if 'L' in self.args.DP_s:
                    self.global_proxy_gradient[k] = signed_update_tensor
                    continue

                if 'M' in self.args.DP_s:
                    beta = self.args.bank_beta
                    self.global_proxy_gradient[k] = beta * self.global_proxy_gradient[k] + signed_update_tensor
                elif 'E' in self.args.DP_s:
                    beta = self.args.bank_beta
                    self.global_proxy_gradient[k] = (
                        beta * self.global_proxy_gradient[k] + (1.0 - beta) * signed_update_tensor
                    )
                else:
                    self.global_proxy_gradient[k] += signed_update_tensor

        set_model_sd(self.model, avg_parameters)

        for k in model_state.keys():
            self.param_sum[k] = None
            self.param_count[k] = 0

    # ---------- 초기 서버 모델 ----------
    def set_initial_model(self):
        self.model = build_model(self.args)
        self.server_masks = {}
        model_sd = get_model_sd(self.model)
        self.original_model_sd = {k: v.detach().clone() for k, v in model_sd.items()}
        self.global_init_sd = {k: v.detach().clone() for k, v in model_sd.items()}

        self.param_sum = {k: None for k in model_sd.keys()}
        self.param_count = {k: 0 for k in model_sd.keys()}
        self.proxy_gradient = [
            {k: torch.zeros_like(v) for k, v in model_sd.items()}
            for _ in range(self.args.nodes)
        ]
        self.global_proxy_gradient = {k: torch.zeros_like(v) for k, v in model_sd.items()}
        self.scale = [
            {k: torch.zeros_like(v) for k, v in model_sd.items()}
            for _ in range(self.args.nodes)
        ]
