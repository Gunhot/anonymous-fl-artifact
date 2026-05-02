import os
import sys
import torch
import numpy as np
import uuid
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from dataLoader.dataset import get_dataset

# [Gemini] Log Suppression
import warnings
from transformers import logging as tr_logging
warnings.filterwarnings("ignore", message=".*fan_in_fan_out.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

tr_logging.set_verbosity_error()

# =============================================================================
# [0] Constants & Helpers
# =============================================================================

ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(ARTIFACT_DIR, 'model')
GPT2S_PATH = os.path.join(MODEL_DIR, 'gpt2')
DISTILBERT_PATH = os.path.join(MODEL_DIR, 'distilbert-base-uncased')
E2E_METRICS_SCRIPT = os.path.join(os.path.dirname(__file__), 'e2e-metrics', 'measure_scores.py')

def seed_process(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_model_sd(model):
    sd = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            sd[name] = param.detach().cpu().clone()
    return sd


def _get_ipc_dir(args):
    ipc_dir = os.path.join(args.log_name, "_ipc")
    os.makedirs(ipc_dir, exist_ok=True)
    return ipc_dir


def save_ipc_state_dict(state_dict, args, prefix):
    if state_dict is None:
        return None
    ipc_dir = _get_ipc_dir(args)
    path = os.path.join(ipc_dir, f"{prefix}_{uuid.uuid4().hex}.pt")
    torch.save(state_dict, path)
    return path


def load_ipc_state_dict(payload):
    if payload is None:
        return None
    if isinstance(payload, str):
        return torch.load(payload, map_location="cpu")
    return payload


def cleanup_ipc_path(path):
    if not isinstance(path, str):
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def set_model_sd(model, weights):
    keys = model.load_state_dict(weights, strict=False)
    if len(keys.unexpected_keys) > 0:
        print(f"[Warning] Unexpected keys in weights: {keys.unexpected_keys}")
    return model

def build_model(args):
    model_type = args.model.lower()

    is_gpt2s = 'gpt2s' in model_type
    is_distilbert = 'distilbert' in model_type

    if is_gpt2s:
        target_path = GPT2S_PATH
    elif is_distilbert:
        target_path = DISTILBERT_PATH
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        target_path, local_files_only=True
    )

    # Pad token handling
    if args.dataset == "e2e":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    # 1: load pretrained weights, 0: randomly initialize from config
    use_pretrained = getattr(args, "use_pretrained", 1)

    if args.dataset == '20news':
        num_labels = 20
        if use_pretrained:
            model = AutoModelForSequenceClassification.from_pretrained(
                target_path,
                num_labels=num_labels,
                torch_dtype=torch.float32,
                local_files_only=True
            )
        else:
            config = AutoConfig.from_pretrained(target_path, local_files_only=True)
            config.num_labels = num_labels
            model = AutoModelForSequenceClassification.from_config(config)

        model.config.pad_token_id = tokenizer.pad_token_id
        task_type = "SEQ_CLS"

    elif args.dataset == "e2e":
        if use_pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                target_path,
                torch_dtype=torch.float32,
                local_files_only=True
            )
        else:
            config = AutoConfig.from_pretrained(target_path, local_files_only=True)
            model = AutoModelForCausalLM.from_config(config)

        model.config.pad_token_id = tokenizer.pad_token_id
        task_type = "CAUSAL_LM"
    else:
        raise ValueError(f"Unsupported dataset for build_model: {args.dataset}")

    # LoRA Configuration
    if 'lora' in model_type:
        from peft import get_peft_model, LoraConfig, TaskType

        task_type = getattr(TaskType, task_type)
        if 'distilbert' in model_type:
            target_modules = ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
            modules_to_save = ["sa_layer_norm", "output_layer_norm", "pre_classifier", "classifier"]
        else: # GPT-2
            target_modules = ["c_attn", "c_proj", "c_fc"]
            modules_to_save = []

        peft_config = LoraConfig(
            task_type=task_type, 
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        
        model = get_peft_model(model, peft_config)
    
    return model, tokenizer


def gpu_train_worker(trainQ, train_ack_q, device, args):
    dataset, _, _ = get_dataset(args)

    # Instantiate models once and reuse
    train_model, _ = build_model(args)

    while True:
        msg = trainQ.get()

        if msg == 'kill':
            break

        elif msg['type'] == 'train':
            processing_node = msg['node']
            weight_payload = msg['weight']
            weight_sd = load_ipc_state_dict(weight_payload)
            set_model_sd(train_model, weight_sd)
            round = msg['round']
            finetune = msg['finetune']
            virtual_payload = msg['virtual_weight']
            virtual_sd = load_ipc_state_dict(virtual_payload)

            model_weight, finetune_weight, virtual_weight = processing_node.train(
                device, msg['lr'], train_model, dataset, round,
                finetune=finetune, virtual_sd=virtual_sd
            )

            result = {
                'weight': save_ipc_state_dict(model_weight, args, f"train_out_r{round}_n{processing_node.nodeID}_weight"),
                'finetune_weight': save_ipc_state_dict(finetune_weight, args, f"train_out_r{round}_n{processing_node.nodeID}_finetune"),
                'virtual_weight': save_ipc_state_dict(virtual_weight, args, f"train_out_r{round}_n{processing_node.nodeID}_virtual"),
                'id': processing_node.nodeID
            }

            train_ack_q.put(result)

            if msg.get('cleanup_weight', False):
                cleanup_ipc_path(weight_payload)
            del model_weight
            del finetune_weight
            del virtual_weight
            del weight_sd
            del virtual_sd
            del result

        del msg


def evaluate_classification(model, test_loader, args, device, roundIdx, type):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

        preds = preds.cpu()
        targets = targets.cpu()
        correct += (preds == targets).sum().item()
        total += len(targets)

    accuracy = correct / total if total > 0 else 0.0
    log_str = f"Round {roundIdx} | Accuracy: {accuracy:.4f}\n"
    print(f"[{type}] {log_str}", end='')

    with open(os.path.join(args.log_name, f"{type}.txt"), "a") as f:
        f.write(log_str)

def evaluate_generation(model, tokenizer, test_dataset, args, device, roundIdx, type):
    import subprocess

    model.to(device)
    model.eval()

    total_results = []
    total_len = len(test_dataset)

    for start in range(0, total_len, args.batch_size):
        batch_raw = test_dataset[start:start + args.batch_size]

        input_ids = batch_raw['input_ids'].to(device)
        attention_mask = batch_raw['attention_mask'].to(device)
        sources = batch_raw['source']
        targets = batch_raw['target']

        input_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=5,
                do_sample=False,
                no_repeat_ngram_size=3,
            )

        generated_tokens = outputs[:, input_len:]
        decoded_gens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for i in range(len(decoded_gens)):
            total_results.append({
                'source': sources[i],
                'gen': decoded_gens[i].strip(),
                'target': targets[i]
            })

    preds = [x['gen'] for x in total_results]
    refs = [x['target'] for x in total_results]
    sources = [x['source'] for x in total_results]

    gen_file = os.path.join(args.log_name, f"{type}_gen_round{roundIdx}.txt")
    ref_file = os.path.join(args.log_name, f"{type}_ref_round{roundIdx}.txt")
    src_file = os.path.join(args.log_name, f"{type}_src_round{roundIdx}.txt")

    with open(gen_file, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(str(p).replace('\n', ' ').strip() + '\n')

    with open(ref_file, 'w', encoding='utf-8') as f:
        for r in refs:
            if isinstance(r, str):
                r = [r]
            for sub_r in r:
                f.write(str(sub_r).replace('\n', ' ').strip() + '\n')
            f.write('\n')

    with open(src_file, 'w', encoding='utf-8') as f:
        for s in sources:
            f.write(str(s).replace('\n', ' ').strip() + '\n')

    all_empty_preds = len(preds) > 0 and all((not p.strip()) for p in preds)
    if all_empty_preds:
        log_str = (
            f"Round {roundIdx} | "
            f"BLEU: 0.0000 | "
            f"NIST: 0.0000 | "
            f"METEOR: 0.0000 | "
            f"ROUGE-L: 0.0000 | "
            f"CIDEr: 0.0000\n"
        )
        print(f"[{type}] {log_str}")
        with open(os.path.join(args.log_name, f"{type}.txt"), "a") as f:
            f.write(log_str)
        return

    cmd = [sys.executable, E2E_METRICS_SCRIPT, ref_file, gen_file, "-p"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output_str = result.stdout
    scores = {}

    for line in output_str.strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            scores[key.strip()] = val.strip()

    log_str = (
        f"Round {roundIdx} | "
        f"BLEU: {float(scores.get('BLEU', '0.0')) * 100:.4f} | "
        f"NIST: {float(scores.get('NIST', '0.0')):.4f} | "
        f"METEOR: {float(scores.get('METEOR', '0.0')) * 100:.4f} | "
        f"ROUGE-L: {float(scores.get('ROUGE_L', '0.0')) * 100:.4f} | "
        f"CIDEr: {float(scores.get('CIDEr', '0.0')):.4f}\n"
    )

    print(f"[{type}] {log_str}", end='')
    with open(os.path.join(args.log_name, f"{type}.txt"), "a") as f:
        f.write(log_str)

def gpu_test_worker(testQ, ackQ, device, args, worker_type="server"):
    _, test_dataset, tokenizer = get_dataset(args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    local_model, _ = build_model(args)
    local_model.to(device)

    while True:
        msg = testQ.get()

        if msg == 'kill':
            break
        elif isinstance(msg, str):
            pass
        elif isinstance(msg, dict) and 'weight' in msg:
            model_payload = msg['weight']
            model_sd = load_ipc_state_dict(model_payload)
            roundIdx = msg['round']
            current_worker_type = msg.get('worker_type', worker_type)

            try:
                set_model_sd(local_model, model_sd)

                if args.dataset == 'e2e':
                    evaluate_generation(local_model, tokenizer, test_dataset, args, device, roundIdx, current_worker_type)

                elif args.dataset == '20news':
                    evaluate_classification(local_model, test_loader, args, device, roundIdx, current_worker_type)
            except Exception as exc:
                print(
                    f"[test_worker] round={roundIdx} type={current_worker_type} failed: {exc}",
                    flush=True
                )
            finally:
                ackQ.put(True)
                if msg.get('cleanup_weight', False):
                    cleanup_ipc_path(model_payload)
                del model_sd

        del msg
