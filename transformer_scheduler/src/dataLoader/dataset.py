import os

import datasets
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BASE_DATA_DIR = os.path.join(ARTIFACT_DIR, 'data')
MODEL_DIR = os.path.join(ARTIFACT_DIR, 'model')
GPT2_PATH = os.path.join(MODEL_DIR, 'gpt2')
DISTILBERT_PATH = os.path.join(MODEL_DIR, 'distilbert-base-uncased')

datasets.logging.set_verbosity_error()


def tokenizer_path(args):
    model_name = args.model.lower()
    if 'distilbert' in model_name:
        return DISTILBERT_PATH
    if 'gpt2' in model_name:
        return GPT2_PATH
    raise ValueError(f"Unsupported model for tokenizer: {args.model}")


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path(args), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    return tokenizer


def get_20news(args):
    data_dir = os.path.join(BASE_DATA_DIR, '20news')
    dataset = load_from_disk(data_dir)

    tokenizer = load_tokenizer(args)
    tokenizer.padding_side = 'right'
    max_len = args.max_len

    def preprocess(examples):
        tokenized = tokenizer(
            examples['text'],
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=max_len,
        )
        labels = examples['label']
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'source': examples['text'],
            'target': labels,
        }

    remove_cols = dataset['train'].column_names
    train_data = dataset['train'].map(
        preprocess,
        batched=True,
        remove_columns=remove_cols,
        keep_in_memory=True,
        load_from_cache_file=False,
    )
    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)

    val_data = dataset['test'].map(
        preprocess,
        batched=True,
        remove_columns=remove_cols,
        keep_in_memory=True,
        load_from_cache_file=False,
    )
    val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)

    train_data = train_data.shuffle(seed=args.seed)
    val_data = val_data.shuffle(seed=args.seed)
    return train_data, val_data, tokenizer


def get_e2e(args):
    data_dir = os.path.join(BASE_DATA_DIR, 'e2e')
    train_file = os.path.join(data_dir, 'trainset.csv')
    val_file = os.path.join(data_dir, 'devset.csv')
    dataset = load_dataset(
        'csv',
        data_files={'train': train_file, 'validation': val_file},
        cache_dir=os.path.join(data_dir, 'csv'),
    )

    tokenizer = load_tokenizer(args)
    tokenizer.padding_side = 'left'
    max_len = args.max_len

    def preprocess_train(examples):
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        batch_sources = []
        batch_targets = []

        for mr, ref in zip(examples['mr'], examples['ref']):
            input_text = f"MR: {mr} \n Ref: "
            target_text = f"{ref} {tokenizer.eos_token}"
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)

            full_input_ids = input_ids + target_ids
            labels = [-100] * len(input_ids) + target_ids
            pad_len = max_len - len(full_input_ids)
            if pad_len < 0:
                continue

            full_input_ids = full_input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * (max_len - pad_len) + [0] * pad_len

            batch_input_ids.append(full_input_ids)
            batch_attention_masks.append(attention_mask)
            batch_labels.append(labels)
            batch_sources.append(mr)
            batch_targets.append(ref)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'labels': batch_labels,
            'source': batch_sources,
            'target': batch_targets,
        }

    def preprocess_val(examples):
        batch_input_ids = []
        batch_attention_masks = []
        batch_sources = []
        batch_targets = []

        prefix_ids = tokenizer.encode("MR: ", add_special_tokens=False)
        suffix_ids = tokenizer.encode(" \n Ref: ", add_special_tokens=False)

        for mr, refs in zip(examples['mr'], examples['ref']):
            mr_ids = tokenizer.encode(mr, add_special_tokens=False)
            input_ids = prefix_ids + mr_ids + suffix_ids
            pad_len = max_len - len(input_ids)
            if pad_len < 64:
                continue

            input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
            attention_mask = [0] * pad_len + [1] * (max_len - pad_len)

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_sources.append(mr)
            batch_targets.append(refs)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'source': batch_sources,
            'target': batch_targets,
        }

    train_cols = dataset['train'].column_names
    train_data = dataset['train'].map(preprocess_train, batched=True, remove_columns=train_cols)
    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)

    val_df = dataset['validation'].to_pandas()
    val_df_grouped = val_df.groupby('mr')['ref'].apply(list).reset_index()
    val_data_raw = datasets.Dataset.from_pandas(val_df_grouped)
    val_data = val_data_raw.map(preprocess_val, batched=True, remove_columns=val_data_raw.column_names)
    val_data.set_format('torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    train_data = train_data.shuffle(seed=args.seed)
    val_data = val_data.shuffle(seed=args.seed)
    return train_data, val_data, tokenizer


def get_dataset(args):
    if args.dataset == '20news':
        return get_20news(args)
    if args.dataset == 'e2e':
        return get_e2e(args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='20news', choices=['20news', 'e2e'])
    parser.add_argument('--model', type=str, default='distilbert')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    train_ds, val_ds, tokenizer = get_dataset(args)
    print(f"train={len(train_ds)} validation={len(val_ds)} tokenizer={tokenizer.__class__.__name__}")
