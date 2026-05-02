#!/usr/bin/env python3
"""Download transformer models for the artifact."""

from pathlib import Path
import shutil

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parent / "model"
HF_CACHE_DIR = MODEL_DIR / ".hf-cache"


def ensure_removed(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def download_gpt2() -> None:
    out_dir = MODEL_DIR / "gpt2"
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=HF_CACHE_DIR)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"[OK] saved GPT-2 small to {out_dir}")


def download_distilbert() -> None:
    out_dir = MODEL_DIR / "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=HF_CACHE_DIR)
    model = AutoModel.from_pretrained("distilbert-base-uncased", cache_dir=HF_CACHE_DIR)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"[OK] saved DistilBERT base uncased to {out_dir}")


def main() -> None:
    ensure_removed(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    download_gpt2()
    download_distilbert()
    ensure_removed(HF_CACHE_DIR)


if __name__ == "__main__":
    main()
