#!/usr/bin/env python3
"""Download datasets for the artifact.

This reproduces the observed layout under the selected data directory:

  data/20news
  data/cifar100
  data/e2e
  data/tiny-imagenet

Usage:
  python artifact/download_datasets.py
  python artifact/download_datasets.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATA_DIR = Path(
    os.environ.get("SCHEDULERS_DATA_DIR", Path(__file__).resolve().parent / "data")
).expanduser()

TWENTY_NEWS_REVISION = "f1b91292074e7cfb69be58b642d583ec262f30ed"

CIFAR100_URL = "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"

E2E_FILES = {
    "trainset.csv": {
        "url": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/trainset.csv",
        "md5": "530eb457155d788912309689cab6f36d",
    },
    "devset.csv": {
        "url": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/devset.csv",
        "md5": "41ab9e06dd4e08ce2a3408f5f652e921",
    },
}

TINY_IMAGENET_URL = "https://huggingface.co/datasets/torch-uncertainty/tiny-imagenet-200/resolve/main/tin.zip"
TINY_IMAGENET_MD5 = "5a785fab37376540e889d1ab3143acf6"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_md5(path: Path, expected: str) -> None:
    actual = file_md5(path)
    if actual != expected:
        raise RuntimeError(f"MD5 mismatch for {path}: expected {expected}, got {actual}")


def download_url(url: str, dst: Path, expected_md5: str | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    ensure_removed(tmp)

    print(f"[GET] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if expected_md5 is not None:
            require_md5(tmp, expected_md5)
        tmp.replace(dst)
    finally:
        ensure_removed(tmp)

    print(f"[OK] downloaded {dst}")


def safe_extract_tar(archive: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if target_dir not in member_path.parents and member_path != target_dir:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(target_dir)


def safe_extract_zip(archive: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_path = (target_dir / member.filename).resolve()
            if target_dir not in member_path.parents and member_path != target_dir:
                raise RuntimeError(f"Unsafe path in zip archive: {member.filename}")
        zf.extractall(target_dir)


def ensure_removed(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def download_20news(data_dir: Path) -> None:
    out_dir = data_dir / "20news"
    ensure_removed(out_dir)

    with tempfile.TemporaryDirectory(prefix="hf-20news-") as cache_dir:
        ds = load_dataset(
            "SetFit/20_newsgroups",
            revision=TWENTY_NEWS_REVISION,
            cache_dir=cache_dir,
        )
        ds.save_to_disk(str(out_dir))
    print(f"[OK] saved 20news to {out_dir}")
    print(f"[INFO] splits: {list(ds.keys())}")


def download_cifar100(data_dir: Path) -> None:
    out_dir = data_dir / "cifar100"
    archive = out_dir / "cifar-100-python.tar.gz"
    extracted = out_dir / "cifar-100-python"

    ensure_removed(out_dir)
    download_url(CIFAR100_URL, archive, CIFAR100_MD5)

    safe_extract_tar(archive, out_dir)
    print(f"[OK] extracted CIFAR-100 to {extracted}")


def download_e2e(data_dir: Path) -> None:
    out_dir = data_dir / "e2e"
    ensure_removed(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, spec in E2E_FILES.items():
        download_url(spec["url"], out_dir / filename, spec["md5"])

    with tempfile.TemporaryDirectory(prefix="hf-e2e-") as cache_dir:
        ds = load_dataset(
            "csv",
            data_files={
                "train": str(out_dir / "trainset.csv"),
                "validation": str(out_dir / "devset.csv"),
            },
            cache_dir=cache_dir,
        )
    print(f"[OK] verified E2E csv files")
    print(f"[INFO] splits: {list(ds.keys())}")


def reorganize_tiny_imagenet_val(root: Path) -> None:
    val_dir = root / "val"
    images_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"

    if not annotations.exists():
        raise RuntimeError(f"Missing Tiny ImageNet annotations: {annotations}")

    if not images_dir.exists():
        return

    with annotations.open("r", encoding="utf-8") as f:
        for line in f:
            image_name, wnid, *_ = line.rstrip("\n").split("\t")
            class_dir = val_dir / wnid
            class_dir.mkdir(exist_ok=True)
            src = images_dir / image_name
            dst = class_dir / image_name
            if src.exists():
                shutil.move(str(src), str(dst))

    try:
        images_dir.rmdir()
    except OSError:
        pass

    print(f"[OK] reorganized Tiny ImageNet validation images under {val_dir}/<wnid>")


def find_tiny_imagenet_root(extract_dir: Path) -> Path:
    candidates = [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]
    for candidate in candidates:
        if (
            (candidate / "train").is_dir()
            and (candidate / "val").is_dir()
            and (candidate / "test").is_dir()
        ):
            return candidate

    raise RuntimeError(f"Could not find Tiny ImageNet root in {extract_dir}")


def download_tiny_imagenet(data_dir: Path) -> None:
    out_dir = data_dir / "tiny-imagenet"
    archive = data_dir / "tiny-imagenet-200.zip"

    ensure_removed(out_dir)
    ensure_removed(archive)
    download_url(TINY_IMAGENET_URL, archive, TINY_IMAGENET_MD5)

    with tempfile.TemporaryDirectory(prefix="tiny-imagenet-", dir=str(data_dir)) as tmp_name:
        tmp_dir = Path(tmp_name)
        safe_extract_zip(archive, tmp_dir)
        extracted = find_tiny_imagenet_root(tmp_dir)

        shutil.move(str(extracted), str(out_dir))

    reorganize_tiny_imagenet_val(out_dir)

    archive.unlink(missing_ok=True)

    print(f"[OK] prepared Tiny ImageNet at {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"dataset root directory (default: {DEFAULT_DATA_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    ensure_removed(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    download_20news(data_dir)
    download_cifar100(data_dir)
    download_e2e(data_dir)
    download_tiny_imagenet(data_dir)


if __name__ == "__main__":
    main()
