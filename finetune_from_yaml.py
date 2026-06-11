#!/usr/bin/env python3
"""Fine-tune Qwen2.5-VL with a LLaMA-Factory YAML config.

This launcher does two things:
1. registers the local ChangeVG caption datasets in LLaMA-Factory's dataset_info.json
2. runs: llamafactory-cli train <yaml>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "qwen2vl_lora_sft.yaml"
DEFAULT_LLAMA_FACTORY_DIR = Path("/data/coding/LLaMA-Factory")

DATASET_ENTRIES = {
    "caption": "data/coding/muti_task_data/train_task_data/caption.json",
    "test_data_extra": "data/coding/muti_task_data/test_task_data/caption.json",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_sharegpt_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset should be a non-empty JSON list: {path}")

    sample = rows[0]
    if not isinstance(sample, dict):
        raise ValueError(f"Dataset sample should be an object: {path}")
    if "conversations" not in sample or "images" not in sample:
        raise ValueError(f"Expected keys 'conversations' and 'images' in: {path}")


def make_dataset_info_entry(path: Path) -> dict:
    return {
        "file_name": str(path),
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }


def register_datasets(dataset_info_path: Path, repo_root: Path, dry_run: bool = False) -> dict:
    entries = {}
    for dataset_name, rel_path in DATASET_ENTRIES.items():
        dataset_path = (repo_root / rel_path).resolve()
        validate_sharegpt_dataset(dataset_path)
        entries[dataset_name] = make_dataset_info_entry(dataset_path)

    if dry_run:
        return entries

    dataset_info = load_json(dataset_info_path)
    dataset_info.update(entries)
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return entries


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to LLaMA-Factory training YAML.")
    parser.add_argument(
        "--llama-factory-dir",
        default=str(DEFAULT_LLAMA_FACTORY_DIR),
        help="Path to the LLaMA-Factory repository.",
    )
    parser.add_argument(
        "--dataset-info",
        default=None,
        help="Path to dataset_info.json. Default: <llama-factory-dir>/data/dataset_info.json",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Path to this ChangeVG repository root.",
    )
    parser.add_argument("--cli", default="llamafactory-cli", help="LLaMA-Factory CLI executable.")
    parser.add_argument("--skip-dataset-info", action="store_true", help="Do not update dataset_info.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without launching training.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    llama_factory_dir = Path(args.llama_factory_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    dataset_info_path = (
        Path(args.dataset_info).resolve()
        if args.dataset_info is not None
        else llama_factory_dir / "data" / "dataset_info.json"
    )

    require_file(config_path, "YAML config")
    if not args.dry_run and not llama_factory_dir.exists():
        raise FileNotFoundError(f"LLaMA-Factory directory not found: {llama_factory_dir}")

    if not args.skip_dataset_info:
        entries = register_datasets(dataset_info_path, repo_root, dry_run=args.dry_run)
        print("Dataset entries:")
        for name, entry in entries.items():
            print(f"  {name}: {entry['file_name']}")
        print(f"dataset_info.json: {dataset_info_path}")

    command = [args.cli, "train", str(config_path)]
    print("Train command:")
    print("  " + " ".join(command))

    if args.dry_run:
        return

    if shutil.which(args.cli) is None:
        raise FileNotFoundError(f"{args.cli} not found in PATH. Activate the LLaMA-Factory environment first.")

    subprocess.run(command, cwd=str(llama_factory_dir), check=True)


if __name__ == "__main__":
    main()
