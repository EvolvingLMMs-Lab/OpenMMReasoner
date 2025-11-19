import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import yaml
from datasets import Dataset, Value, load_dataset, load_from_disk
from safetensors.torch import load_file
from tqdm import tqdm
from vicinity import Backend
from vicinity.backends import get_backend_class

from openmmdata.utils import prepare_concat_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate parquet files with OpenAI message format."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default="cache/v1",
        help="Path to the cache safetensors file.",
    )
    parser.add_argument(
        "--dataset-cache-folder",
        type=str,
        default="cache/v1/dataset_cache",
        help="Path to the dataset cache folder where the deduplicated dataset will be saved.",
    )
    parser.add_argument("--nprocess", type=int, default=8, help="Number of GPU to use.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether to push the deduplicated dataset to the Hugging Face Hub.",
    )
    parser.add_argument("--repo-id", type=str, help="The repo to push")
    parser.add_argument(
        "--force-push", action="store_true", help="Force push to the hub."
    )
    parser.add_argument(
        "--source-type",
        type=str,
        help="The source type to use.",
        default="sft",
        choices=["sft", "rl"],
    )
    parser.add_argument(
        "--threshold", type=float, help="The threshold to use.", default=0.99
    )
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def one_to_all_deduplicate(
    dataset_name: str,
    cache_dict: dict[str, torch.Tensor],
    task_dict: dict[str, Dataset],
    exclude_from_db_list: List[str],
    threshold: float = 0.99,
) -> Dataset:
    """
    Deduplicate the dataset using one-to-all deduplication.
    """
    vectors = cache_dict[dataset_name].cpu().numpy()
    db = [
        cache_dict[subset]
        for subset in task_dict.keys()
        if subset != dataset_name and subset not in exclude_from_db_list
    ]
    if len(db) == 0:
        print(f"No other subsets to deduplicate against for {dataset_name}.")
        return task_dict[dataset_name]
    db = torch.cat(db, dim=0).cpu().numpy()
    dataset = task_dict[dataset_name]

    backend = Backend.USEARCH
    backend = get_backend_class(backend)
    print(f"Building vector store with backend ...")
    backend = backend.from_vectors(db)
    results = backend.threshold(vectors, threshold=1 - threshold, max_k=10)

    select_ids = []
    for idx, (indices, distances) in enumerate(results):
        if len(indices) == 0:
            select_ids.append(idx)
    original_size = len(dataset)
    dataset = dataset.select(select_ids)
    print(f"Deduplicated {original_size} samples to {len(dataset)} samples.")

    return dataset


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    cache_folder = args.cache_folder
    force_push = args.force_push
    dataset_cache_folder = args.dataset_cache_folder
    os.makedirs(dataset_cache_folder, exist_ok=True)
    nprocess = args.nprocess
    push_to_hub = args.push_to_hub
    repo_id = args.repo_id

    # Get dataset names from config
    dataset_configs = config["datasets"]
    dataset_names = [Path(config["path"]).stem for config in dataset_configs]

    cache_dict_by_rank = {}
    for rank in range(nprocess):
        cache_file = f"{cache_folder}/vectors_rank_{rank}.safetensors"
        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} does not exist.")
            continue
        cache_dict_by_rank[rank] = load_file(cache_file, device="cuda")

    print(f"Loaded cache files from {cache_folder} for {nprocess} ranks.")

    # Merge the cache tensors across ranks
    cache_dict = defaultdict(list)
    for rank in range(nprocess):
        for dataset_name in dataset_names:
            if dataset_name in cache_dict_by_rank[rank]:
                cache_dict[dataset_name].append(cache_dict_by_rank[rank][dataset_name])

    if push_to_hub:
        from datasets import get_dataset_config_names

        subset_uploaded_list = get_dataset_config_names(repo_id)

    # Concatenate the tensors for each dataset
    task_dict = {}
    processed_list = []
    for dataset_name in dataset_names:
        if dataset_name in cache_dict and len(cache_dict[dataset_name]) > 0:
            cache_dict[dataset_name] = torch.cat(cache_dict[dataset_name], dim=0)

            # Find the corresponding config
            dataset_config = next(
                config
                for config in dataset_configs
                if Path(config["path"]).stem == dataset_name
            )

            # Create dataset from parquet
            dataset = Dataset.from_parquet(dataset_config["path"])
            task_dict[dataset_name] = dataset
        else:
            print(f"No cache data found for {dataset_name}, skipping...")

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        if dataset_name not in task_dict:
            print(f"Skipping {dataset_name} as it has no task data.")
            continue

        print(f"Deduplicating dataset: {dataset_name}")
        try:
            if args.source_type == "sft":
                deduplicated_dataset = load_from_disk(
                    f"{dataset_cache_folder}/{dataset_name}",
                )
            elif args.source_type == "rl":
                deduplicated_dataset = Dataset.from_parquet(
                    f"{dataset_cache_folder}/{dataset_name}.parquet",
                )
            print(
                f"Loaded deduplicated dataset from {dataset_cache_folder}/{dataset_name}."
            )
        except FileNotFoundError:
            dataset = task_dict[dataset_name]
            assert (
                len(dataset) == cache_dict[dataset_name].shape[0]
            ), f"Dataset length {len(dataset)} does not match cache tensor shape {cache_dict[dataset_name].shape[0]} for dataset {dataset_name}."

            deduplicated_dataset = one_to_all_deduplicate(
                dataset_name,
                cache_dict,
                task_dict,
                processed_list,
                threshold=args.threshold,
            )
            # Once the deduplicated is been remove from the current dataset,
            # we consider it is processed and not to be processed again.
            if args.source_type == "sft":
                deduplicated_dataset.save_to_disk(
                    f"{dataset_cache_folder}/{dataset_name}",
                )
            elif args.source_type == "rl":
                # extra_info_features = {'answer': Value('string'), 'index': Value('int32'), 'question': Value('string'), 'split': Value('string')}
                # deduplicated_dataset = deduplicated_dataset.cast_column('extra_info', extra_info_features)
                deduplicated_dataset = deduplicated_dataset.remove_columns(
                    ["extra_info"]
                )
                deduplicated_dataset.to_parquet(
                    f"{dataset_cache_folder}/{dataset_name}.parquet",
                )
        processed_list.append(dataset_name)

        if push_to_hub:
            if force_push or dataset_name not in subset_uploaded_list:
                print(f"Pushing deduplicated dataset {dataset_name} to the hub.")
                deduplicated_dataset.push_to_hub(
                    repo_id=repo_id,
                    config_name=dataset_name,
                    private=True,
                    split="train",
                )
