import argparse
import os
from pathlib import Path

import torch.distributed as dist
import yaml
from datasets import Dataset
from tqdm import tqdm

from openmmdata.strategy import FILTER_STRATEGY_MAPPING
from openmmdata.utils import maybe_load_dataset_type, simple_parse_args_string


def parse_args():
    parser = argparse.ArgumentParser(description="Filter parquet files.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="cache/v1/filtered",
        help="Path to the output folder where filtered datasets will be saved.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether to push the filtered dataset to the Hugging Face Hub.",
    )
    parser.add_argument("--repo-id", type=str, help="The repo to push")
    parser.add_argument(
        "--force-push", action="store_true", help="Force push to the hub."
    )
    parser.add_argument(
        "--filter-strategy", type=str, help="The filter strategy to use."
    )
    parser.add_argument("--limit", type=int, help="The number of examples to filter.")
    parser.add_argument("--filter-kwargs", type=str, help="The filter kwargs to use.")
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def filter_dataset(dataset: Dataset, filter_strategy: str, **kwargs) -> Dataset:
    filter_strategy = FILTER_STRATEGY_MAPPING[filter_strategy](**kwargs)
    origin_len = len(dataset)
    filtered_dataset = filter_strategy.filter(dataset)
    print(
        f"Before filtering: {origin_len}, After filtering: {len(filtered_dataset)}, filtered ratio: {(origin_len - len(filtered_dataset)) / origin_len:.2f}"
    )
    return filtered_dataset


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    output_folder = args.output_folder
    force_push = args.force_push
    push_to_hub = args.push_to_hub
    repo_id = args.repo_id
    filter_strategy = args.filter_strategy
    limit = args.limit
    # Filter kwargs would be in "key1=value1,key2=value2" format
    # Make the value from string to dictionary
    filter_kwargs = simple_parse_args_string(args.filter_kwargs)
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Get dataset names from config
    dataset_configs = config["datasets"]
    dataset_names = [Path(config["path"]).stem for config in dataset_configs]
    dataset_paths = [config["path"] for config in dataset_configs]

    # Process each dataset
    pbar = tqdm(total=len(dataset_names), desc="Filtering datasets")
    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        print(f"Processing dataset: {dataset_name}")

        dataset = maybe_load_dataset_type(dataset_path)
        if limit:
            dataset = dataset.select(list(range(limit)))
        filtered_dataset = filter_dataset(dataset, filter_strategy, **filter_kwargs)
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        if rank == 0:
            filtered_dataset.save_to_disk(f"{output_folder}/{dataset_name}")

        print(f"Completed processing {dataset_name}")
        pbar.update(1)
    pbar.close()

    print("Filtering process completed!")
