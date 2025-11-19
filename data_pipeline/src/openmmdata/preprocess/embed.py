import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from datasets import Dataset, load_dataset
from PIL import Image
from safetensors.torch import load_file, save_file
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from openmmdata.encoder import CNNEncoder, WordVec
from openmmdata.utils import (
    gather_tensor_to_rank_0,
    rank0_print,
    remove_padding,
    torch_dist_init,
)

Image.MAX_IMAGE_PIXELS = 933120000


def parse_openai_messages(messages, data_folder):
    """
    Parse OpenAI messages format to extract text and image content.

    Args:
        messages: List of OpenAI message dictionaries
        data_folder: Path to folder containing image files

    Returns:
        tuple: (text_content, image_paths)
    """
    text_content = ""
    image_paths = []

    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", [])
            if isinstance(content, str):
                text_content += content + " "
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        text_content += item.get("text", "") + " "
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        else:
                            url = image_url
                        # Convert URL to local file path
                        if url.startswith("file://"):
                            image_path = url[7:]  # Remove file:// prefix
                        else:
                            # Assume it's a relative path in data_folder
                            image_path = os.path.join(data_folder, url)
                        image_paths.append(image_path)

    return text_content.strip(), image_paths


def parse_rl_messages(data):
    text_content = ""
    messages = data["prompt"]
    images = data["images"]
    for message in messages:
        role = message["role"]
        # Skip system because they are all the same
        if role == "system":
            continue
        content = message["content"]
        content_list = []
        segments = re.split("(<image>|<video>)", content)
        segments = [item for item in segments if item != ""]
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image"})
            elif segment == "<video>":
                content_list.append({"type": "video"})
            else:
                text_content += segment + " "
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list
    return text_content.strip(), images


def collate_fn(batch):
    """Custom collate function for batching data."""
    images = [item["images"] for item in batch]
    texts = [item["text"] for item in batch]
    ids = [item["id"] for item in batch]
    return {
        "images": images,
        "text": texts,
        "ids": ids,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed parquet files with OpenAI message format."
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
        "--use-cache", action="store_true", help="Use the cache file if it exists."
    )
    parser.add_argument(
        "--source-type",
        type=str,
        help="The source type to use.",
        default="sft",
        choices=["sft", "rl"],
    )
    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_dataset(
    dataset, data_folder, image_encoder, wordvec, local_rank, source_type
):
    """Process a single dataset to extract features."""
    print(f"Processing dataset with {len(dataset)} samples...")

    # Extract text and images from messages
    texts = []
    image_lists = []
    valid_ids = []

    for item in tqdm(dataset, desc="Extracting content", disable=local_rank != 0):
        try:
            if source_type == "sft":
                text_content, image_paths = parse_openai_messages(
                    item["messages"], data_folder
                )

                # Filter out items with no text content
                if not text_content.strip():
                    continue

                texts.append(text_content)
                image_lists.append(image_paths)
                valid_ids.append(item["id"])
            elif source_type == "rl":
                text_content, images = parse_rl_messages(item)
                texts.append(text_content)
                image_lists.append(images)
                valid_ids.append(item["extra_info"]["index"])
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            continue

    if len(texts) == 0:
        print("No valid items found in dataset")
        return None, None, None

    print(f"Extracted {len(texts)} valid items")

    # Encode text
    text_emb = wordvec.encode(texts, batch_size=4096, show_progress_bar=local_rank == 0)
    text_emb = np.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)
    text_emb = torch.tensor(text_emb, device=f"cuda:{local_rank}")

    # Process images if any
    if any(len(img_list) > 0 for img_list in image_lists):
        image_vectors_list = []

        for i, img_list in enumerate(
            tqdm(image_lists, desc="Processing images", disable=local_rank != 0)
        ):
            if len(img_list) == 0:
                # No images, create zero vector
                zero_vector = torch.zeros(
                    image_encoder.embedding_dim, device=f"cuda:{local_rank}"
                )
                image_vectors_list.append(zero_vector)
                continue

            # Process images for this item
            img_vectors = []
            for img_path in img_list:
                try:
                    if isinstance(img_path, Image.Image):
                        img = img_path.convert("RGB")
                    elif os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                    else:
                        print(f"Image not found: {img_path}")
                        continue

                    img_tensor = image_encoder.transform(img)
                    img_tensor = img_tensor.to(image_encoder.device)

                    with torch.no_grad():
                        img_vector = image_encoder.encode(img_tensor.unsqueeze(0))
                        img_vectors.append(img_vector.squeeze(0))
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    continue

            if img_vectors:
                # Average the image vectors
                avg_vector = torch.stack(img_vectors).mean(dim=0)
                image_vectors_list.append(avg_vector)
            else:
                # No valid images, create zero vector
                zero_vector = torch.zeros(
                    image_encoder.embedding_dim, device=f"cuda:{local_rank}"
                )
                image_vectors_list.append(zero_vector)

        image_vectors = torch.stack(image_vectors_list, dim=0)
        image_vectors = np.nan_to_num(
            image_vectors.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0
        )
        image_vectors = normalize(image_vectors, axis=1, norm="l2")
        image_vectors = torch.tensor(image_vectors, device=f"cuda:{local_rank}")
    else:
        # No images, create zero tensors
        image_vectors = torch.zeros(
            len(texts), image_encoder.embedding_dim, device=f"cuda:{local_rank}"
        )

    # Concatenate text and image features
    features = torch.cat([text_emb, image_vectors], dim=1)

    return features, valid_ids, texts


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank, world_size = torch_dist_init()
    else:
        local_rank, world_size = 0, 1

    args = parse_args()
    config = load_config(args.config)
    cache_folder = args.cache_folder

    if args.use_cache:
        try:
            cache_dict = load_file(
                f"{cache_folder}/vectors_rank_{local_rank}.safetensors",
                device=f"cuda:{local_rank}",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cache file: {e}. Please check the cache folder or remove it."
            )
    else:
        os.makedirs(cache_folder, exist_ok=True)
        cache_dict = {}

    # Initialize encoders
    image_encoder = CNNEncoder()
    wordvec = WordVec()

    # Process each dataset
    for dataset_config in config["datasets"]:
        parquet_path = dataset_config["path"]
        data_folder = dataset_config["data_folder"]

        # Generate dataset name from parquet path
        dataset_name = Path(parquet_path).stem

        if dataset_name in cache_dict:
            rank0_print(f"Skipping {dataset_name} as it is already processed.")
            continue

        print(f"Processing dataset: {dataset_name}")
        print(f"Parquet path: {parquet_path}")
        print(f"Data folder: {data_folder}")

        # Load dataset from parquet
        try:
            dataset = load_dataset("parquet", data_files=parquet_path, split="train")
        except Exception as e:
            print(f"Error loading dataset {parquet_path}: {e}")
            continue

        # Shard dataset for distributed processing
        if world_size > 1:
            local_dataset = dataset.shard(
                num_shards=world_size, index=local_rank, contiguous=True
            )
        else:
            local_dataset = dataset

        # Process dataset
        features, valid_ids, texts = process_dataset(
            local_dataset,
            data_folder,
            image_encoder,
            wordvec,
            local_rank,
            args.source_type,
        )

        if features is not None:
            # Save features to cache
            cache_dict[dataset_name] = features

            # Save cache
            save_file(
                cache_dict,
                f"{cache_folder}/vectors_rank_{local_rank}.safetensors",
                metadata={"dtype": "float16"},
            )

            print(f"Saved features for {dataset_name}: {features.shape}")

        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    print("Embedding completed!")
