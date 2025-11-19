import json
import os
import re
from datetime import timedelta

import datasets
import torch
import torch.distributed as dist
from datasets import Dataset, get_dataset_config_info, load_dataset, load_from_disk


def torch_dist_init():
    if not dist.is_initialized():
        local_device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=300),
            device_id=local_device,
        )
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        print(
            f"Initialized process group with rank {dist.get_rank()} on device {torch.cuda.current_device()}"
        )
    else:
        print("Process group already initialized.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    return local_rank, world_size


def rank0_print(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def remove_padding(list_of_tensor: list[torch.tensor], list_of_actual_size: list[int]):
    actual_tensors = []
    for tensor, actual_size in zip(list_of_tensor, list_of_actual_size):
        actual_tensors.append(tensor[:actual_size])
    return actual_tensors


def gather_tensor_to_rank_0(vectors, actual_size: list[int]):
    if not dist.is_initialized():
        return vectors

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    vector_list = [
        torch.zeros_like(vectors, device=f"cuda:{local_rank}").clone()
        for _ in range(world_size)
    ]
    # Gather vectors to rank 0
    dist.gather(vectors, gather_list=vector_list if local_rank == 0 else None, dst=0)
    if local_rank == 0:
        vector_list = remove_padding(vector_list, actual_size)
        vectors = torch.cat(vector_list, dim=0).cpu().numpy()
        return vectors


def prepare_concat_dataset(
    data_source: str, dataset_name: str, add_id: bool = True
) -> Dataset:
    dataset_info = get_dataset_config_info(data_source, dataset_name)
    split_info = dataset_info.splits
    dataset_list = []
    for split_name, split_info in split_info.items():
        rank0_print(f"Loading {dataset_name} - {split_name} split...")
        dataset = load_dataset(data_source, dataset_name, split=split_name)
        dataset_list.append(dataset)

    dataset: Dataset = datasets.concatenate_datasets(dataset_list)
    if add_id:
        data_id = Dataset.from_dict({"ids": list(range(len(dataset)))})
        dataset = datasets.concatenate_datasets([data_id, dataset], axis=1)
    return dataset


def extract_json_code(text: str) -> str:
    """Extract json code from text."""
    # Find the first occurrence of ```json
    json_start = text.find("```json")
    if json_start != -1:
        text = text.replace("```json", "")
        json_end = text.find("```")
        if json_end != -1:
            text = text.replace("```", "")
        return text

    # If no ```json found, try to find just ```
    json_start = text.find("```")
    if json_start != -1:
        text = text.replace("```", "")
        return text

    return text


class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r"([^\\])\\([^\\])"), r"\1\\\\\2"),
            (re.compile(r",(\s*])"), r"\1"),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)


def convert_open_to_hf(messages):
    hf_messages = []
    for message in messages:
        new_message = {"role": message["role"], "content": []}
        for content in message["content"]:
            if content["type"] == "image_url":
                new_message["content"].append(
                    {"type": "image", "image_url": content["image_url"]["url"]}
                )
            elif content["type"] == "audio_url":
                new_message["content"].append(
                    {"type": "audio", "audio_url": content["audio_url"]["url"]}
                )
            elif content["type"] == "video_url":
                new_message["content"].append(
                    {"type": "video", "video_url": content["video_url"]["url"]}
                )
            else:
                new_content = {"type": "text", "text": content["text"]}
                if "audio_text" in content:
                    new_content["audio_text"] = content["audio_text"]
                new_message["content"].append(new_content)
        hf_messages.append(new_message)

    return hf_messages


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def maybe_load_dataset_type(dataset_path: str) -> Dataset:
    if dataset_path.endswith(".parquet"):
        return Dataset.from_parquet(dataset_path)
    else:
        return load_from_disk(dataset_path)
