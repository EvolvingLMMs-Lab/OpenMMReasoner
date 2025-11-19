import os
from typing import Callable

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from openmmdata.utils import convert_open_to_hf

from .base_filter import BaseFilterStrategy


class PPLFilter(BaseFilterStrategy):
    def __init__(
        self,
        filter_ppl: float,
        model_name: str,
        processor_name: str,
        filter_key: str = "messages",
        **kwargs,
    ) -> None:
        super().__init__()
        self.filter_key = filter_key
        self.filter_ppl = filter_ppl
        self.kwargs = kwargs
        self.processor_name = processor_name
        self.model_name = model_name
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.model_type = self.model_config.model_type
        if type(self.model_config) in AutoModelForCausalLM._model_mapping.keys():
            model_cls = AutoModelForCausalLM
        elif (
            type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys()
        ):
            model_cls = AutoModelForImageTextToText
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = f"cuda:{rank}"
        self.model = model_cls.from_pretrained(
            self.model_name, device_map={"": device_map}, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(self.processor_name)

    def filter_one_sample(self, sample: dict) -> bool:
        messages = convert_open_to_hf(sample[self.filter_key])
        content = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
            return_tensors=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=content, images=image_inputs, videos=video_inputs, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            outputs = self.model(labels=labels, **inputs)
        loss = outputs.loss

        return loss < self.filter_ppl

    def make_default_filter_fn(self) -> Callable:
        return self.filter_one_sample

    def filter(self, dataset: Dataset, filter_fn: Callable = None) -> Dataset:
        if filter_fn is None:
            filter_fn = self.make_default_filter_fn()

        rank = os.environ.get("LOCAL_RANK", None)
        world_size = os.environ.get("WORLD_SIZE", None)
        # If use torch.distributed, we need to initialize the process group
        if rank is not None and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device(f"cuda:{rank}"),
            )
            torch.cuda.set_device(torch.device(f"cuda:{rank}"))

        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        dataset = dataset.shard(
            num_shards=dist.get_world_size(), index=self.rank, contiguous=True
        )

        dataset = dataset.filter(filter_fn, num_proc=1)

        if dist.is_initialized():
            dataset_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dataset_list, dataset)
            dataset = concatenate_datasets(dataset_list)
        else:
            dataset = dataset

        return dataset
