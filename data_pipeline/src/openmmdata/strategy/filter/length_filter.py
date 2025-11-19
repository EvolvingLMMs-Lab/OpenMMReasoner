from typing import Callable

from datasets import Dataset
from transformers import AutoProcessor

from openmmdata.utils import convert_open_to_hf

from .base_filter import BaseFilterStrategy


class LengthFilter(BaseFilterStrategy):
    def __init__(
        self,
        filter_length: int,
        processor_name: str,
        filter_key: str = "messages",
        **kwargs,
    ) -> None:
        super().__init__()
        self.filter_key = filter_key
        self.filter_length = filter_length
        self.kwargs = kwargs
        self.processor_name = processor_name
        self.processor = AutoProcessor.from_pretrained(self.processor_name)

    def filter_one_sample(self, sample: dict) -> bool:
        messages = convert_open_to_hf(sample["messages"])
        content = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )[0]
        return len(content) > self.filter_length

    def make_default_filter_fn(self) -> Callable:
        return self.filter_one_sample

    def filter(self, dataset: Dataset, filter_fn: Callable = None) -> Dataset:
        if filter_fn is None:
            filter_fn = self.make_default_filter_fn()
        return dataset.filter(filter_fn, num_proc=32)
