from abc import ABC, abstractmethod
from typing import Callable

from datasets import Dataset


class BaseFilterStrategy(ABC):
    @abstractmethod
    def filter_one_sample(self, sample: dict) -> bool:
        pass

    @abstractmethod
    def make_default_filter_fn(self, **kwargs) -> Callable:
        pass

    @abstractmethod
    def filter(self, dataset: Dataset, filter_fn: Callable, **kwargs) -> Dataset:
        pass
