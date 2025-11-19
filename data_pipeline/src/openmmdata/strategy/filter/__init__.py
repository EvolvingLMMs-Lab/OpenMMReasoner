from .base_filter import BaseFilterStrategy
from .length_filter import LengthFilter
from .ppl_filter import PPLFilter

FILTER_STRATEGY_MAPPING = {
    "length": LengthFilter,
    "ppl": PPLFilter,
}

__all__ = [
    "BaseFilterStrategy",
    "LengthFilter",
    "PPLFilter",
    "FILTER_STRATEGY_MAPPING",
]
