"""Workload generators."""

from llmbench.datasets.base import LengthDist, TokenSpace, Workload, ideal_prefix_hit_rate
from llmbench.datasets.synthetic import prefix_workload, random_workload
from llmbench.datasets.traces import azure_workload, mooncake_workload, sharegpt_workload

__all__ = [
    "LengthDist",
    "TokenSpace",
    "Workload",
    "azure_workload",
    "ideal_prefix_hit_rate",
    "mooncake_workload",
    "prefix_workload",
    "random_workload",
    "sharegpt_workload",
]
