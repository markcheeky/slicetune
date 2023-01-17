from . import nn, utils
from .model import (
    describe,
    fuse,
    get_slicetune_layers,
    get_tuners,
    mark_for_training,
    pretty_describe,
)

patch_linears = nn.Linear.patch

__all__ = [
    "nn",
    "model",
    "utils",
    "patch_linears",
    "describe",
    "fuse",
    "get_slicetune_layers",
    "get_tuners",
    "pretty_describe",
    "mark_for_training",
]
