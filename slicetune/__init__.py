from . import nn, utils
from .utils import describe, fuse, get_slicetune_layers, get_tuners, pretty_describe, set_trainable

patch_linears = nn.Linear.patch

__all__ = [
    "nn",
    "utils",
    "patch_linears",
    "describe",
    "fuse",
    "get_slicetune_layers",
    "get_tuners",
    "pretty_describe",
    "set_trainable",
]
