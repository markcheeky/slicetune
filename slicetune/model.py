from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, NamedTuple

import torch

import slicetune.nn
import slicetune.utils


def mark_for_training(
    model: torch.nn.Module,
    train_tuners: bool = True,
    train_biases: bool = False,
    train_rest: bool = False,
) -> None:

    for param in model.parameters():
        param.requires_grad_(train_rest)

    for param in get_bias_params(model):
        param.requires_grad_(train_biases)

    for param in get_tuners(model):
        param.requires_grad_(train_tuners)


def fuse(model: torch.nn.Module) -> None:
    """
    Replaces all slicetune layers in the model with equivalent standard layers.
    Enables gradient computation for all parameters.
    """
    # prevent invalidating the iterator by setattr
    module_infos = list(slicetune.utils.named_modules_with_parent(model))
    for module_info in module_infos:
        if isinstance(module_info.module, slicetune.nn.Layer):
            replacement = module_info.module.to_standard()
            setattr(module_info.parent, module_info.attr, replacement)
        module_infos = list(slicetune.utils.named_modules_with_parent(model))
    for param in model.parameters():
        param.requires_grad = True


def get_slicetune_layers(
    model: torch.nn.Module,
) -> Iterable[slicetune.nn.Layer]:
    """
    Returns an iterable of all slicetune layers in the model.
    """
    for module in model.modules():
        if isinstance(module, slicetune.nn.Layer):
            yield module


def get_tuners(
    model: torch.nn.Module,
    only_tuning_params: bool = True,
) -> Iterable[torch.nn.Parameter]:
    """
    Returns an iterable of all slicetuning parameters in the model.
    """
    for module in model.modules():
        if isinstance(module, slicetune.nn.Layer):
            if only_tuning_params:
                params = module.get_tuners()
            else:
                params = module.parameters()

            yield from params


def get_bias_params(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    """
    Returns an iterable of all bias parameters in the model.
    """
    for module in model.modules():
        bias = getattr(module, "bias", None)
        if bias is not None and isinstance(bias, torch.nn.Parameter):
            yield bias


def get_trainable_params(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    """
    Returns an iterable of all trainable parameters in the model.
    """
    for param in model.parameters():
        if param.requires_grad:
            yield param


class SlicetuningStats(NamedTuple):
    params_total: int
    params_total_trainable: int
    params_tuners: int
    params_tuners_trainable: int
    params_biases: int
    params_biases_trainable: int
    layers_count: dict[str, int]

    def _pretty(self, value: Any) -> str:
        if isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, float):
            return f"{value:.2%}"
        return str(value)

    def _frac(self, num: int) -> float:
        return num / self.params_total

    @property
    def trainable_frac(self) -> float:
        return self.params_total_trainable / self.params_total


def describe(model: torch.nn.Module) -> SlicetuningStats:
    params_all = list(model.parameters())
    params_total = sum(p.numel() for p in params_all)
    params_total_trainable = sum(p.numel() for p in params_all if p.requires_grad)

    params_tuners_all = list(get_tuners(model))
    params_tuners = sum(p.numel() for p in params_tuners_all)
    params_tuners_trainable = sum(p.numel() for p in params_tuners_all if p.requires_grad)

    biases_all = list(get_bias_params(model))
    params_biases = sum(p.numel() for p in biases_all)
    params_biases_trainable = sum(p.numel() for p in biases_all if p.requires_grad)

    layers_count: Counter[str] = Counter()
    for layer in get_slicetune_layers(model):
        class_name = f"{layer.__class__.__module__}.{layer.__class__.__qualname__}"
        layers_count[class_name] += 1

    return SlicetuningStats(
        params_total,
        params_total_trainable,
        params_tuners,
        params_tuners_trainable,
        params_biases,
        params_biases_trainable,
        layers_count,
    )


def pretty_describe(
    model: torch.nn.Module,
    include_layer_info: bool = True,
    include_param_info: bool = True,
    param_info_brief: bool = False,
    supress_warning: bool = False,
    min_width: int = 50,
) -> str:
    try:
        import prettytable
    except ImportError:
        raise ImportError(
            "prettytable is needed for pretty_describe(). "
            "It comes as an optional dependency slicetune[pretty]"
        )

    stats = describe(model)

    table_params = prettytable.PrettyTable()
    table_params.title = "Slicetuning parameter info".upper().center(min_width)
    table_params.field_names = ["name", "params", "% of total"]
    table_params.align["name"] = "l"
    table_params.align["params"] = "r"
    table_params.align["% of total"] = "r"
    table_params.padding_width = 2

    table_params.add_row(
        [
            "Total".upper(),
            stats._pretty(stats.params_total),
            "",
        ]
    )

    table_params.add_row(
        [
            "|-> trainable",
            stats._pretty(stats.params_total_trainable),
            stats._pretty(stats._frac(stats.params_total_trainable)),
        ]
    )

    table_params.add_row(
        [
            "└-> non-trainable",
            stats._pretty(stats.params_total - stats.params_total_trainable),
            stats._pretty(stats._frac(stats.params_total - stats.params_total_trainable)),
        ]
    )

    if not param_info_brief:

        table_params.add_row(["", "", ""])
        table_params.add_row(["Trainable".upper(), "", ""])

        table_params.add_row(
            [
                "|-> slicetuners",
                stats._pretty(stats.params_tuners_trainable),
                stats._pretty(stats._frac(stats.params_tuners_trainable)),
            ]
        )

        table_params.add_row(
            [
                "|-> biases",
                stats._pretty(stats.params_biases_trainable),
                stats._pretty(stats._frac(stats.params_biases_trainable)),
            ]
        )

        table_params.add_row(
            [
                "└-> others",
                stats._pretty(
                    stats.params_total_trainable
                    - stats.params_tuners_trainable
                    - stats.params_biases_trainable
                ),
                stats._pretty(
                    stats._frac(
                        stats.params_total_trainable
                        - stats.params_tuners_trainable
                        - stats.params_biases_trainable
                    )
                ),
            ]
        )

        table_params.add_row(["", "", ""])
        table_params.add_row(["Non-trainable".upper(), "", ""])

        table_params.add_row(
            [
                "|-> slicetuners",
                stats._pretty(stats.params_tuners - stats.params_tuners_trainable),
                stats._pretty(stats._frac(stats.params_tuners - stats.params_tuners_trainable)),
            ]
        )

        table_params.add_row(
            [
                "|-> biases",
                stats._pretty(stats.params_biases - stats.params_biases_trainable),
                stats._pretty(stats._frac(stats.params_biases - stats.params_biases_trainable)),
            ]
        )

        table_params.add_row(
            [
                "└-> others",
                stats._pretty(stats.params_total - stats.params_tuners - stats.params_biases),
                stats._pretty(
                    stats._frac(stats.params_total - stats.params_tuners - stats.params_biases)
                ),
            ]
        )

    table_params_str = table_params.get_string()

    table_layers = prettytable.PrettyTable()
    table_layers.field_names = ["name", "count"]
    table_layers.header = False
    table_layers.header_style = "upper"
    table_layers.title = "Slicetuning layers".upper().center(min_width)
    table_layers.align["name"] = "l"
    table_layers.align["count"] = "r"
    table_layers.padding_width = 2

    for name, count in stats.layers_count.items():
        table_layers.add_row([name, str(count) + "x"])

    table_layers_str = table_layers.get_string()

    outputs: list[str] = []
    if include_param_info:
        outputs.append(table_params_str)
    if include_layer_info:
        outputs.append(table_layers_str)
    if not supress_warning:
        outputs.append(
            "Note that the number of parameters can be incorrect \n"
            "if you share parameters across modules."
        )

    return "\n\n".join(outputs)
