from __future__ import annotations

from typing import Iterable, NamedTuple

import torch


class _ModuleInfo(NamedTuple):
    attr: str
    path: str
    module: torch.nn.Module
    parent: torch.nn.Module | None


def named_modules_with_parent(model: torch.nn.Module) -> Iterable[_ModuleInfo]:
    for path, module in model.named_modules():
        if path == "":
            assert module is model
            yield _ModuleInfo("", "", model, parent=None)
            continue

        *parent_path, attr = path.split(".")
        parent = model
        for child in parent_path:
            parent = getattr(parent, child)
        assert getattr(parent, attr) is module, (
            "named_modules_with_parents failed to find module in parent by name. "
            "If this occurs, please report a bug."
        )
        yield _ModuleInfo(attr, path, module, parent)
