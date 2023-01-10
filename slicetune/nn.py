from __future__ import annotations

import abc
import random
from typing import Any, Callable, Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Self


class Layer(abc.ABC, torch.nn.Module):
    @classmethod
    @abc.abstractmethod
    def from_standard(cls, layer: torch.nn.Module, **kwargs: Any) -> Self:
        ...

    @classmethod
    @abc.abstractmethod
    def patch(cls, model: torch.nn.Module, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def to_standard(self) -> torch.nn.Module:
        ...

    @abc.abstractmethod
    def get_tuners(self) -> Iterable[torch.nn.Parameter]:
        ...

    @staticmethod
    def get_operation_fn(
        operation: Literal["add", "mul", "pow"]
    ) -> Callable[[Tensor, Tensor], Any]:
        if operation == "add":
            return Tensor.add_
        elif operation == "mul":
            return Tensor.mul_
        elif operation == "pow":
            return Tensor.pow_
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def get_init_value(operation: Literal["add", "mul", "pow"]) -> float:
        if operation == "add":
            return 0.0
        elif operation == "mul":
            return 1.0
        elif operation == "pow":
            return 1.0
        else:
            raise ValueError(f"Unknown operation: {operation}")


class Linear(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tuner_size: tuple[int, int] | float | None,
        bias: bool = True,
        operation: Literal["add", "mul", "pow"] | None = "add",
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        copy_params_from: torch.nn.Linear | None = None,
        tuner_slice: tuple[slice, slice] | None = None,
        random_generator: random.Random | None = None,
    ) -> None:
        super().__init__()

        self.weight: torch.nn.Parameter
        self.tuner: torch.nn.Parameter
        self.bias: torch.nn.Parameter | None
        self.tuner_dropout: torch.nn.Dropout
        self.in_features: int
        self.out_features: int
        self.operation: Literal["add", "mul", "pow"]
        self.operation_fn: Callable[[Tensor, Tensor], Any]
        self.tuner_slice: tuple[slice, slice]

        if operation is None:
            operation = "add"

        if random_generator is None:
            random_generator = random.Random()

        if in_features < 1:
            raise ValueError(f"invalid in_features: {in_features}")

        if out_features < 1:
            raise ValueError(f"invalid out_features: {out_features}")

        if tuner_size is None and tuner_slice is None:
            raise ValueError("Either tuner_size or tuner_slice must be specified")

        if tuner_size is not None and tuner_slice is not None:
            if isinstance(tuner_size, float):
                raise ValueError(
                    "tuner_size as a fraction and tuner_slice cannot be specified together"
                )
            else:
                w, h = tuner_slice
                if tuner_size != (w.stop - w.start, h.stop - h.start):
                    raise ValueError(
                        f"tuner_size and tuner_slice do not match {tuner_size}, {tuner_slice}"
                    )

        if isinstance(tuner_size, float) and not 0 <= tuner_size <= 1:
            raise ValueError(f"invalid tuner_size: {tuner_size}")

        tensor_kwargs: dict[str, Any] = dict(device=device, dtype=dtype)
        weight_shape = out_features, in_features

        if isinstance(tuner_size, float):
            tuner_shape = self._get_random_crop(tuner_size, weight_shape, random_generator)
        elif tuner_size is None:
            assert tuner_slice is not None
            w, h = tuner_slice
            tuner_shape = w.stop - w.start, h.stop - h.start
        else:
            tuner_shape = tuner_size

        if tuner_slice is None:
            tuner_slice = self._get_random_coords(weight_shape, tuner_shape, random_generator)

        if not all(
            [
                0 <= tuner_slice[0].start < tuner_slice[0].stop <= weight_shape[0],
                0 <= tuner_slice[1].start < tuner_slice[1].stop <= weight_shape[1],
                tuner_shape[0] == tuner_slice[0].stop - tuner_slice[0].start,
                tuner_shape[1] == tuner_slice[1].stop - tuner_slice[1].start,
            ]
        ):
            raise ValueError(f"invalid tuner_slice: {tuner_slice}")

        self.in_features = in_features
        self.out_features = out_features
        self.tuner_slice = tuner_slice
        self.operation = operation
        self.operation_fn = self.get_operation_fn(operation)
        self.weight = torch.nn.Parameter(torch.empty(weight_shape, **tensor_kwargs))
        self.tuner = torch.nn.Parameter(torch.empty(tuner_shape, **tensor_kwargs))
        self.tuner_dropout = torch.nn.Dropout(dropout)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **tensor_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(copy_params_from)

    def reset_parameters(self, linear: torch.nn.Linear | None = None) -> None:
        if linear is None:
            # this gives us the same initialization as torch.nn.Linear
            linear = torch.nn.Linear(self.in_features, self.out_features, self.bias is not None)

        with torch.no_grad():
            self.weight[:] = linear.weight
            if self.bias is not None:
                self.bias[:] = linear.bias
            self.tuner.fill_(self.get_init_value(self.operation))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self._get_w(), self.bias)

    def _get_w(self) -> Tensor:
        w = self.weight.clone()
        tuner = self.tuner_dropout(self.tuner)
        self.operation_fn(w[self.tuner_slice], tuner)
        return w

    @classmethod  # type: ignore
    def from_standard(
        cls,
        layer: torch.nn.Module,
        tuner_size: float | tuple[int, int],
        operation: Literal["add", "mul", "pow"] = "add",
        dropout: float = 0.0,
        random_generator: random.Random | None = None,
    ) -> Self:
        """
        Create a slicetune Linear layer from a standard Linear layer.
        By default, the layer will be initialized so that without any training,
        forward() will return the same result as the original Linear layer.
        """
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError(f"Expected linear to be of type torch.nn.Linear, got {type(layer)}")

        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            copy_params_from=layer,
            tuner_size=tuner_size,
            operation=operation,
            dropout=dropout,
            random_generator=random_generator,
        )

    def to_standard(self) -> torch.nn.Linear:
        """
        Converts the slicetune Linear layer to a standard Linear layer, such that
        forward() of the two layers will return the same result (in eval mode)
        """
        was_training = self.training
        self.eval()
        layer = torch.nn.Linear(
            self.in_features,
            self.out_features,
            self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            layer.weight[:] = self._get_w()
            if layer.bias is not None:
                assert self.bias is not None
                layer.bias[:] = self.bias
        if was_training:
            self.train()
        return layer

    def get_tuners(self) -> Iterable[torch.nn.Parameter]:
        return [self.tuner]

    @classmethod  # type: ignore
    def patch(
        cls,
        model: torch.nn.Module,
        tuner_size: float | tuple[int, int],
        operation: Literal["add", "mul", "pow"] = "add",
        dropout: float = 0.0,
        random_generator: random.Random | None = None,
    ) -> None:
        """
        Replaces all compatible layers in the model with equivalent slicetune layers.
        Disables gradient computation for all parameters except for the slicetuners.
        """
        if isinstance(model, torch.nn.Linear):
            raise ValueError(
                "patch() cannot be called on a torch.nn.Linear. patch() changes model's "
                "layers inplace. For converting a layer, use slicetune.nn.Linear.from_standard() "
            )

        if random_generator is None:
            random_generator = random.Random()

        # prevent invalidating the iterator by setattr
        modules = list(model.named_modules())
        for name, module in modules:
            if isinstance(module, torch.nn.Linear):
                replaced = Linear.from_standard(
                    module,
                    tuner_size=tuner_size,
                    operation=operation,
                    dropout=dropout,
                    random_generator=random_generator,
                )
                setattr(model, name, replaced)

    def __repr__(self) -> str:
        return (
            f"{self.__module__}.{self.__class__.__qualname__}"
            "("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"tuner_size={self.tuner.shape}, "
            f"tuner_slice={self.tuner_slice}, "
            f"operation={self.operation}, "
            f"dropout={self.tuner_dropout.p}"
            ")"
        )

    @staticmethod
    def _get_random_crop(
        fraction: float,
        rect: tuple[int, int],
        random_generator: random.Random,
    ) -> tuple[int, int]:
        a, b = rect
        # We want to have:
        #   c * d ~= fraction * a * b
        # then it holds that:
        #   c ~= fraction * a * b / d
        # if c grows, d must shrink to make up for that area
        #   => to get maximum value for c, plug in d = 1 to the formula above
        #   => to get minimum value for c, plug in d = b
        # this will give us a range of valid values for c
        # now we can sample c from that range and compute d accordingly
        c_max = max(1, min(a, round(fraction * a * b)))
        c_min = max(1, min(a, round(fraction * a)))
        c = random_generator.randint(c_min, c_max)
        d = min(b, max(1, round(fraction * a * b / c)))
        return c, d

    @staticmethod
    def _get_random_coords(
        shape: tuple[int, int],
        crop: tuple[int, int],
        random_generator: random.Random,
    ) -> tuple[slice, slice]:
        start_out = random_generator.randint(0, shape[0] - crop[0])
        start_in = random_generator.randint(0, shape[1] - crop[1])
        return slice(start_out, start_out + crop[0]), slice(start_in, start_in + crop[1])
