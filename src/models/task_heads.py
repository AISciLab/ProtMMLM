from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence

from src.models._head_utils import LinearLayer, Matrix, apply_activation


class BinaryClassificationHead:
    def __init__(self, input_dim: int, *, seed: int = 200) -> None:
        self.input_dim = input_dim
        self.seed = seed
        self.layer = LinearLayer(input_dim, 1, seed=seed)
        self._torch_layer = None

    def forward(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        if _is_torch_tensor(features):
            return _forward_torch_linear(
                features,
                input_dim=self.input_dim,
                output_dim=1,
                seed=self.seed,
                layer_ref=self,
                attribute_name="_torch_layer",
            )
        return self.layer(features)

    def __call__(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(features)

    def parameters(self) -> Sequence[object]:
        if self._torch_layer is None:
            return []
        return list(self._torch_layer.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "torch_layer": None if self._torch_layer is None else self._torch_layer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        _load_torch_linear_state(
            layer_ref=self,
            attribute_name="_torch_layer",
            state_dict=state_dict.get("torch_layer"),
        )


class BinaryMLPClassificationHead:
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.1,
        seed: int = 230,
    ) -> None:
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.dropout = float(dropout)
        self.seed = seed
        layer_dims = (input_dim, *self.hidden_dims, 1)
        self.layers = [
            LinearLayer(layer_dims[index], layer_dims[index + 1], seed=seed + index)
            for index in range(len(layer_dims) - 1)
        ]
        self._torch_mlp = None

    def forward(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        if _is_torch_tensor(features):
            return _forward_torch_mlp(
                features,
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                seed=self.seed,
                layer_ref=self,
                attribute_name="_torch_mlp",
            )
        output = features
        for index, layer in enumerate(self.layers):
            output = layer(output)
            if index < len(self.layers) - 1:
                output = apply_activation(output, "relu")
        return output

    def __call__(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(features)

    def train(self, mode: bool = True) -> None:
        if self._torch_mlp is not None:
            self._torch_mlp.train(mode)

    def eval(self) -> None:
        self.train(False)

    def parameters(self) -> Sequence[object]:
        if self._torch_mlp is None:
            return []
        return list(self._torch_mlp.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "torch_mlp": None if self._torch_mlp is None else self._torch_mlp.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        _load_torch_mlp_state(
            layer_ref=self,
            attribute_name="_torch_mlp",
            state_dict=state_dict.get("torch_mlp"),
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )


class MultiLabelClassificationHead:
    def __init__(self, input_dim: int, num_labels: int, *, seed: int = 300) -> None:
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.seed = seed
        self.layer = LinearLayer(input_dim, num_labels, seed=seed)
        self._torch_layer = None

    def forward(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        if _is_torch_tensor(features):
            return _forward_torch_linear(
                features,
                input_dim=self.input_dim,
                output_dim=self.num_labels,
                seed=self.seed,
                layer_ref=self,
                attribute_name="_torch_layer",
            )
        return self.layer(features)

    def __call__(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(features)

    def parameters(self) -> Sequence[object]:
        if self._torch_layer is None:
            return []
        return list(self._torch_layer.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "torch_layer": None if self._torch_layer is None else self._torch_layer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        _load_torch_linear_state(
            layer_ref=self,
            attribute_name="_torch_layer",
            state_dict=state_dict.get("torch_layer"),
        )


class MultiLabelMLPHead:
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        *,
        hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.1,
        seed: int = 350,
    ) -> None:
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.dropout = float(dropout)
        self.seed = seed
        self.layers = []
        layer_dims = (input_dim, *self.hidden_dims, num_labels)
        for index in range(len(layer_dims) - 1):
            self.layers.append(LinearLayer(layer_dims[index], layer_dims[index + 1], seed=seed + index))
        self._torch_mlp = None

    def forward(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        if _is_torch_tensor(features):
            return _forward_torch_multilabel_mlp(
                features,
                input_dim=self.input_dim,
                output_dim=self.num_labels,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                seed=self.seed,
                layer_ref=self,
                attribute_name="_torch_mlp",
            )
        output = features
        for index, layer in enumerate(self.layers):
            output = layer(output)
            if index < len(self.layers) - 1:
                output = apply_activation(output, "relu")
        return output

    def __call__(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(features)

    def train(self, mode: bool = True) -> None:
        if self._torch_mlp is not None:
            self._torch_mlp.train(mode)

    def eval(self) -> None:
        self.train(False)

    def parameters(self) -> Sequence[object]:
        if self._torch_mlp is None:
            return []
        return list(self._torch_mlp.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "torch_mlp": None if self._torch_mlp is None else self._torch_mlp.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        _load_torch_multilabel_mlp_state(
            layer_ref=self,
            attribute_name="_torch_mlp",
            state_dict=state_dict.get("torch_mlp"),
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.num_labels,
            dropout=self.dropout,
        )


class RegressionHead:
    def __init__(self, input_dim: int, *, seed: int = 400) -> None:
        self.input_dim = input_dim
        self.seed = seed
        self.layer = LinearLayer(input_dim, 1, seed=seed)
        self._torch_layer = None

    def forward(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        if _is_torch_tensor(features):
            return _forward_torch_linear(
                features,
                input_dim=self.input_dim,
                output_dim=1,
                seed=self.seed,
                layer_ref=self,
                attribute_name="_torch_layer",
            )
        return self.layer(features)

    def __call__(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(features)

    def parameters(self) -> Sequence[object]:
        if self._torch_layer is None:
            return []
        return list(self._torch_layer.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "torch_layer": None if self._torch_layer is None else self._torch_layer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        _load_torch_linear_state(
            layer_ref=self,
            attribute_name="_torch_layer",
            state_dict=state_dict.get("torch_layer"),
        )


class MLPRegressionHead:
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (512, 128),
        dropout: float = 0.1,
        seed: int = 500,
    ) -> None:
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")
        self.input_dim = input_dim
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.dropout = float(dropout)
        self.seed = seed
        layer_dims = (input_dim, *self.hidden_dims, 1)
        self.layers = [
            LinearLayer(layer_dims[index], layer_dims[index + 1], seed=seed + index)
            for index in range(len(layer_dims) - 1)
        ]
        self._torch_mlp = None

    def forward(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        if _is_torch_tensor(features):
            return _forward_torch_mlp(
                features,
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                seed=self.seed,
                layer_ref=self,
                attribute_name="_torch_mlp",
            )
        output = features
        for index, layer in enumerate(self.layers):
            output = layer(output)
            if index < len(self.layers) - 1:
                output = apply_activation(output, "relu")
        return output

    def __call__(self, features: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(features)

    def train(self, mode: bool = True) -> None:
        if self._torch_mlp is not None:
            self._torch_mlp.train(mode)

    def eval(self) -> None:
        self.train(False)

    def parameters(self) -> Sequence[object]:
        if self._torch_mlp is None:
            return []
        return list(self._torch_mlp.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "torch_mlp": None if self._torch_mlp is None else self._torch_mlp.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        _load_torch_mlp_state(
            layer_ref=self,
            attribute_name="_torch_mlp",
            state_dict=state_dict.get("torch_mlp"),
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )


def _is_torch_tensor(value: object) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    torch = importlib.import_module("torch")
    return isinstance(value, torch.Tensor)


def _forward_torch_linear(
    features: object,
    *,
    input_dim: int,
    output_dim: int,
    seed: int,
    layer_ref: object,
    attribute_name: str,
) -> object:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    tensor = features
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Task head expects a 1D or 2D tensor input, got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[1] != input_dim:
        raise ValueError(
            f"Task head expected feature dim {input_dim}, got {tensor.shape[1]}."
        )

    tensor = tensor.to(dtype=torch.float32)
    layer = getattr(layer_ref, attribute_name)
    if layer is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layer = nn.Linear(input_dim, output_dim)
        layer = layer.to(device=tensor.device, dtype=tensor.dtype)
        setattr(layer_ref, attribute_name, layer)
    return layer(tensor)


def _load_torch_linear_state(
    *,
    layer_ref: object,
    attribute_name: str,
    state_dict: object,
) -> None:
    if state_dict is None:
        return
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Loading torch task-head state requires torch to be installed.")

    nn = importlib.import_module("torch.nn")
    weight = state_dict["weight"]
    layer = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
    ).to(device=weight.device, dtype=weight.dtype)
    layer.load_state_dict(state_dict)
    setattr(layer_ref, attribute_name, layer)


def _forward_torch_mlp(
    features: object,
    *,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    seed: int,
    layer_ref: object,
    attribute_name: str,
) -> object:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    tensor = features
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Task head expects a 1D or 2D tensor input, got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[1] != input_dim:
        raise ValueError(
            f"Task head expected feature dim {input_dim}, got {tensor.shape[1]}."
        )

    tensor = tensor.to(dtype=torch.float32)
    mlp = getattr(layer_ref, attribute_name)
    if mlp is None:
        layer_dims = (input_dim, *hidden_dims, 1)
        modules = []
        with torch.random.fork_rng(devices=[]):
            for index in range(len(layer_dims) - 1):
                torch.manual_seed(seed + index)
                linear = nn.Linear(layer_dims[index], layer_dims[index + 1])
                modules.append(linear)
                if index < len(layer_dims) - 2:
                    modules.append(nn.ReLU())
                    modules.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*modules)
        mlp = mlp.to(device=tensor.device, dtype=tensor.dtype)
        setattr(layer_ref, attribute_name, mlp)
    return mlp(tensor)


def _load_torch_mlp_state(
    *,
    layer_ref: object,
    attribute_name: str,
    state_dict: object,
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> None:
    if state_dict is None:
        return
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Loading torch task-head state requires torch to be installed.")

    nn = importlib.import_module("torch.nn")
    layer_dims = (input_dim, *hidden_dims, 1)
    modules = []
    for index in range(len(layer_dims) - 1):
        modules.append(nn.Linear(layer_dims[index], layer_dims[index + 1]))
        if index < len(layer_dims) - 2:
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*modules)

    first_weight = state_dict.get("0.weight")
    if first_weight is not None:
        mlp = mlp.to(device=first_weight.device, dtype=first_weight.dtype)
    mlp.load_state_dict(state_dict)
    setattr(layer_ref, attribute_name, mlp)


def _forward_torch_multilabel_mlp(
    features: object,
    *,
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    seed: int,
    layer_ref: object,
    attribute_name: str,
) -> object:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    tensor = features
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Task head expects a 1D or 2D tensor input, got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[1] != input_dim:
        raise ValueError(
            f"Task head expected feature dim {input_dim}, got {tensor.shape[1]}."
        )

    tensor = tensor.to(dtype=torch.float32)
    mlp = getattr(layer_ref, attribute_name)
    if mlp is None:
        layer_dims = (input_dim, *hidden_dims, output_dim)
        modules = []
        with torch.random.fork_rng(devices=[]):
            for index in range(len(layer_dims) - 1):
                torch.manual_seed(seed + index)
                linear = nn.Linear(layer_dims[index], layer_dims[index + 1])
                modules.append(linear)
                if index < len(layer_dims) - 2:
                    modules.append(nn.ReLU())
                    modules.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*modules)
        mlp = mlp.to(device=tensor.device, dtype=tensor.dtype)
        setattr(layer_ref, attribute_name, mlp)
    return mlp(tensor)


def _load_torch_multilabel_mlp_state(
    *,
    layer_ref: object,
    attribute_name: str,
    state_dict: object,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
) -> None:
    if state_dict is None:
        return
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Loading torch task-head state requires torch to be installed.")

    nn = importlib.import_module("torch.nn")
    layer_dims = (input_dim, *hidden_dims, output_dim)
    modules = []
    for index in range(len(layer_dims) - 1):
        modules.append(nn.Linear(layer_dims[index], layer_dims[index + 1]))
        if index < len(layer_dims) - 2:
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*modules)

    first_weight = state_dict.get("0.weight")
    if first_weight is not None:
        mlp = mlp.to(device=first_weight.device, dtype=first_weight.dtype)
    mlp.load_state_dict(state_dict)
    setattr(layer_ref, attribute_name, mlp)
