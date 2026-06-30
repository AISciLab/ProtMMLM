from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence

from src.models._head_utils import LinearLayer, Matrix, apply_activation


class ProjectionHead:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        activation: str = "tanh",
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.seed = seed

        if hidden_dim is None:
            self.input_layer = LinearLayer(input_dim, output_dim, seed=seed)
            self.output_layer = None
        else:
            self.input_layer = LinearLayer(input_dim, hidden_dim, seed=seed)
            self.output_layer = LinearLayer(hidden_dim, output_dim, seed=seed + 1)
        self._torch_input_layer = None
        self._torch_output_layer = None

    def forward(self, embeddings: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        if _is_torch_tensor(embeddings):
            return self._forward_torch(embeddings)

        hidden = self.input_layer(embeddings)
        if self.output_layer is None:
            return hidden

        activated_hidden = apply_activation(hidden, self.activation)
        return self.output_layer(activated_hidden)

    def __call__(self, embeddings: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(embeddings)

    def parameters(self) -> Sequence[object]:
        parameters: list[object] = []
        if self._torch_input_layer is not None:
            parameters.extend(self._torch_input_layer.parameters())
        if self._torch_output_layer is not None:
            parameters.extend(self._torch_output_layer.parameters())
        return parameters

    def state_dict(self) -> dict[str, object]:
        return {
            "input_layer": None if self._torch_input_layer is None else self._torch_input_layer.state_dict(),
            "output_layer": None if self._torch_output_layer is None else self._torch_output_layer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if not state_dict:
            return
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("Loading torch ProjectionHead state requires torch to be installed.")

        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")

        input_state = state_dict.get("input_layer")
        output_state = state_dict.get("output_layer")
        if input_state is not None:
            input_weight = input_state["weight"]
            self._torch_input_layer = nn.Linear(
                int(input_weight.shape[1]),
                int(input_weight.shape[0]),
            ).to(device=input_weight.device, dtype=input_weight.dtype)
            self._torch_input_layer.load_state_dict(input_state)
        if output_state is not None:
            output_weight = output_state["weight"]
            self._torch_output_layer = nn.Linear(
                int(output_weight.shape[1]),
                int(output_weight.shape[0]),
            ).to(device=output_weight.device, dtype=output_weight.dtype)
            self._torch_output_layer.load_state_dict(output_state)

    def _forward_torch(self, embeddings: object) -> object:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")

        tensor = embeddings
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"ProjectionHead expects a 1D or 2D tensor input, got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[1] != self.input_dim:
            raise ValueError(
                f"ProjectionHead expected feature dim {self.input_dim}, got {tensor.shape[1]}."
            )

        tensor = tensor.to(dtype=torch.float32)
        if self._torch_input_layer is None:
            self._torch_input_layer = _initialize_torch_linear_layer(
                nn=nn,
                input_dim=self.input_dim,
                output_dim=self.hidden_dim or self.output_dim,
                seed=self.seed,
                device=tensor.device,
                dtype=tensor.dtype,
                torch_module=torch,
            )

        hidden = self._torch_input_layer(tensor)
        if self.hidden_dim is None:
            return hidden

        hidden = _apply_torch_activation(hidden, self.activation, torch_module=torch)
        if self._torch_output_layer is None:
            assert self.hidden_dim is not None
            self._torch_output_layer = _initialize_torch_linear_layer(
                nn=nn,
                input_dim=self.hidden_dim,
                output_dim=self.output_dim,
                seed=self.seed + 1,
                device=tensor.device,
                dtype=tensor.dtype,
                torch_module=torch,
            )
        return self._torch_output_layer(hidden)


def _is_torch_tensor(value: object) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    torch = importlib.import_module("torch")
    return isinstance(value, torch.Tensor)


def _apply_torch_activation(tensor: object, activation: str, *, torch_module: object) -> object:
    if activation == "identity":
        return tensor
    if activation == "tanh":
        return torch_module.tanh(tensor)
    if activation == "relu":
        return torch_module.relu(tensor)
    raise ValueError(f"Unsupported activation: {activation}")


def _initialize_torch_linear_layer(
    *,
    nn: object,
    input_dim: int,
    output_dim: int,
    seed: int,
    device: object,
    dtype: object,
    torch_module: object,
) -> object:
    with torch_module.random.fork_rng(devices=[]):
        torch_module.manual_seed(seed)
        layer = nn.Linear(input_dim, output_dim)
    return layer.to(device=device, dtype=dtype)
