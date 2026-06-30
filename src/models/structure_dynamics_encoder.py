from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any, Protocol, Sequence

from src.datasets.structure_io import (
    build_structural_token_sequence,
    load_structural_token_cache,
)
from src.models.model_outputs import SequenceBackboneOutput, SequenceEncoderOutput


SUPPORTED_POOLING_MODES = frozenset({"cls", "mean_pool"})
SUPPORTED_BACKEND_MODES = frozenset({"auto", "real"})


class STBackboneProtocol(Protocol):
    def encode(
        self,
        token_features: Any,
        token_mask: Any | None = None,
    ) -> SequenceBackboneOutput:
        ...

    def freeze(self) -> None:
        ...

    def unfreeze(self) -> None:
        ...

    def train(self, mode: bool = True) -> None:
        ...

    def eval(self) -> None:
        ...

    def parameters(self) -> Sequence[Any]:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...


class STTransformer:
    def __init__(
        self,
        *,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling: str = "cls",
        backend_mode: str = "auto",
        backend: STBackboneProtocol | None = None,
        device: str = "cpu",
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if pooling not in SUPPORTED_POOLING_MODES:
            raise ValueError(
                f"Unsupported pooling '{pooling}'. "
                f"Expected one of: {sorted(SUPPORTED_POOLING_MODES)}"
            )
        if backend_mode not in SUPPORTED_BACKEND_MODES:
            raise ValueError(
                f"Unsupported backend_mode '{backend_mode}'. "
                f"Expected one of: {sorted(SUPPORTED_BACKEND_MODES)}"
            )

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.pooling = pooling
        self.backend_mode = backend_mode
        self.device = device
        self._backend = backend
        self._training = True
        self._trainable = True

    def forward(
        self,
        token_features: Any,
        token_mask: Any | None = None,
    ) -> SequenceEncoderOutput:
        backbone_output = self._get_backend().encode(token_features, token_mask)
        pooled_embedding = (
            backbone_output.cls_embedding
            if self.pooling == "cls"
            else backbone_output.mean_pooled_embedding
        )
        return SequenceEncoderOutput(
            token_embeddings=backbone_output.token_embeddings,
            cls_embedding=backbone_output.cls_embedding,
            pooled_embedding=pooled_embedding,
        )

    def __call__(
        self,
        token_features: Any,
        token_mask: Any | None = None,
    ) -> SequenceEncoderOutput:
        return self.forward(token_features, token_mask)

    def encode_paths(
        self,
        *,
        nature_path: str | None = None,
        md_path: str | None = None,
        max_residues: int = 100,
        max_frames: int = 160,
    ) -> SequenceEncoderOutput:
        structural_tokens = build_structural_token_sequence(
            nature_path=nature_path,
            md_path=md_path,
            max_residues=max_residues,
            max_frames=max_frames,
        )
        batch_features = [structural_tokens.token_features]
        batch_mask = [structural_tokens.token_mask]
        return self.forward(batch_features, batch_mask)

    def encode_cached_tokens(self, *, cache_path: str) -> SequenceEncoderOutput:
        structural_tokens = load_structural_token_cache(cache_path)
        batch_features = [structural_tokens.token_features]
        batch_mask = [structural_tokens.token_mask]
        return self.forward(batch_features, batch_mask)

    def freeze_backbone(self) -> None:
        self._trainable = False
        backend = self._get_backend_if_initialized()
        if backend is not None and hasattr(backend, "freeze"):
            backend.freeze()

    def unfreeze_backbone(self) -> None:
        self._trainable = True
        backend = self._get_backend_if_initialized()
        if backend is not None and hasattr(backend, "unfreeze"):
            backend.unfreeze()

    def train(self, mode: bool = True) -> None:
        self._training = bool(mode)
        backend = self._get_backend_if_initialized()
        if backend is not None and hasattr(backend, "train"):
            backend.train(self._training)

    def eval(self) -> None:
        self.train(False)

    def _get_backend(self) -> STBackboneProtocol:
        if self._backend is None:
            if self.backend_mode != "real":
                raise RuntimeError(
                    "STTransformer requires an explicit fake backend injection for the default "
                    "fake-path workflow. Use backend_mode='real' to enable the torch ST-Transformer path."
                )
            self._backend = TorchSTTransformerBackend(
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                device=self.device,
                training=self._training,
                trainable=self._trainable,
            )
        return self._backend

    def _get_backend_if_initialized(self) -> STBackboneProtocol | None:
        return self._backend

    def parameters(self) -> Sequence[Any]:
        backend = self._get_backend()
        if hasattr(backend, "parameters"):
            return list(backend.parameters())
        return []

    def state_dict(self) -> dict[str, Any]:
        backend = self._get_backend()
        if hasattr(backend, "state_dict"):
            return dict(backend.state_dict())
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        backend = self._get_backend()
        if hasattr(backend, "load_state_dict"):
            backend.load_state_dict(state_dict)
            return
        if state_dict:
            raise TypeError("STTransformer backend does not support load_state_dict().")


@dataclass
class TorchSTTransformerBackend:
    d_model: int
    num_layers: int
    num_heads: int
    dropout: float
    device: str = "cpu"
    training: bool = True
    trainable: bool = True
    _projection: Any | None = None
    _encoder: Any | None = None
    _cls_token: Any | None = None
    _device_obj: Any | None = None

    def encode(
        self,
        token_features: Any,
        token_mask: Any | None = None,
    ) -> SequenceBackboneOutput:
        self._ensure_runtime_dependencies()

        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")

        feature_tensor = _coerce_feature_tensor(token_features, torch)
        batch_size, token_count, feature_dim = feature_tensor.shape
        mask_tensor = _coerce_mask_tensor(token_mask, batch_size, token_count, torch)

        self._initialize_modules_if_needed(feature_dim, nn, torch)
        assert self._projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None
        assert self._device_obj is not None

        feature_tensor = feature_tensor.to(self._device_obj)
        mask_tensor = mask_tensor.to(self._device_obj)

        projected_tokens = self._projection(feature_tensor)
        cls_tokens = self._cls_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_tokens, projected_tokens], dim=1)

        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=self._device_obj)
        encoder_mask = torch.cat([cls_mask, mask_tensor], dim=1)

        encoded = self._encoder(
            encoder_input,
            src_key_padding_mask=~encoder_mask,
        )
        token_embeddings = encoded[:, 1:, :]
        cls_embedding = encoded[:, 0, :]
        mean_pooled_embedding = _masked_mean_pool(
            token_embeddings=token_embeddings,
            token_mask=mask_tensor,
            torch_module=torch,
        )
        return SequenceBackboneOutput(
            token_embeddings=token_embeddings,
            cls_embedding=cls_embedding,
            mean_pooled_embedding=mean_pooled_embedding,
        )

    def freeze(self) -> None:
        self.trainable = False
        if self._projection is None:
            return
        for parameter in self._iter_parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        self.trainable = True
        if self._projection is None:
            return
        for parameter in self._iter_parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)
        if self._projection is not None and hasattr(self._projection, "train"):
            self._projection.train(self.training)
        if self._encoder is not None and hasattr(self._encoder, "train"):
            self._encoder.train(self.training)

    def eval(self) -> None:
        self.train(False)

    def _initialize_modules_if_needed(self, feature_dim: int, nn: Any, torch: Any) -> None:
        if self._projection is not None:
            return

        device_obj = torch.device(self.device)
        self._device_obj = device_obj
        self._projection = nn.Linear(feature_dim, self.d_model).to(device_obj)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        ).to(device_obj)
        self._cls_token = nn.Parameter(
            torch.zeros((1, 1, self.d_model), device=device_obj)
        )
        self.train(self.training)
        for parameter in self._iter_parameters():
            parameter.requires_grad = self.trainable

    def _ensure_modules_initialized_for_freeze(self) -> None:
        if self._projection is None:
            raise RuntimeError(
                "ST-Transformer backend has not been initialized yet. "
                "Run a forward pass before toggling backbone freezing."
            )

    def _iter_parameters(self) -> Sequence[Any]:
        parameters: list[Any] = []
        assert self._projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None

        parameters.extend(self._projection.parameters())
        parameters.extend(self._encoder.parameters())
        parameters.append(self._cls_token)
        return parameters

    @staticmethod
    def _ensure_runtime_dependencies() -> None:
        missing_dependencies = missing_st_transformer_runtime_dependencies()
        if missing_dependencies:
            raise RuntimeError(
                "Torch ST-Transformer backend requires additional runtime dependencies: "
                f"{', '.join(missing_dependencies)}"
            )

    def parameters(self) -> Sequence[Any]:
        if self._projection is None:
            return []
        return list(self._iter_parameters())

    def state_dict(self) -> dict[str, Any]:
        if self._projection is None:
            return {}
        assert self._projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None
        return {
            "projection": self._projection.state_dict(),
            "encoder": self._encoder.state_dict(),
            "cls_token": self._cls_token.detach().clone(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return

        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")

        projection_state = state_dict["projection"]
        feature_dim = int(projection_state["weight"].shape[1])
        self._initialize_modules_if_needed(feature_dim, nn, torch)
        assert self._projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None
        assert self._device_obj is not None

        self._projection.load_state_dict(projection_state)
        self._encoder.load_state_dict(state_dict["encoder"])
        cls_token = state_dict["cls_token"].to(self._device_obj)
        with torch.no_grad():
            self._cls_token.copy_(cls_token)


def missing_st_transformer_runtime_dependencies() -> list[str]:
    return ["torch"] if importlib.util.find_spec("torch") is None else []


def _coerce_feature_tensor(token_features: Any, torch_module: Any) -> Any:
    tensor = torch_module.as_tensor(token_features, dtype=torch_module.float32)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            "token_features must be a 2D [tokens, features] or 3D "
            f"[batch, tokens, features] tensor-like object, got shape {tuple(tensor.shape)}."
        )
    return tensor


def _coerce_mask_tensor(
    token_mask: Any | None,
    batch_size: int,
    token_count: int,
    torch_module: Any,
) -> Any:
    if token_mask is None:
        return torch_module.ones(
            (batch_size, token_count),
            dtype=torch_module.bool,
        )

    mask_tensor = torch_module.as_tensor(token_mask, dtype=torch_module.bool)
    if mask_tensor.ndim == 1:
        mask_tensor = mask_tensor.unsqueeze(0)
    expected_shape = (batch_size, token_count)
    if tuple(mask_tensor.shape) != expected_shape:
        raise ValueError(
            f"token_mask must have shape {expected_shape}, got {tuple(mask_tensor.shape)}."
        )
    return mask_tensor


def _masked_mean_pool(
    *,
    token_embeddings: Any,
    token_mask: Any,
    torch_module: Any,
) -> Any:
    expanded_mask = token_mask.unsqueeze(-1)
    masked_embeddings = token_embeddings * expanded_mask.to(token_embeddings.dtype)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1)
    return masked_embeddings.sum(dim=1) / token_counts.to(token_embeddings.dtype)
