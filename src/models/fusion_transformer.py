from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Any, Protocol, Sequence


SUPPORTED_POOLING_MODES = frozenset({"cls", "mean_pool"})
SUPPORTED_BACKEND_MODES = frozenset({"auto", "real"})


@dataclass(frozen=True)
class FusionTransformerOutput:
    fused_cls: Any
    fused_pooled: Any
    fused_token_embeddings: Any | None = None


class FusionBackendProtocol(Protocol):
    def fuse(
        self,
        seq_embedding: Any,
        dyn_embedding: Any | None,
        has_dyn: Sequence[bool],
    ) -> FusionTransformerOutput:
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


class FusionTransformer:
    def __init__(
        self,
        *,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        pooling: str = "cls",
        backend_mode: str = "auto",
        use_availability_embedding: bool = True,
        backend: FusionBackendProtocol | None = None,
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
        self.use_availability_embedding = use_availability_embedding
        self.device = device
        self._backend = backend
        self._training = True
        self._trainable = True

    def forward(
        self,
        seq_embedding: Any,
        dyn_embedding: Any | None = None,
        has_dyn: bool | Sequence[bool] | None = None,
    ) -> FusionTransformerOutput:
        batch_size = _infer_batch_size(seq_embedding)
        normalized_has_dyn = _normalize_has_dyn(has_dyn, batch_size, dyn_embedding)

        if dyn_embedding is not None:
            dyn_batch_size = _infer_batch_size(dyn_embedding)
            if dyn_batch_size != batch_size:
                raise ValueError(
                    f"dyn_embedding batch size {dyn_batch_size} does not match "
                    f"seq_embedding batch size {batch_size}."
                )

        if any(normalized_has_dyn) and dyn_embedding is None:
            raise ValueError(
                "dyn_embedding is required when any sample has has_dyn=True."
            )

        return self._get_backend().fuse(
            seq_embedding,
            dyn_embedding,
            normalized_has_dyn,
        )

    def __call__(
        self,
        seq_embedding: Any,
        dyn_embedding: Any | None = None,
        has_dyn: bool | Sequence[bool] | None = None,
    ) -> FusionTransformerOutput:
        return self.forward(seq_embedding, dyn_embedding, has_dyn)

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

    def _get_backend(self) -> FusionBackendProtocol:
        if self._backend is None:
            if self.backend_mode != "real":
                raise RuntimeError(
                    "FusionTransformer requires an explicit fake backend injection for the default "
                    "fake-path workflow. Use backend_mode='real' to enable the torch fusion path."
                )
            self._backend = TorchFusionBackend(
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                pooling=self.pooling,
                use_availability_embedding=self.use_availability_embedding,
                device=self.device,
                training=self._training,
                trainable=self._trainable,
            )
        return self._backend

    def _get_backend_if_initialized(self) -> FusionBackendProtocol | None:
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
            raise TypeError("FusionTransformer backend does not support load_state_dict().")


@dataclass
class TorchFusionBackend:
    d_model: int
    num_layers: int
    num_heads: int
    dropout: float
    pooling: str = "cls"
    use_availability_embedding: bool = True
    device: str = "cpu"
    training: bool = True
    trainable: bool = True
    _seq_projection: Any | None = None
    _dyn_projection: Any | None = None
    _seq_input_dim: int | None = None
    _dyn_input_dim: int | None = None
    _encoder: Any | None = None
    _cls_token: Any | None = None
    _dyn_mask_token: Any | None = None
    _modality_type_embedding: Any | None = None
    _availability_embedding: Any | None = None
    _device_obj: Any | None = None

    def fuse(
        self,
        seq_embedding: Any,
        dyn_embedding: Any | None,
        has_dyn: Sequence[bool],
    ) -> FusionTransformerOutput:
        self._ensure_runtime_dependencies()

        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")

        seq_tensor = _coerce_embedding_tensor(seq_embedding, torch)
        batch_size, seq_dim = seq_tensor.shape
        has_dyn_tensor = _coerce_has_dyn_tensor(has_dyn, batch_size, torch)

        dyn_tensor = None
        dyn_dim = None
        if dyn_embedding is not None:
            dyn_tensor = _coerce_embedding_tensor(dyn_embedding, torch)
            if dyn_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"dyn_embedding batch size {dyn_tensor.shape[0]} does not match "
                    f"seq_embedding batch size {batch_size}."
                )
            dyn_dim = int(dyn_tensor.shape[1])

        self._initialize_modules_if_needed(seq_dim, dyn_dim, nn, torch)
        assert self._seq_projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None
        assert self._dyn_mask_token is not None
        assert self._modality_type_embedding is not None
        assert self._device_obj is not None

        seq_tensor = seq_tensor.to(self._device_obj)
        has_dyn_tensor = has_dyn_tensor.to(self._device_obj)
        seq_token = self._seq_projection(seq_tensor)

        if dyn_tensor is not None:
            dyn_tensor = dyn_tensor.to(self._device_obj)
            self._ensure_dyn_projection_initialized(dyn_tensor.shape[1], nn)
            assert self._dyn_projection is not None
            dyn_available_token = self._dyn_projection(dyn_tensor)
        else:
            dyn_available_token = None

        dyn_mask_token = self._dyn_mask_token.expand(batch_size, -1).to(self._device_obj)
        if dyn_available_token is None:
            dyn_token = dyn_mask_token
        else:
            dyn_token = torch.where(
                has_dyn_tensor.unsqueeze(-1),
                dyn_available_token,
                dyn_mask_token,
            )

        seq_modality_ids = torch.zeros((batch_size,), dtype=torch.long, device=self._device_obj)
        dyn_modality_ids = torch.ones((batch_size,), dtype=torch.long, device=self._device_obj)
        seq_token = seq_token + self._modality_type_embedding(seq_modality_ids)
        dyn_token = dyn_token + self._modality_type_embedding(dyn_modality_ids)

        if self.use_availability_embedding:
            assert self._availability_embedding is not None
            available_ids = torch.ones((batch_size,), dtype=torch.long, device=self._device_obj)
            seq_token = seq_token + self._availability_embedding(available_ids)
            dyn_token = dyn_token + self._availability_embedding(has_dyn_tensor.long())

        branch_tokens = torch.stack([seq_token, dyn_token], dim=1)
        cls_token = self._cls_token.expand(batch_size, -1, -1).to(self._device_obj)
        encoder_input = torch.cat([cls_token, branch_tokens], dim=1)

        encoded = self._encoder(encoder_input)
        fused_cls = encoded[:, 0, :]
        fused_token_embeddings = encoded[:, 1:, :]
        fused_pooled = (
            fused_cls
            if self.pooling == "cls"
            else fused_token_embeddings.mean(dim=1)
        )
        return FusionTransformerOutput(
            fused_cls=fused_cls,
            fused_pooled=fused_pooled,
            fused_token_embeddings=fused_token_embeddings,
        )

    def freeze(self) -> None:
        self.trainable = False
        if self._seq_projection is None:
            return
        for parameter in self._iter_parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        self.trainable = True
        if self._seq_projection is None:
            return
        for parameter in self._iter_parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)
        for module in (
            self._seq_projection,
            self._dyn_projection,
            self._encoder,
            self._modality_type_embedding,
            self._availability_embedding,
        ):
            if module is not None and hasattr(module, "train"):
                module.train(self.training)

    def eval(self) -> None:
        self.train(False)

    def _initialize_modules_if_needed(
        self,
        seq_dim: int,
        dyn_dim: int | None,
        nn: Any,
        torch: Any,
    ) -> None:
        if self._seq_projection is None:
            device_obj = torch.device(self.device)
            self._device_obj = device_obj
            self._seq_input_dim = seq_dim
            self._seq_projection = nn.Linear(seq_dim, self.d_model).to(device_obj)
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
            self._dyn_mask_token = nn.Parameter(
                torch.zeros((1, self.d_model), device=device_obj)
            )
            self._modality_type_embedding = nn.Embedding(2, self.d_model).to(device_obj)
            if self.use_availability_embedding:
                self._availability_embedding = nn.Embedding(2, self.d_model).to(device_obj)
            self.train(self.training)
            for parameter in self._iter_parameters():
                parameter.requires_grad = self.trainable
        elif self._seq_input_dim != seq_dim:
            raise ValueError(
                f"seq_embedding feature dim {seq_dim} does not match initialized "
                f"fusion seq feature dim {self._seq_input_dim}."
            )

        if dyn_dim is not None:
            self._ensure_dyn_projection_initialized(dyn_dim, nn)

    def _ensure_dyn_projection_initialized(self, dyn_dim: int, nn: Any) -> None:
        if self._dyn_projection is None:
            assert self._device_obj is not None
            self._dyn_input_dim = dyn_dim
            self._dyn_projection = nn.Linear(dyn_dim, self.d_model).to(self._device_obj)
            self._dyn_projection.train(self.training)
            for parameter in self._dyn_projection.parameters():
                parameter.requires_grad = self.trainable
            return
        if self._dyn_input_dim != dyn_dim:
            raise ValueError(
                f"dyn_embedding feature dim {dyn_dim} does not match initialized "
                f"fusion dyn feature dim {self._dyn_input_dim}."
            )

    def _ensure_modules_initialized_for_freeze(self) -> None:
        if self._seq_projection is None:
            raise RuntimeError(
                "Fusion backend has not been initialized yet. "
                "Run a forward pass before toggling backbone freezing."
            )

    def _iter_parameters(self) -> list[Any]:
        parameters: list[Any] = []
        assert self._seq_projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None
        assert self._dyn_mask_token is not None
        assert self._modality_type_embedding is not None

        parameters.extend(self._seq_projection.parameters())
        if self._dyn_projection is not None:
            parameters.extend(self._dyn_projection.parameters())
        parameters.extend(self._encoder.parameters())
        parameters.append(self._cls_token)
        parameters.append(self._dyn_mask_token)
        parameters.extend(self._modality_type_embedding.parameters())
        if self._availability_embedding is not None:
            parameters.extend(self._availability_embedding.parameters())
        return parameters

    @staticmethod
    def _ensure_runtime_dependencies() -> None:
        missing_dependencies = missing_fusion_transformer_runtime_dependencies()
        if missing_dependencies:
            raise RuntimeError(
                "Torch Fusion backend requires additional runtime dependencies: "
                f"{', '.join(missing_dependencies)}"
            )

    def parameters(self) -> Sequence[Any]:
        if self._seq_projection is None:
            return []
        return list(self._iter_parameters())

    def state_dict(self) -> dict[str, Any]:
        if self._seq_projection is None:
            return {}
        assert self._seq_projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None
        assert self._dyn_mask_token is not None
        assert self._modality_type_embedding is not None
        return {
            "seq_projection": self._seq_projection.state_dict(),
            "dyn_projection": None if self._dyn_projection is None else self._dyn_projection.state_dict(),
            "encoder": self._encoder.state_dict(),
            "cls_token": self._cls_token.detach().clone(),
            "dyn_mask_token": self._dyn_mask_token.detach().clone(),
            "modality_type_embedding": self._modality_type_embedding.state_dict(),
            "availability_embedding": None
            if self._availability_embedding is None
            else self._availability_embedding.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return

        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")

        seq_state = state_dict["seq_projection"]
        dyn_state = state_dict.get("dyn_projection")
        seq_dim = int(seq_state["weight"].shape[1])
        dyn_dim = None if dyn_state is None else int(dyn_state["weight"].shape[1])
        self._initialize_modules_if_needed(seq_dim, dyn_dim, nn, torch)
        assert self._seq_projection is not None
        assert self._encoder is not None
        assert self._cls_token is not None
        assert self._dyn_mask_token is not None
        assert self._modality_type_embedding is not None
        assert self._device_obj is not None

        self._seq_projection.load_state_dict(seq_state)
        if dyn_state is not None:
            self._ensure_dyn_projection_initialized(dyn_dim, nn)
            assert self._dyn_projection is not None
            self._dyn_projection.load_state_dict(dyn_state)
        self._encoder.load_state_dict(state_dict["encoder"])
        self._modality_type_embedding.load_state_dict(state_dict["modality_type_embedding"])
        with torch.no_grad():
            self._cls_token.copy_(state_dict["cls_token"].to(self._device_obj))
            self._dyn_mask_token.copy_(state_dict["dyn_mask_token"].to(self._device_obj))
        availability_state = state_dict.get("availability_embedding")
        if availability_state is not None:
            assert self._availability_embedding is not None
            self._availability_embedding.load_state_dict(availability_state)


def missing_fusion_transformer_runtime_dependencies() -> list[str]:
    return ["torch"] if importlib.util.find_spec("torch") is None else []


def _infer_batch_size(embedding: Any) -> int:
    shape = getattr(embedding, "shape", None)
    if shape is not None:
        if len(shape) == 1:
            return 1
        if len(shape) >= 2:
            return int(shape[0])

    if isinstance(embedding, Sequence) and not isinstance(embedding, (str, bytes)):
        if not embedding:
            raise ValueError("Embedding batch cannot be empty.")
        first = embedding[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return len(embedding)
        return 1

    raise ValueError(
        "Unable to infer batch size from embedding input. "
        "Expected tensor-like input or a 1D/2D sequence."
    )


def _normalize_has_dyn(
    has_dyn: bool | Sequence[bool] | None,
    batch_size: int,
    dyn_embedding: Any | None,
) -> list[bool]:
    if has_dyn is None:
        inferred_value = dyn_embedding is not None
        return [inferred_value] * batch_size

    if isinstance(has_dyn, bool):
        return [has_dyn] * batch_size

    normalized = [bool(value) for value in has_dyn]
    if len(normalized) != batch_size:
        raise ValueError(
            f"has_dyn must have length {batch_size}, got {len(normalized)}."
        )
    return normalized


def _coerce_embedding_tensor(embedding: Any, torch_module: Any) -> Any:
    tensor = torch_module.as_tensor(embedding, dtype=torch_module.float32)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            "Branch embeddings must be a 1D [features] or 2D [batch, features] "
            f"tensor-like object, got shape {tuple(tensor.shape)}."
        )
    return tensor


def _coerce_has_dyn_tensor(
    has_dyn: Sequence[bool],
    batch_size: int,
    torch_module: Any,
) -> Any:
    tensor = torch_module.as_tensor(has_dyn, dtype=torch_module.bool)
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(
            f"has_dyn must have shape ({batch_size},), got {tuple(tensor.shape)}."
        )
    return tensor
