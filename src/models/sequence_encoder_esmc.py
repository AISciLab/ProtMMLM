from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence

from src.models.model_outputs import SequenceBackboneOutput, SequenceEncoderOutput
from src.utils.sequence import normalize_sequence


SUPPORTED_ESMC_MODELS = frozenset({"esmc_600m"})
SUPPORTED_POOLING_MODES = frozenset({"cls", "mean_pool"})
SUPPORTED_BACKEND_MODES = frozenset({"auto", "real"})
DEFAULT_MAX_SEQUENCE_LENGTH = 100


@dataclass(frozen=True)
class ESMCModelSpec:
    model_name: str
    checkpoint_relative_path: Path
    d_model: int
    n_heads: int
    n_layers: int


MODEL_SPECS = {
    "esmc_600m": ESMCModelSpec(
        model_name="esmc_600m",
        checkpoint_relative_path=Path("checkpoints/esmc-600m/data/weights/esmc_600m_2024_12_v0.pth"),
        d_model=1152,
        n_heads=18,
        n_layers=36,
    ),
}


class ESMCBackendProtocol(Protocol):
    def encode(self, sequences: Sequence[str]) -> SequenceBackboneOutput:
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


class SeqEncoderESMC:
    def __init__(
        self,
        model_name: str,
        *,
        pooling: str = "cls",
        backend_mode: str = "auto",
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        use_flash_attn: bool = False,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        backend: ESMCBackendProtocol | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        if model_name not in SUPPORTED_ESMC_MODELS:
            raise ValueError(
                f"Unsupported ESMC model '{model_name}'. "
                f"Expected one of: {sorted(SUPPORTED_ESMC_MODELS)}"
            )
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
        if max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {max_sequence_length}.")

        self.model_name = model_name
        self.pooling = pooling
        self.backend_mode = backend_mode
        self.device = device
        self.use_flash_attn = use_flash_attn
        self.max_sequence_length = int(max_sequence_length)
        self.repo_root = Path(repo_root) if repo_root is not None else _default_repo_root()
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self._backend = backend
        self._training = True
        self._trainable = True

    def forward(self, sequences: str | Sequence[str]) -> SequenceEncoderOutput:
        normalized_sequences = _coerce_and_normalize_sequences(
            sequences,
            max_sequence_length=self.max_sequence_length,
        )
        backbone_output = self._get_backend().encode(normalized_sequences)

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

    def __call__(self, sequences: str | Sequence[str]) -> SequenceEncoderOutput:
        return self.forward(sequences)

    def resolve_checkpoint_path(self) -> Path:
        return resolve_local_checkpoint_path(
            self.model_name,
            repo_root=self.repo_root,
            checkpoint_path=self.checkpoint_path,
        )

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

    def train_last_n_layers(self, num_layers: int) -> None:
        raise NotImplementedError(
            "Selective unfreezing is reserved for a later stage. "
            f"Requested num_layers={num_layers}."
        )

    def attach_adapter(self, adapter_name: str, **kwargs: Any) -> None:
        raise NotImplementedError(
            "LoRA/adapter integration is reserved for a later stage. "
            f"Requested adapter_name={adapter_name!r}."
        )

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
            raise TypeError("SeqEncoderESMC backend does not support load_state_dict().")

    def _get_backend(self) -> ESMCBackendProtocol:
        if self._backend is None:
            if self.backend_mode != "real":
                raise RuntimeError(
                    "SeqEncoderESMC requires an explicit fake backend injection for the default "
                    "fake-path workflow. Use backend_mode='real' to enable the local torch ESMC path."
                )
            self._backend = LocalESMCBackend(
                model_name=self.model_name,
                checkpoint_path=self.resolve_checkpoint_path(),
                device=self.device,
                use_flash_attn=self.use_flash_attn,
                repo_root=self.repo_root,
                training=self._training,
                trainable=self._trainable,
            )
        return self._backend

    def _get_backend_if_initialized(self) -> ESMCBackendProtocol | None:
        return self._backend


@dataclass
class LocalESMCBackend:
    model_name: str
    checkpoint_path: Path
    device: str = "cpu"
    use_flash_attn: bool = False
    repo_root: Path | None = None
    training: bool = True
    trainable: bool = True
    _model: Any | None = None

    def encode(self, sequences: Sequence[str]) -> SequenceBackboneOutput:
        model = self._load_model()
        torch = self._import_torch()

        # Extension point: drop this private helper once the vendored ESMC tokenizer path has
        # a stable public batching API that does not require internal helpers.
        sequence_tokens = model._tokenize(list(sequences))
        output = model(sequence_tokens=sequence_tokens)
        token_embeddings = output.embeddings
        if token_embeddings is None:
            raise RuntimeError("ESMC backend returned no token embeddings.")

        cls_embedding = token_embeddings[:, 0, :]
        mean_pooled_embedding = _mean_pool_without_special_tokens(
            token_embeddings=token_embeddings,
            sequence_tokens=sequence_tokens,
            pad_token_id=model.tokenizer.pad_token_id,
            cls_token_id=model.tokenizer.cls_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            torch_module=torch,
        )
        return SequenceBackboneOutput(
            token_embeddings=token_embeddings,
            cls_embedding=cls_embedding,
            mean_pooled_embedding=mean_pooled_embedding,
        )

    def freeze(self) -> None:
        self.trainable = False
        if self._model is not None:
            _set_requires_grad(self._model.parameters(), False)

    def unfreeze(self) -> None:
        self.trainable = True
        if self._model is not None:
            _set_requires_grad(self._model.parameters(), True)

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)
        if self._model is not None:
            self._model.train(self.training)

    def eval(self) -> None:
        self.train(False)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        self._ensure_runtime_dependencies()
        self._ensure_vendored_esm_on_sys_path()

        torch = self._import_torch()
        esmc_module = importlib.import_module("esm.models.esmc")
        sequence_tokenizer_module = importlib.import_module("esm.tokenization.sequence_tokenizer")

        model_spec = MODEL_SPECS[self.model_name]
        device = torch.device(self.device)
        tokenizer = sequence_tokenizer_module.EsmSequenceTokenizer()
        esmc_init_kwargs = {
            "d_model": model_spec.d_model,
            "n_heads": model_spec.n_heads,
            "n_layers": model_spec.n_layers,
            "tokenizer": tokenizer,
        }
        if "use_flash_attn" in inspect.signature(esmc_module.ESMC.__init__).parameters:
            esmc_init_kwargs["use_flash_attn"] = self.use_flash_attn
        model = esmc_module.ESMC(**esmc_init_kwargs)
        state_dict = torch.load(self.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model_dtype = _resolve_model_dtype(
            torch_module=torch,
            device=device,
            use_flash_attn=self.use_flash_attn,
        )
        model = model.to(device=device, dtype=model_dtype)
        _set_requires_grad(model.parameters(), self.trainable)
        model.train(self.training)
        self._model = model
        return model

    def _ensure_runtime_dependencies(self) -> None:
        missing_dependencies = missing_real_backend_dependencies()
        if missing_dependencies:
            raise RuntimeError(
                "Local ESMC backend requires additional runtime dependencies: "
                f"{', '.join(missing_dependencies)}"
            )

    def _ensure_vendored_esm_on_sys_path(self) -> None:
        repo_root = self.repo_root if self.repo_root is not None else _default_repo_root()
        vendored_package_root = repo_root / "esm"
        if not vendored_package_root.exists():
            raise FileNotFoundError(
                f"Vendored esm package root not found: {vendored_package_root}"
            )
        if str(vendored_package_root) not in sys.path:
            sys.path.insert(0, str(vendored_package_root))

    @staticmethod
    def _import_torch() -> Any:
        return importlib.import_module("torch")

    def parameters(self) -> Sequence[Any]:
        return list(self._load_model().parameters())

    def state_dict(self) -> dict[str, Any]:
        return dict(self._load_model().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._load_model().load_state_dict(state_dict)


def _set_requires_grad(parameters: Sequence[Any], requires_grad: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def _resolve_model_dtype(*, torch_module: Any, device: Any, use_flash_attn: bool) -> Any:
    if device.type != "cuda":
        return torch_module.float32
    if use_flash_attn:
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def resolve_local_checkpoint_path(
    model_name: str,
    *,
    repo_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path:
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported ESMC model '{model_name}'. "
            f"Expected one of: {sorted(SUPPORTED_ESMC_MODELS)}"
        )

    if checkpoint_path is not None:
        resolved_path = Path(checkpoint_path)
    else:
        root = Path(repo_root) if repo_root is not None else _default_repo_root()
        resolved_path = root / MODEL_SPECS[model_name].checkpoint_relative_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Local checkpoint for {model_name} was not found at {resolved_path}. "
            "Provide checkpoint_path explicitly or place the local weights there."
        )

    return resolved_path


def has_local_checkpoint(
    model_name: str,
    *,
    repo_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> bool:
    try:
        resolve_local_checkpoint_path(
            model_name,
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
        )
    except FileNotFoundError:
        return False
    return True


def missing_real_backend_dependencies() -> list[str]:
    required_modules = ("torch", "transformers", "tokenizers", "httpx", "attrs", "requests", "huggingface_hub")
    return [
        module_name
        for module_name in required_modules
        if importlib.util.find_spec(module_name) is None
    ]


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _coerce_and_normalize_sequences(
    sequences: str | Sequence[str],
    *,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> list[str]:
    if isinstance(sequences, str):
        sequence_list = [sequences]
    else:
        sequence_list = list(sequences)

    if not sequence_list:
        raise ValueError("At least one sequence is required.")

    return [
        _truncate_for_encoder(normalize_sequence(sequence), max_sequence_length=max_sequence_length)
        for sequence in sequence_list
    ]


def _mean_pool_without_special_tokens(
    *,
    token_embeddings: Any,
    sequence_tokens: Any,
    pad_token_id: int | None,
    cls_token_id: int | None,
    eos_token_id: int | None,
    torch_module: Any,
) -> Any:
    mask = sequence_tokens != pad_token_id
    if cls_token_id is not None:
        mask = mask & (sequence_tokens != cls_token_id)
    if eos_token_id is not None:
        mask = mask & (sequence_tokens != eos_token_id)

    mask = mask.unsqueeze(-1)
    masked_embeddings = token_embeddings * mask.to(token_embeddings.dtype)
    token_counts = mask.sum(dim=1).clamp(min=1)
    return masked_embeddings.sum(dim=1) / token_counts.to(token_embeddings.dtype)


def _truncate_for_encoder(sequence: str, *, max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH) -> str:
    # Keep manifest hashing on the full normalized sequence; cap only the encoder input.
    return sequence[:max_sequence_length]
