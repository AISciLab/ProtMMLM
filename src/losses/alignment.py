from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence

from src.losses._utils import (
    apply_pair_mask,
    dot,
    ensure_matrix,
    l2_norm,
    logsumexp,
    mean,
)


def info_nce_loss(
    z_seq: Sequence[float] | Sequence[Sequence[float]],
    z_dyn: Sequence[float] | Sequence[Sequence[float]],
    *,
    temperature: float = 0.1,
    valid_mask: Sequence[bool] | None = None,
    symmetric: bool = True,
    normalize: bool = True,
) -> float:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    if _uses_torch_tensors(z_seq, z_dyn):
        return _info_nce_loss_torch(
            z_seq,
            z_dyn,
            temperature=temperature,
            valid_mask=valid_mask,
            symmetric=symmetric,
            normalize=normalize,
        )

    seq_matrix = ensure_matrix(z_seq, name="z_seq")
    dyn_matrix = ensure_matrix(z_dyn, name="z_dyn")
    seq_matrix, dyn_matrix = apply_pair_mask(seq_matrix, dyn_matrix, valid_mask)
    if not seq_matrix:
        return 0.0

    if normalize:
        seq_matrix = [_normalize_vector(row) for row in seq_matrix]
        dyn_matrix = [_normalize_vector(row) for row in dyn_matrix]

    forward_loss = _directional_info_nce(seq_matrix, dyn_matrix, temperature=temperature)
    if not symmetric:
        return forward_loss

    backward_loss = _directional_info_nce(dyn_matrix, seq_matrix, temperature=temperature)
    return 0.5 * (forward_loss + backward_loss)


def _directional_info_nce(
    queries: list[list[float]],
    targets: list[list[float]],
    *,
    temperature: float,
) -> float:
    losses: list[float] = []
    for index, query in enumerate(queries):
        similarities = [
            dot(query, target) / temperature
            for target in targets
        ]
        losses.append(-similarities[index] + logsumexp(similarities))
    return mean(losses)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = l2_norm(vector)
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _uses_torch_tensors(*values: object) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    torch = importlib.import_module("torch")
    return any(isinstance(value, torch.Tensor) for value in values)


def _coerce_matrix_tensor(
    values: object,
    *,
    name: str,
    torch_module: object,
    device: object | None = None,
) -> object:
    if isinstance(values, torch_module.Tensor):
        tensor = values
    else:
        tensor = torch_module.as_tensor(values, dtype=torch_module.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D tensor-like input, got shape {tuple(tensor.shape)}.")
    return tensor.to(dtype=torch_module.float32)


def _coerce_mask_tensor(
    valid_mask: Sequence[bool] | None,
    *,
    batch_size: int,
    torch_module: object,
    device: object,
) -> object:
    if valid_mask is None:
        return torch_module.ones((batch_size,), dtype=torch_module.bool, device=device)
    tensor = torch_module.as_tensor(valid_mask, dtype=torch_module.bool, device=device)
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"valid_mask must have shape ({batch_size},), got {tuple(tensor.shape)}.")
    return tensor


def _info_nce_loss_torch(
    z_seq: object,
    z_dyn: object,
    *,
    temperature: float,
    valid_mask: Sequence[bool] | None,
    symmetric: bool,
    normalize: bool,
) -> object:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    seq_tensor = _coerce_matrix_tensor(z_seq, name="z_seq", torch_module=torch)
    dyn_tensor = _coerce_matrix_tensor(
        z_dyn,
        name="z_dyn",
        torch_module=torch,
        device=seq_tensor.device,
    )
    if seq_tensor.shape != dyn_tensor.shape:
        raise ValueError(
            f"z_seq and z_dyn must have matching shapes, got {tuple(seq_tensor.shape)} and {tuple(dyn_tensor.shape)}."
        )

    mask_tensor = _coerce_mask_tensor(
        valid_mask,
        batch_size=int(seq_tensor.shape[0]),
        torch_module=torch,
        device=seq_tensor.device,
    )
    seq_tensor = seq_tensor[mask_tensor]
    dyn_tensor = dyn_tensor[mask_tensor]
    if seq_tensor.shape[0] == 0:
        return (seq_tensor.sum() + dyn_tensor.sum()) * 0.0

    if normalize:
        seq_tensor = functional.normalize(seq_tensor, dim=-1)
        dyn_tensor = functional.normalize(dyn_tensor, dim=-1)

    logits = seq_tensor @ dyn_tensor.transpose(0, 1)
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    forward_loss = functional.cross_entropy(logits, labels)
    if not symmetric:
        return forward_loss

    backward_loss = functional.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (forward_loss + backward_loss)
