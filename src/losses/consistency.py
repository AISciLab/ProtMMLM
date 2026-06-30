from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence

from src.losses._utils import (
    apply_pair_mask,
    cosine_similarity,
    ensure_matrix,
    mean,
    mse,
)


def consistency_loss(
    h_missing: Sequence[float] | Sequence[Sequence[float]],
    h_full: Sequence[float] | Sequence[Sequence[float]],
    *,
    mode: str = "cosine",
    valid_mask: Sequence[bool] | None = None,
) -> float:
    if _uses_torch_tensors(h_missing, h_full):
        return _consistency_loss_torch(
            h_missing,
            h_full,
            mode=mode,
            valid_mask=valid_mask,
        )

    missing_matrix = ensure_matrix(h_missing, name="h_missing")
    full_matrix = ensure_matrix(h_full, name="h_full")
    missing_matrix, full_matrix = apply_pair_mask(missing_matrix, full_matrix, valid_mask)
    if not missing_matrix:
        return 0.0

    if mode == "cosine":
        return mean(
            [1.0 - cosine_similarity(missing_row, full_row) for missing_row, full_row in zip(missing_matrix, full_matrix)]
        )
    if mode == "mse":
        return mean(
            [mse(missing_row, full_row) for missing_row, full_row in zip(missing_matrix, full_matrix)]
        )

    raise ValueError(f"Unsupported consistency mode: {mode}")


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


def _consistency_loss_torch(
    h_missing: object,
    h_full: object,
    *,
    mode: str,
    valid_mask: Sequence[bool] | None,
) -> object:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    missing_tensor = _coerce_matrix_tensor(h_missing, name="h_missing", torch_module=torch)
    full_tensor = _coerce_matrix_tensor(
        h_full,
        name="h_full",
        torch_module=torch,
        device=missing_tensor.device,
    )
    if missing_tensor.shape != full_tensor.shape:
        raise ValueError(
            f"h_missing and h_full must have matching shapes, got {tuple(missing_tensor.shape)} and {tuple(full_tensor.shape)}."
        )

    mask_tensor = _coerce_mask_tensor(
        valid_mask,
        batch_size=int(missing_tensor.shape[0]),
        torch_module=torch,
        device=missing_tensor.device,
    )
    missing_tensor = missing_tensor[mask_tensor]
    full_tensor = full_tensor[mask_tensor].detach()
    if missing_tensor.shape[0] == 0:
        return (missing_tensor.sum() + full_tensor.sum()) * 0.0

    if mode == "cosine":
        return (1.0 - functional.cosine_similarity(missing_tensor, full_tensor, dim=-1)).mean()
    if mode == "mse":
        return functional.mse_loss(missing_tensor, full_tensor)
    raise ValueError(f"Unsupported consistency mode: {mode}")
