from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence

from src.losses._utils import apply_pair_mask, ensure_matrix, mean, mse


def reconstruction_loss(
    predicted_dyn: Sequence[float] | Sequence[Sequence[float]],
    target_dyn: Sequence[float] | Sequence[Sequence[float]],
    *,
    valid_mask: Sequence[bool] | None = None,
) -> float:
    if _uses_torch_tensors(predicted_dyn, target_dyn):
        return _reconstruction_loss_torch(
            predicted_dyn,
            target_dyn,
            valid_mask=valid_mask,
        )

    predicted_matrix = ensure_matrix(predicted_dyn, name="predicted_dyn")
    target_matrix = ensure_matrix(target_dyn, name="target_dyn")
    predicted_matrix, target_matrix = apply_pair_mask(
        predicted_matrix,
        target_matrix,
        valid_mask,
    )
    if not predicted_matrix:
        return 0.0

    return mean(
        [mse(predicted_row, target_row) for predicted_row, target_row in zip(predicted_matrix, target_matrix)]
    )


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


def _reconstruction_loss_torch(
    predicted_dyn: object,
    target_dyn: object,
    *,
    valid_mask: Sequence[bool] | None,
) -> object:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    predicted_tensor = _coerce_matrix_tensor(
        predicted_dyn,
        name="predicted_dyn",
        torch_module=torch,
    )
    target_tensor = _coerce_matrix_tensor(
        target_dyn,
        name="target_dyn",
        torch_module=torch,
        device=predicted_tensor.device,
    )
    if predicted_tensor.shape != target_tensor.shape:
        raise ValueError(
            "predicted_dyn and target_dyn must have matching shapes, "
            f"got {tuple(predicted_tensor.shape)} and {tuple(target_tensor.shape)}."
        )

    mask_tensor = _coerce_mask_tensor(
        valid_mask,
        batch_size=int(predicted_tensor.shape[0]),
        torch_module=torch,
        device=predicted_tensor.device,
    )
    predicted_tensor = predicted_tensor[mask_tensor]
    target_tensor = target_tensor[mask_tensor].detach()
    if predicted_tensor.shape[0] == 0:
        return (predicted_tensor.sum() + target_tensor.sum()) * 0.0
    return functional.mse_loss(predicted_tensor, target_tensor)
