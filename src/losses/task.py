from __future__ import annotations

import importlib
import importlib.util
from typing import Sequence

from src.losses._utils import (
    apply_sample_mask,
    binary_cross_entropy_with_logits,
    ensure_mask,
    ensure_matrix,
    ensure_vector,
    huber,
    mean,
)


def binary_classification_loss(
    logits: Sequence[float] | Sequence[Sequence[float]],
    targets: Sequence[float] | Sequence[Sequence[float]],
    *,
    valid_mask: Sequence[bool] | None = None,
) -> float:
    if _uses_torch_tensors(logits, targets):
        return _binary_classification_loss_torch(
            logits,
            targets,
            valid_mask=valid_mask,
        )

    logit_matrix = ensure_matrix(logits, name="logits")
    if any(len(row) != 1 for row in logit_matrix):
        raise ValueError("Binary classification logits must have shape [batch, 1].")

    target_values = _coerce_target_values(targets, name="targets")
    if len(logit_matrix) != len(target_values):
        raise ValueError("Binary classification logits and targets must have the same batch size.")

    mask = ensure_mask(valid_mask, batch_size=len(logit_matrix))
    losses = [
        binary_cross_entropy_with_logits(logit_row[0], target_value)
        for logit_row, target_value, keep in zip(logit_matrix, target_values, mask)
        if keep
    ]
    return mean(losses)


def multilabel_classification_loss(
    logits: Sequence[float] | Sequence[Sequence[float]],
    targets: Sequence[float] | Sequence[Sequence[float]],
    *,
    valid_mask: Sequence[bool] | None = None,
    pos_weight_mode: str = "none",
    max_pos_weight: float = 20.0,
) -> float:
    if _uses_torch_tensors(logits, targets):
        return _multilabel_classification_loss_torch(
            logits,
            targets,
            valid_mask=valid_mask,
            pos_weight_mode=pos_weight_mode,
            max_pos_weight=max_pos_weight,
        )

    logit_matrix = ensure_matrix(logits, name="logits")
    target_matrix = ensure_matrix(targets, name="targets")
    if len(logit_matrix) != len(target_matrix):
        raise ValueError("Multilabel logits and targets must have the same batch size.")
    if any(len(logit_row) != len(target_row) for logit_row, target_row in zip(logit_matrix, target_matrix)):
        raise ValueError("Multilabel logits and targets must have matching label dimensions.")

    filtered_logits = apply_sample_mask(logit_matrix, valid_mask)
    filtered_targets = apply_sample_mask(target_matrix, valid_mask)
    if not filtered_logits:
        return 0.0

    pos_weights = _multilabel_pos_weights(
        filtered_targets,
        mode=pos_weight_mode,
        max_pos_weight=max_pos_weight,
    )
    per_sample_losses = []
    for logit_row, target_row in zip(filtered_logits, filtered_targets):
        losses = []
        for label_index, (logit_value, target_value) in enumerate(zip(logit_row, target_row)):
            loss = binary_cross_entropy_with_logits(logit_value, target_value)
            if float(target_value) >= 0.5:
                loss *= pos_weights[label_index]
            losses.append(loss)
        per_sample_losses.append(mean(losses))
    return mean(per_sample_losses)


def regression_loss(
    predictions: Sequence[float] | Sequence[Sequence[float]],
    targets: Sequence[float] | Sequence[Sequence[float]],
    *,
    valid_mask: Sequence[bool] | None = None,
    mode: str = "huber",
    delta: float = 1.0,
) -> float:
    if _uses_torch_tensors(predictions, targets):
        return _regression_loss_torch(
            predictions,
            targets,
            valid_mask=valid_mask,
            mode=mode,
            delta=delta,
        )

    prediction_matrix = ensure_matrix(predictions, name="predictions")
    if any(len(row) != 1 for row in prediction_matrix):
        raise ValueError("Regression predictions must have shape [batch, 1].")

    target_values = _coerce_target_values(targets, name="targets")
    if len(prediction_matrix) != len(target_values):
        raise ValueError("Regression predictions and targets must have the same batch size.")

    mask = ensure_mask(valid_mask, batch_size=len(prediction_matrix))
    losses: list[float] = []
    for prediction_row, target_value, keep in zip(prediction_matrix, target_values, mask):
        if not keep:
            continue
        error = prediction_row[0] - target_value
        if mode == "huber":
            losses.append(huber(error, delta=delta))
        elif mode == "mse":
            losses.append(error * error)
        else:
            raise ValueError(f"Unsupported regression loss mode: {mode}")

    return mean(losses)


def _coerce_target_values(
    targets: Sequence[float] | Sequence[Sequence[float]],
    *,
    name: str,
) -> list[float]:
    if not targets:
        raise ValueError(f"{name} cannot be empty.")

    first = targets[0]  # type: ignore[index]
    if isinstance(first, (list, tuple)):
        matrix = ensure_matrix(targets, name=name)
        if any(len(row) != 1 for row in matrix):
            raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
        return [row[0] for row in matrix]

    return [float(value) for value in targets]  # type: ignore[arg-type]


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


def _coerce_target_vector_tensor(
    targets: object,
    *,
    name: str,
    torch_module: object,
    device: object,
) -> object:
    if isinstance(targets, torch_module.Tensor):
        tensor = targets
    else:
        tensor = torch_module.as_tensor(targets, dtype=torch_module.float32, device=device)
    if tensor.ndim == 2:
        if tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D or 2D [batch, 1], got shape {tuple(tensor.shape)}.")
    return tensor.to(device=device, dtype=torch_module.float32)


def _coerce_target_matrix_tensor(
    targets: object,
    *,
    name: str,
    torch_module: object,
    device: object,
) -> object:
    if isinstance(targets, torch_module.Tensor):
        tensor = targets
    else:
        tensor = torch_module.as_tensor(targets, dtype=torch_module.float32, device=device)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor-like input, got shape {tuple(tensor.shape)}.")
    return tensor.to(device=device, dtype=torch_module.float32)


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


def _binary_classification_loss_torch(
    logits: object,
    targets: object,
    *,
    valid_mask: Sequence[bool] | None,
) -> object:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    logit_tensor = _coerce_matrix_tensor(logits, name="logits", torch_module=torch)
    if logit_tensor.shape[1] != 1:
        raise ValueError("Binary classification logits must have shape [batch, 1].")
    target_tensor = _coerce_target_vector_tensor(
        targets,
        name="targets",
        torch_module=torch,
        device=logit_tensor.device,
    )
    if logit_tensor.shape[0] != target_tensor.shape[0]:
        raise ValueError("Binary classification logits and targets must have the same batch size.")

    mask_tensor = _coerce_mask_tensor(
        valid_mask,
        batch_size=int(logit_tensor.shape[0]),
        torch_module=torch,
        device=logit_tensor.device,
    )
    filtered_logits = logit_tensor.squeeze(-1)[mask_tensor]
    filtered_targets = target_tensor[mask_tensor]
    if filtered_logits.numel() == 0:
        return logit_tensor.sum() * 0.0
    return functional.binary_cross_entropy_with_logits(filtered_logits, filtered_targets)


def _multilabel_classification_loss_torch(
    logits: object,
    targets: object,
    *,
    valid_mask: Sequence[bool] | None,
    pos_weight_mode: str,
    max_pos_weight: float,
) -> object:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    logit_tensor = _coerce_matrix_tensor(logits, name="logits", torch_module=torch)
    target_tensor = _coerce_target_matrix_tensor(
        targets,
        name="targets",
        torch_module=torch,
        device=logit_tensor.device,
    )
    if logit_tensor.shape != target_tensor.shape:
        raise ValueError(
            f"Multilabel logits and targets must have matching shapes, got {tuple(logit_tensor.shape)} and {tuple(target_tensor.shape)}."
        )

    mask_tensor = _coerce_mask_tensor(
        valid_mask,
        batch_size=int(logit_tensor.shape[0]),
        torch_module=torch,
        device=logit_tensor.device,
    )
    filtered_logits = logit_tensor[mask_tensor]
    filtered_targets = target_tensor[mask_tensor]
    if filtered_logits.shape[0] == 0:
        return logit_tensor.sum() * 0.0
    pos_weight = _multilabel_pos_weight_tensor(
        filtered_targets,
        mode=pos_weight_mode,
        max_pos_weight=max_pos_weight,
        torch_module=torch,
    )
    return functional.binary_cross_entropy_with_logits(
        filtered_logits,
        filtered_targets,
        pos_weight=pos_weight,
    )


def _multilabel_pos_weights(
    targets: list[list[float]],
    *,
    mode: str,
    max_pos_weight: float,
) -> list[float]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"", "none", "false", "off"}:
        return [1.0 for _ in range(len(targets[0]))]
    if normalized_mode != "batch":
        raise ValueError(f"Unsupported multilabel pos_weight_mode={mode!r}. Expected none or batch.")

    num_labels = len(targets[0])
    weights: list[float] = []
    for label_index in range(num_labels):
        positives = sum(1 for row in targets if float(row[label_index]) >= 0.5)
        negatives = len(targets) - positives
        if positives <= 0:
            weights.append(1.0)
        else:
            weights.append(min(float(max_pos_weight), max(1.0, negatives / float(positives))))
    return weights


def _multilabel_pos_weight_tensor(
    target_tensor: object,
    *,
    mode: str,
    max_pos_weight: float,
    torch_module: object,
) -> object | None:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"", "none", "false", "off"}:
        return None
    if normalized_mode != "batch":
        raise ValueError(f"Unsupported multilabel pos_weight_mode={mode!r}. Expected none or batch.")

    positives = (target_tensor >= 0.5).sum(dim=0).to(dtype=torch_module.float32)
    total = torch_module.tensor(
        float(target_tensor.shape[0]),
        dtype=torch_module.float32,
        device=target_tensor.device,
    )
    negatives = total - positives
    weights = negatives / positives.clamp_min(1.0)
    weights = weights.clamp(min=1.0, max=float(max_pos_weight))
    weights = torch_module.where(positives > 0.0, weights, torch_module.ones_like(weights))
    return weights.to(device=target_tensor.device, dtype=torch_module.float32)


def _regression_loss_torch(
    predictions: object,
    targets: object,
    *,
    valid_mask: Sequence[bool] | None,
    mode: str,
    delta: float,
) -> object:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")

    prediction_tensor = _coerce_matrix_tensor(predictions, name="predictions", torch_module=torch)
    if prediction_tensor.shape[1] != 1:
        raise ValueError("Regression predictions must have shape [batch, 1].")
    target_tensor = _coerce_target_vector_tensor(
        targets,
        name="targets",
        torch_module=torch,
        device=prediction_tensor.device,
    )
    if prediction_tensor.shape[0] != target_tensor.shape[0]:
        raise ValueError("Regression predictions and targets must have the same batch size.")

    mask_tensor = _coerce_mask_tensor(
        valid_mask,
        batch_size=int(prediction_tensor.shape[0]),
        torch_module=torch,
        device=prediction_tensor.device,
    )
    filtered_predictions = prediction_tensor.squeeze(-1)[mask_tensor]
    filtered_targets = target_tensor[mask_tensor]
    if filtered_predictions.numel() == 0:
        return prediction_tensor.sum() * 0.0

    if mode == "huber":
        return functional.huber_loss(filtered_predictions, filtered_targets, delta=delta)
    if mode == "mse":
        return functional.mse_loss(filtered_predictions, filtered_targets)
    raise ValueError(f"Unsupported regression loss mode: {mode}")
