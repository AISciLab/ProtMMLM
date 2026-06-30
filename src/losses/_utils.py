from __future__ import annotations

import math
from typing import Iterable, List, Sequence


Matrix = List[List[float]]
Vector = List[float]


def ensure_matrix(values: Sequence[float] | Sequence[Sequence[float]], *, name: str) -> Matrix:
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    first = values[0]  # type: ignore[index]
    if isinstance(first, (list, tuple)):
        matrix = [[float(component) for component in row] for row in values]  # type: ignore[arg-type]
    else:
        matrix = [[float(component) for component in values]]  # type: ignore[arg-type]

    feature_dim = len(matrix[0])
    if feature_dim == 0:
        raise ValueError(f"{name} rows cannot be empty.")
    for row in matrix:
        if len(row) != feature_dim:
            raise ValueError(f"{name} must have consistent row lengths.")

    return matrix


def ensure_vector(values: Sequence[float] | Sequence[Sequence[float]], *, name: str) -> Vector:
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    first = values[0]  # type: ignore[index]
    if isinstance(first, (list, tuple)):
        if len(values) != 1:  # type: ignore[arg-type]
            raise ValueError(f"{name} must be 1D, got multiple rows.")
        return [float(component) for component in first]  # type: ignore[arg-type]
    return [float(component) for component in values]  # type: ignore[arg-type]


def ensure_mask(valid_mask: Sequence[bool] | None, *, batch_size: int) -> list[bool]:
    if valid_mask is None:
        return [True] * batch_size

    mask = [bool(value) for value in valid_mask]
    if len(mask) != batch_size:
        raise ValueError(f"valid_mask must have length {batch_size}, got {len(mask)}.")
    return mask


def apply_sample_mask(matrix: Matrix, valid_mask: Sequence[bool] | None) -> Matrix:
    mask = ensure_mask(valid_mask, batch_size=len(matrix))
    return [row for row, keep in zip(matrix, mask) if keep]


def apply_pair_mask(
    first: Matrix,
    second: Matrix,
    valid_mask: Sequence[bool] | None,
) -> tuple[Matrix, Matrix]:
    if len(first) != len(second):
        raise ValueError("Paired matrices must have the same batch size.")

    mask = ensure_mask(valid_mask, batch_size=len(first))
    filtered_first = [row for row, keep in zip(first, mask) if keep]
    filtered_second = [row for row, keep in zip(second, mask) if keep]
    return filtered_first, filtered_second


def dot(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length for dot product.")
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def l2_norm(vector: Vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: Vector, right: Vector) -> float:
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot(left, right) / (left_norm * right_norm)


def mse(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length for MSE.")
    if not left:
        return 0.0
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)) / len(left)


def logsumexp(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        raise ValueError("logsumexp requires at least one value.")
    max_value = max(values_list)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values_list))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def binary_cross_entropy_with_logits(logit: float, target: float) -> float:
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def huber(error: float, *, delta: float) -> float:
    absolute_error = abs(error)
    if absolute_error <= delta:
        return 0.5 * error * error
    return delta * (absolute_error - 0.5 * delta)
