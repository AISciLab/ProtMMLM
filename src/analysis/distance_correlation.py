from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Literal

import numpy as np


DistanceMetric = Literal["cosine", "euclidean"]
CorrelationMethod = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class MatrixComparison:
    pearson: float
    spearman: float
    num_pairs: int
    mantel_r: float | None = None
    mantel_p: float | None = None


def pairwise_embedding_distance(
    embeddings: np.ndarray,
    *,
    metric: DistanceMetric = "cosine",
    eps: float = 1e-12,
) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be a 2D array, got shape {matrix.shape}.")
    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / np.maximum(norms, eps)
        distances = 1.0 - normalized @ normalized.T
        np.fill_diagonal(distances, 0.0)
        return distances
    if metric == "euclidean":
        diff = matrix[:, None, :] - matrix[None, :, :]
        return np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    raise ValueError(f"Unsupported embedding distance metric: {metric}")


def compare_distance_matrices(
    reference_matrix: np.ndarray,
    embedding_matrix: np.ndarray,
    *,
    valid_pair_mask: np.ndarray | None = None,
    mantel_permutations: int = 0,
    mantel_method: CorrelationMethod = "spearman",
    random_seed: int = 0,
) -> MatrixComparison:
    reference_values, embedding_values = condensed_valid_pairs(
        reference_matrix,
        embedding_matrix,
        valid_pair_mask=valid_pair_mask,
    )
    pearson = pearson_correlation(reference_values, embedding_values)
    spearman = spearman_correlation(reference_values, embedding_values)

    mantel_r = None
    mantel_p = None
    if mantel_permutations > 0:
        mantel_r, mantel_p = mantel_test(
            reference_matrix,
            embedding_matrix,
            valid_pair_mask=valid_pair_mask,
            permutations=mantel_permutations,
            method=mantel_method,
            random_seed=random_seed,
        )

    return MatrixComparison(
        pearson=pearson,
        spearman=spearman,
        num_pairs=int(reference_values.size),
        mantel_r=mantel_r,
        mantel_p=mantel_p,
    )


def condensed_valid_pairs(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    valid_pair_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(matrix_a, dtype=np.float64)
    right = np.asarray(matrix_b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Matrix shapes must match, got {left.shape} and {right.shape}.")
    if left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError(f"Matrices must be square, got shape {left.shape}.")

    upper_mask = np.triu(np.ones(left.shape, dtype=bool), k=1)
    if valid_pair_mask is not None:
        mask = np.asarray(valid_pair_mask, dtype=bool)
        if mask.shape != left.shape:
            raise ValueError(f"valid_pair_mask shape {mask.shape} does not match matrix shape {left.shape}.")
        upper_mask &= mask
    upper_mask &= np.isfinite(left) & np.isfinite(right)
    return left[upper_mask], right[upper_mask]


def pearson_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    left = np.asarray(values_a, dtype=np.float64)
    right = np.asarray(values_b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Vector shapes must match, got {left.shape} and {right.shape}.")
    if left.size < 2:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def spearman_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if importlib.util.find_spec("scipy") is not None:
        from scipy.stats import spearmanr  # type: ignore

        result = spearmanr(values_a, values_b)
        return float(result.correlation)
    return pearson_correlation(_rankdata(values_a), _rankdata(values_b))


def mantel_test(
    reference_matrix: np.ndarray,
    embedding_matrix: np.ndarray,
    *,
    valid_pair_mask: np.ndarray | None,
    permutations: int,
    method: CorrelationMethod,
    random_seed: int,
) -> tuple[float, float]:
    if permutations <= 0:
        raise ValueError(f"permutations must be positive, got {permutations}.")
    reference = np.asarray(reference_matrix, dtype=np.float64)
    embedding = np.asarray(embedding_matrix, dtype=np.float64)
    rng = np.random.default_rng(random_seed)

    observed = _matrix_correlation(reference, embedding, valid_pair_mask, method)
    if not np.isfinite(observed):
        return observed, float("nan")

    exceed_count = 0
    for _ in range(permutations):
        permutation = rng.permutation(reference.shape[0])
        permuted = embedding[permutation][:, permutation]
        permuted_r = _matrix_correlation(reference, permuted, valid_pair_mask, method)
        if np.isfinite(permuted_r) and abs(permuted_r) >= abs(observed):
            exceed_count += 1
    p_value = (1.0 + exceed_count) / (1.0 + permutations)
    return float(observed), float(p_value)


def _matrix_correlation(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    valid_pair_mask: np.ndarray | None,
    method: CorrelationMethod,
) -> float:
    values_a, values_b = condensed_valid_pairs(
        matrix_a,
        matrix_b,
        valid_pair_mask=valid_pair_mask,
    )
    if method == "pearson":
        return pearson_correlation(values_a, values_b)
    if method == "spearman":
        return spearman_correlation(values_a, values_b)
    raise ValueError(f"Unsupported correlation method: {method}")


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(array, kind="mergesort")
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(array.size)
    sorted_values = array[sorter]
    ranks = np.zeros(array.size, dtype=np.float64)

    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = average_rank
        start = end
    return ranks[inverse]
