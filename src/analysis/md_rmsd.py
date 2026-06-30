from __future__ import annotations

from typing import Sequence

import numpy as np

from src.datasets.structure_io import CAFrame


def kabsch_rmsd(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Coordinate shapes must match, got {a.shape} and {b.shape}.")
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape [N, 3], got {a.shape}.")
    if a.shape[0] == 0:
        return float("nan")

    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    covariance = a_centered.T @ b_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T

    diff = a_centered @ rotation - b_centered
    return float(np.sqrt(np.maximum(np.sum(diff * diff) / max(a.shape[0], 1), eps * 0.0)))


def align_common_residues(
    frame_a: CAFrame,
    frame_b: CAFrame,
    *,
    min_common_residues: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if min_common_residues <= 0:
        raise ValueError(f"min_common_residues must be positive, got {min_common_residues}.")

    b_index_by_key = {residue_key: index for index, residue_key in enumerate(frame_b.residue_keys)}
    a_indices: list[int] = []
    b_indices: list[int] = []
    for a_index, residue_key in enumerate(frame_a.residue_keys):
        b_index = b_index_by_key.get(residue_key)
        if b_index is None:
            continue
        a_indices.append(a_index)
        b_indices.append(b_index)

    if len(a_indices) < min_common_residues:
        return None

    coords_a = np.asarray(frame_a.coordinates, dtype=np.float64)[a_indices]
    coords_b = np.asarray(frame_b.coordinates, dtype=np.float64)[b_indices]
    return coords_a, coords_b


def pairwise_rmsd_matrix(
    frames: Sequence[CAFrame],
    *,
    min_common_residues: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(frames)
    rmsd_matrix = np.zeros((frame_count, frame_count), dtype=np.float64)
    valid_pair_mask = np.ones((frame_count, frame_count), dtype=bool)

    for row_index in range(frame_count):
        for col_index in range(row_index + 1, frame_count):
            aligned = align_common_residues(
                frames[row_index],
                frames[col_index],
                min_common_residues=min_common_residues,
            )
            if aligned is None:
                rmsd_value = np.nan
                valid_pair_mask[row_index, col_index] = False
                valid_pair_mask[col_index, row_index] = False
            else:
                rmsd_value = kabsch_rmsd(aligned[0], aligned[1])
            rmsd_matrix[row_index, col_index] = rmsd_value
            rmsd_matrix[col_index, row_index] = rmsd_value

    return rmsd_matrix, valid_pair_mask
