from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from src.datasets.pretrain_dataset import PretrainSample
from src.datasets.structure_io import CAFrame, _build_feature_vector


FrameEmbeddingMode = Literal["token_aggregate", "per_frame"]
FusionEmbeddingSource = Literal["fused_pooled", "fused_cls"]


@dataclass(frozen=True)
class MDTokens:
    token_features: list[list[float]]
    token_mask: list[bool]
    token_frame_indices: list[int]


@dataclass(frozen=True)
class TrajectoryEmbeddings:
    protein_id: str
    frame_indices: np.ndarray
    st_embeddings: np.ndarray
    fusion_embeddings: np.ndarray
    sequence_embedding: np.ndarray


def build_md_tokens_from_ca_frames(frames: Sequence[CAFrame]) -> MDTokens:
    if not frames:
        raise ValueError("At least one CA frame is required to build MD tokens.")
    total_frames = len(frames)
    token_features: list[list[float]] = []
    token_frame_indices: list[int] = []

    for frame_index, frame in enumerate(frames):
        total_residues = len(frame.coordinates)
        if total_residues <= 0:
            continue
        for residue_index, (x_coord, y_coord, z_coord) in enumerate(frame.coordinates):
            token_features.append(
                _build_feature_vector(
                    x=x_coord,
                    y=y_coord,
                    z=z_coord,
                    residue_index=residue_index,
                    total_residues=total_residues,
                    frame_index=frame_index,
                    total_frames=total_frames,
                    source_flag=1.0,
                )
            )
            token_frame_indices.append(frame_index)

    if not token_features:
        raise ValueError("No MD tokens could be built from CA frames.")
    return MDTokens(
        token_features=token_features,
        token_mask=[True] * len(token_features),
        token_frame_indices=token_frame_indices,
    )


def extract_trajectory_embeddings(
    *,
    sample: PretrainSample,
    frames: Sequence[CAFrame],
    sequence_encoder: Any,
    structure_encoder: Any,
    fusion_transformer: Any,
    frame_embedding_mode: FrameEmbeddingMode = "token_aggregate",
    fusion_embedding_source: FusionEmbeddingSource = "fused_pooled",
) -> TrajectoryEmbeddings:
    st_embeddings_tensor = extract_st_frame_embeddings_tensor(
        frames=frames,
        structure_encoder=structure_encoder,
        frame_embedding_mode=frame_embedding_mode,
    )

    torch = _torch_from_tensor(st_embeddings_tensor)
    seq_output = sequence_encoder([sample.sequence])
    sequence_embedding_tensor = _ensure_2d_tensor(seq_output.pooled_embedding, torch)[0]
    frame_count = int(st_embeddings_tensor.shape[0])
    seq_batch = sequence_embedding_tensor.unsqueeze(0).expand(frame_count, -1)

    fusion_output = fusion_transformer(
        seq_batch,
        st_embeddings_tensor,
        has_dyn=[True] * frame_count,
    )
    fusion_tensor = getattr(fusion_output, fusion_embedding_source)
    fusion_tensor = _ensure_2d_tensor(fusion_tensor, torch)

    return TrajectoryEmbeddings(
        protein_id=sample.protein_id,
        frame_indices=np.asarray([frame.frame_index for frame in frames], dtype=np.int64),
        st_embeddings=_to_numpy(st_embeddings_tensor),
        fusion_embeddings=_to_numpy(fusion_tensor),
        sequence_embedding=_to_numpy(sequence_embedding_tensor),
    )


def extract_st_frame_embeddings(
    *,
    frames: Sequence[CAFrame],
    structure_encoder: Any,
    frame_embedding_mode: FrameEmbeddingMode = "token_aggregate",
) -> np.ndarray:
    return _to_numpy(
        extract_st_frame_embeddings_tensor(
            frames=frames,
            structure_encoder=structure_encoder,
            frame_embedding_mode=frame_embedding_mode,
        )
    )


def extract_st_frame_embeddings_tensor(
    *,
    frames: Sequence[CAFrame],
    structure_encoder: Any,
    frame_embedding_mode: FrameEmbeddingMode = "token_aggregate",
) -> Any:
    if frame_embedding_mode == "token_aggregate":
        return _extract_token_aggregate_st_embeddings(
            frames=frames,
            structure_encoder=structure_encoder,
        )
    if frame_embedding_mode == "per_frame":
        return _extract_per_frame_st_embeddings(
            frames=frames,
            structure_encoder=structure_encoder,
        )
    raise ValueError(f"Unsupported frame_embedding_mode: {frame_embedding_mode}")


def _extract_token_aggregate_st_embeddings(*, frames: Sequence[CAFrame], structure_encoder: Any) -> Any:
    md_tokens = build_md_tokens_from_ca_frames(frames)
    output = structure_encoder([md_tokens.token_features], [md_tokens.token_mask])
    token_embeddings = output.token_embeddings[0]
    torch = _torch_from_tensor(token_embeddings)
    frame_index_tensor = torch.as_tensor(
        md_tokens.token_frame_indices,
        dtype=torch.long,
        device=token_embeddings.device,
    )
    frame_embeddings = []
    for frame_index in range(len(frames)):
        mask = frame_index_tensor == frame_index
        if not bool(mask.any().item()):
            raise ValueError(f"Frame {frame_index} has no ST tokens.")
        frame_embeddings.append(token_embeddings[mask].mean(dim=0))
    return torch.stack(frame_embeddings, dim=0)


def _extract_per_frame_st_embeddings(*, frames: Sequence[CAFrame], structure_encoder: Any) -> Any:
    frame_rows = []
    for frame in frames:
        md_tokens = build_md_tokens_from_ca_frames([
            CAFrame(
                frame_index=0,
                residue_keys=frame.residue_keys,
                coordinates=frame.coordinates,
            )
        ])
        output = structure_encoder([md_tokens.token_features], [md_tokens.token_mask])
        frame_rows.append(_ensure_2d_tensor(output.pooled_embedding, _torch_from_tensor(output.pooled_embedding))[0])
    torch = _torch_from_tensor(frame_rows[0])
    return torch.stack(frame_rows, dim=0)


def _ensure_2d_tensor(value: Any, torch: Any) -> Any:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D tensor-like embedding, got shape {tuple(tensor.shape)}.")
    return tensor


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _torch_from_tensor(value: Any) -> Any:
    if hasattr(value, "new_empty"):
        return __import__("torch")
    return __import__("torch")
