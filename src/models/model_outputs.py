from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TensorLike = Any


@dataclass(frozen=True)
class SequenceBackboneOutput:
    token_embeddings: TensorLike
    cls_embedding: TensorLike
    mean_pooled_embedding: TensorLike


@dataclass(frozen=True)
class SequenceEncoderOutput:
    token_embeddings: TensorLike
    cls_embedding: TensorLike
    pooled_embedding: TensorLike
