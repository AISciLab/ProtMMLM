from __future__ import annotations

from typing import Any

from src.datasets.downstream_dataset import DownstreamDataset
from src.datasets.mmseqs_cluster_splits import (
    DEFAULT_CLUSTER_COVERAGE,
    DEFAULT_CLUSTER_COV_MODE,
    DEFAULT_CLUSTER_MIN_SEQ_ID,
    DEFAULT_CLUSTER_SEED,
    DEFAULT_CLUSTER_TRAIN_FRACTION,
    DEFAULT_CLUSTER_VALIDATION_FRACTION,
    DEFAULT_MMSEQS_BINARY,
    split_dataset_by_mmseqs_clusters,
)


def split_prmftp_dataset_by_mmseqs_clusters(
    dataset: DownstreamDataset,
    *,
    mmseqs_binary: str = DEFAULT_MMSEQS_BINARY,
    cluster_min_seq_id: float = DEFAULT_CLUSTER_MIN_SEQ_ID,
    cluster_coverage: float = DEFAULT_CLUSTER_COVERAGE,
    cluster_cov_mode: int = DEFAULT_CLUSTER_COV_MODE,
    cluster_train_fraction: float = DEFAULT_CLUSTER_TRAIN_FRACTION,
    cluster_validation_fraction: float = DEFAULT_CLUSTER_VALIDATION_FRACTION,
    cluster_seed: int | None = DEFAULT_CLUSTER_SEED,
) -> dict[str, Any]:
    if dataset.task_name != "prmftp":
        raise ValueError(
            "MMseqs clustered split is only supported for PrMFTP, "
            f"got task_name={dataset.task_name!r}."
        )
    if dataset.task_kind != "multilabel":
        raise ValueError(
            "MMseqs clustered split expects a multilabel PrMFTP dataset, "
            f"got task_kind={dataset.task_kind!r}."
        )
    return split_dataset_by_mmseqs_clusters(
        dataset,
        cluster_sequence_field="sequence",
        mmseqs_binary=mmseqs_binary,
        cluster_min_seq_id=cluster_min_seq_id,
        cluster_coverage=cluster_coverage,
        cluster_cov_mode=cluster_cov_mode,
        cluster_train_fraction=cluster_train_fraction,
        cluster_validation_fraction=cluster_validation_fraction,
        cluster_seed=cluster_seed,
        split_source_name="prmftp_mmseqs_clustered",
    )
