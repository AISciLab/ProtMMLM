from __future__ import annotations

from collections import defaultdict
import hashlib
import random
from typing import Any

from src.datasets.downstream_dataset import DownstreamDataset
from src.datasets.mmseqs_cluster_splits import split_dataset_by_mmseqs_clusters
from src.datasets.prmftp_cluster_splits import split_prmftp_dataset_by_mmseqs_clusters


def split_downstream_dataset(
    dataset: DownstreamDataset,
    *,
    num_folds: int,
    test_fold_index: int,
    val_fold_index: int | None,
    val_fold_offset: int,
    validation_fraction: float = 0.1,
    validation_seed: int | None = None,
    manifest_validation_policy: str = "random_grouped_9_1",
    split_strategy: str | None = None,
    mmseqs_binary: str = "mmseqs",
    cluster_min_seq_id: float = 0.4,
    cluster_coverage: float = 0.8,
    cluster_cov_mode: int = 0,
    cluster_train_fraction: float = 0.8,
    cluster_validation_fraction: float = 0.1,
    cluster_seed: int | None = None,
    cluster_sequence_field: str | None = None,
    cluster_balance_labels: bool = False,
    cluster_shuffle_order: bool = False,
    balance_binary_train_split: bool = True,
) -> dict[str, Any]:
    resolved_split_strategy = _normalize_split_strategy(split_strategy)
    if resolved_split_strategy == "prmftp_mmseqs_clustered":
        split_bundle = split_prmftp_dataset_by_mmseqs_clusters(
            dataset,
            mmseqs_binary=mmseqs_binary,
            cluster_min_seq_id=cluster_min_seq_id,
            cluster_coverage=cluster_coverage,
            cluster_cov_mode=cluster_cov_mode,
            cluster_train_fraction=cluster_train_fraction,
            cluster_validation_fraction=cluster_validation_fraction,
            cluster_seed=cluster_seed,
        )
    elif resolved_split_strategy == "sequence_mmseqs_clustered":
        split_bundle = split_dataset_by_mmseqs_clusters(
            dataset,
            cluster_sequence_field=cluster_sequence_field or "sequence",
            mmseqs_binary=mmseqs_binary,
            cluster_min_seq_id=cluster_min_seq_id,
            cluster_coverage=cluster_coverage,
            cluster_cov_mode=cluster_cov_mode,
            cluster_train_fraction=cluster_train_fraction,
            cluster_validation_fraction=cluster_validation_fraction,
            cluster_seed=cluster_seed,
            split_source_name="sequence_mmseqs_clustered",
            cluster_balance_labels=cluster_balance_labels,
            cluster_shuffle_order=cluster_shuffle_order,
        )
    elif has_manifest_split(dataset):
        split_bundle = split_downstream_dataset_from_manifest_splits(
            dataset,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
            manifest_validation_policy=manifest_validation_policy,
        )
    else:
        split_bundle = split_downstream_dataset_by_folds(
            dataset,
            num_folds=num_folds,
            test_fold_index=test_fold_index,
            val_fold_index=val_fold_index,
            val_fold_offset=val_fold_offset,
        )
    if not balance_binary_train_split:
        summary = dict(split_bundle["summary"])
        summary["train_balance_applied"] = False
        summary["train_balance_disabled"] = True
        return {**split_bundle, "summary": summary}
    return _maybe_balance_binary_train_split(split_bundle)


def split_downstream_dataset_by_folds(
    dataset: DownstreamDataset,
    *,
    num_folds: int,
    test_fold_index: int,
    val_fold_index: int | None,
    val_fold_offset: int,
) -> dict[str, Any]:
    if num_folds < 3:
        raise ValueError(f"num_folds must be at least 3, got {num_folds}.")

    normalized_test_fold = test_fold_index % num_folds
    normalized_val_fold = (
        (normalized_test_fold + val_fold_offset) % num_folds
        if val_fold_index is None
        else val_fold_index % num_folds
    )
    if normalized_val_fold == normalized_test_fold:
        raise ValueError("Validation fold must differ from the test fold.")

    fold_assignments = assign_folds(dataset, num_folds=num_folds)
    train_samples: list[Any] = []
    validation_samples: list[Any] = []
    test_samples: list[Any] = []
    fold_counts = {fold_index: 0 for fold_index in range(num_folds)}

    for sample in dataset:
        fold_index = fold_assignments[sample.sample_id]
        fold_counts[fold_index] += 1
        if fold_index == normalized_test_fold:
            test_samples.append(sample)
        elif fold_index == normalized_val_fold:
            validation_samples.append(sample)
        else:
            train_samples.append(sample)

    if not train_samples or not validation_samples or not test_samples:
        raise ValueError(
            "Cross-validation split produced an empty subset. "
            f"train={len(train_samples)}, val={len(validation_samples)}, test={len(test_samples)}."
        )

    return {
        "train": DownstreamDataset(train_samples, task_name=dataset.task_name, task_kind=dataset.task_kind),
        "validation": DownstreamDataset(
            validation_samples,
            task_name=dataset.task_name,
            task_kind=dataset.task_kind,
        ),
        "test": DownstreamDataset(test_samples, task_name=dataset.task_name, task_kind=dataset.task_kind),
        "summary": {
            "num_folds": num_folds,
            "test_fold_index": normalized_test_fold,
            "val_fold_index": normalized_val_fold,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "test_samples": len(test_samples),
            **{f"fold_{fold_index}_samples": fold_counts[fold_index] for fold_index in sorted(fold_counts)},
        },
    }


OFFICIAL_TRAIN_VALIDATION_MODULUS = 10
OFFICIAL_TRAIN_VALIDATION_REMAINDER = 0


def split_downstream_dataset_from_manifest_splits(
    dataset: DownstreamDataset,
    *,
    validation_fraction: float = 0.1,
    validation_seed: int | None = None,
    manifest_validation_policy: str = "random_grouped_9_1",
) -> dict[str, Any]:
    train_pool: list[Any] = []
    validation_samples: list[Any] = []
    test_samples: list[Any] = []
    missing_split_samples: list[str] = []
    split_counts = {"train": 0, "validation": 0, "test": 0, "missing": 0}

    for sample in dataset:
        split_name = getattr(sample, "split", None)
        if split_name == "train":
            train_pool.append(sample)
            split_counts["train"] += 1
        elif split_name == "validation":
            validation_samples.append(sample)
            split_counts["validation"] += 1
        elif split_name == "test":
            test_samples.append(sample)
            split_counts["test"] += 1
        else:
            missing_split_samples.append(sample.sample_id)
            split_counts["missing"] += 1

    if missing_split_samples:
        preview = ", ".join(missing_split_samples[:5])
        raise ValueError(
            "Manifest contains a split column, but some samples have no split value. "
            f"First missing sample_id(s): {preview}"
        )
    if not test_samples:
        raise ValueError(
            "Manifest split mode requires at least one official test sample. "
            "If you used --sample-limit, it may have truncated away the test split."
        )
    if not train_pool:
        raise ValueError(
            "Manifest split mode requires at least one official train sample. "
            "If you used --sample-limit, it may have truncated away the train split."
        )

    validation_source = "manifest"
    validation_metadata: dict[str, Any] = {}
    if not validation_samples:
        resolved_policy = _normalize_manifest_validation_policy(manifest_validation_policy)
        if resolved_policy == "deterministic_grouped_9_1":
            validation_source = "train_split_hash_9_1"
            train_pool, validation_samples, validation_metadata = _deterministic_split_official_train_samples(
                train_pool,
                task_name=dataset.task_name,
                task_kind=dataset.task_kind,
                validation_fraction=validation_fraction,
            )
        else:
            validation_source = "official_train_random_grouped"
            train_pool, validation_samples, validation_metadata = _random_split_official_train_samples(
                train_pool,
                task_name=dataset.task_name,
                task_kind=dataset.task_kind,
                validation_fraction=validation_fraction,
                validation_seed=validation_seed,
            )

    if not train_pool or not validation_samples:
        raise ValueError(
            "Manifest split mode produced an empty train or validation subset. "
            f"train={len(train_pool)}, val={len(validation_samples)}, test={len(test_samples)}."
        )

    summary: dict[str, Any] = {
        "split_source": "manifest",
        "test_split_source": "official_manifest",
        "official_train_samples": split_counts["train"],
        "official_validation_samples": split_counts["validation"],
        "official_test_samples": split_counts["test"],
        "validation_source": validation_source,
        "train_samples": len(train_pool),
        "validation_samples": len(validation_samples),
        "test_samples": len(test_samples),
        "official_test_used_for_training": False,
    }
    if validation_metadata:
        summary.update(validation_metadata)

    return {
        "train": DownstreamDataset(train_pool, task_name=dataset.task_name, task_kind=dataset.task_kind),
        "validation": DownstreamDataset(
            validation_samples,
            task_name=dataset.task_name,
            task_kind=dataset.task_kind,
        ),
        "test": DownstreamDataset(test_samples, task_name=dataset.task_name, task_kind=dataset.task_kind),
        "summary": summary,
    }


def has_manifest_split(dataset: DownstreamDataset) -> bool:
    return any(getattr(sample, "split", None) for sample in dataset)


def _normalize_split_strategy(raw_value: str | None) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"", "default", "auto"}:
        return "default"
    if normalized in {"prmftp_mmseqs_clustered", "prmftp_clustered", "mmseqs_clustered"}:
        return "prmftp_mmseqs_clustered"
    if normalized in {
        "sequence_mmseqs_clustered",
        "sequence_clustered",
        "protein_mmseqs_clustered",
        "generic_mmseqs_clustered",
        "conotoxin_mmseqs_clustered",
    }:
        return "sequence_mmseqs_clustered"
    raise ValueError(
        f"Unsupported split_strategy {raw_value!r}. "
        "Expected default, prmftp_mmseqs_clustered, or sequence_mmseqs_clustered."
    )


def _maybe_balance_binary_train_split(split_bundle: dict[str, Any]) -> dict[str, Any]:
    train_dataset = split_bundle["train"]
    summary = dict(split_bundle["summary"])
    if train_dataset.task_kind != "binary":
        summary["train_balance_applied"] = False
        return {
            **split_bundle,
            "summary": summary,
        }

    positive_samples = [sample for sample in train_dataset if _binary_label_value(sample) == 1]
    negative_samples = [sample for sample in train_dataset if _binary_label_value(sample) == 0]
    summary["train_original_positive_samples"] = len(positive_samples)
    summary["train_original_negative_samples"] = len(negative_samples)

    if not positive_samples or not negative_samples:
        summary["train_balance_applied"] = False
        summary["train_balanced_positive_samples"] = len(positive_samples)
        summary["train_balanced_negative_samples"] = len(negative_samples)
        return {
            **split_bundle,
            "summary": summary,
        }

    if len(positive_samples) == len(negative_samples):
        summary["train_balance_applied"] = False
        summary["train_balanced_positive_samples"] = len(positive_samples)
        summary["train_balanced_negative_samples"] = len(negative_samples)
        return {
            **split_bundle,
            "summary": summary,
        }

    minority_samples = positive_samples if len(positive_samples) < len(negative_samples) else negative_samples
    majority_samples = negative_samples if minority_samples is positive_samples else positive_samples
    balanced_minority_samples = _oversample_to_count(minority_samples, target_count=len(majority_samples))
    balanced_samples = list(majority_samples) + balanced_minority_samples
    balanced_samples.sort(key=lambda sample: stable_sample_key(sample.sample_id, sample.sequence_hash, sample.raw_label))

    num_positive = sum(1 for sample in balanced_samples if _binary_label_value(sample) == 1)
    num_negative = len(balanced_samples) - num_positive
    summary["train_balance_applied"] = True
    summary["train_balanced_positive_samples"] = num_positive
    summary["train_balanced_negative_samples"] = num_negative
    summary["train_samples"] = len(balanced_samples)

    return {
        **split_bundle,
        "train": DownstreamDataset(
            balanced_samples,
            task_name=train_dataset.task_name,
            task_kind=train_dataset.task_kind,
        ),
        "summary": summary,
    }


def _binary_label_value(sample: Any) -> int:
    return int(float(sample.target))


def _oversample_to_count(samples: list[Any], *, target_count: int) -> list[Any]:
    if not samples:
        return []
    if len(samples) >= target_count:
        return list(samples[:target_count])
    repeats, remainder = divmod(target_count, len(samples))
    balanced_samples = list(samples) * repeats
    if remainder:
        balanced_samples.extend(samples[:remainder])
    return balanced_samples


def assign_folds(
    dataset: DownstreamDataset,
    *,
    num_folds: int,
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    if dataset.task_kind == "regression":
        grouped_samples = _group_samples_by_fold_key(dataset.samples)
        ordered_groups = sorted(
            grouped_samples.items(),
            key=lambda item: (float(item[1][0].target), stable_sample_key(item[0])),
        )
        return _assign_grouped_samples(ordered_groups, num_folds=num_folds)

    grouped_samples: dict[str, list[Any]] = defaultdict(list)
    for sample in dataset:
        grouped_samples[label_bucket_key(sample.raw_label)].append(sample)

    for bucket_key, bucket_samples in sorted(grouped_samples.items()):
        grouped_bucket_samples = _group_samples_by_fold_key(bucket_samples)
        ordered_groups = sorted(
            grouped_bucket_samples.items(),
            key=lambda item: stable_sample_key(item[0], bucket_key),
        )
        assignments.update(_assign_grouped_samples(ordered_groups, num_folds=num_folds))
    return assignments


def label_bucket_key(raw_label: str) -> str:
    return str(raw_label).strip()


def _group_samples_by_fold_key(samples: list[Any]) -> dict[str, list[Any]]:
    grouped_samples: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped_samples[_fold_group_key(sample)].append(sample)
    return grouped_samples


def _assign_grouped_samples(
    ordered_groups: list[tuple[str, list[Any]]],
    *,
    num_folds: int,
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for index, (_, group_samples) in enumerate(ordered_groups):
        fold_index = index % num_folds
        for sample in group_samples:
            assignments[sample.sample_id] = fold_index
    return assignments


def _fold_group_key(sample: Any) -> str:
    peptide_hash = str(getattr(sample, "peptide_sequence_hash", "") or "").strip()
    if peptide_hash:
        return f"{sample.sequence_hash}::{peptide_hash}"
    return str(sample.sequence_hash)


def _random_split_official_train_samples(
    samples: list[Any],
    *,
    task_name: str,
    task_kind: str,
    validation_fraction: float,
    validation_seed: int | None,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be between 0 and 1, got {validation_fraction}.")

    seed_value = validation_seed if validation_seed is not None else random.SystemRandom().randrange(2**63)
    rng = random.Random(seed_value)
    train_samples: list[Any] = []
    validation_samples: list[Any] = []
    label_group_counts: dict[str, int] = {}

    label_buckets: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        label_buckets[label_bucket_key(sample.raw_label)].append(sample)

    for label_key, bucket_samples in sorted(label_buckets.items()):
        grouped_samples = _group_samples_by_fold_key(bucket_samples)
        grouped_items = sorted(grouped_samples.items(), key=lambda item: stable_sample_key(item[0], label_key))
        rng.shuffle(grouped_items)
        target_validation_samples = max(1, round(len(bucket_samples) * validation_fraction))
        current_validation_samples = 0
        for _, group in grouped_items:
            if current_validation_samples < target_validation_samples:
                validation_samples.extend(group)
                current_validation_samples += len(group)
            else:
                train_samples.extend(group)
        label_group_counts[f"validation_label_bucket_{label_key}_samples"] = current_validation_samples

    train_samples.sort(key=lambda sample: stable_sample_key(sample.sample_id, sample.sequence_hash, sample.raw_label))
    validation_samples.sort(key=lambda sample: stable_sample_key(sample.sample_id, sample.sequence_hash, sample.raw_label))
    return train_samples, validation_samples, {
        "validation_split_method": "random_grouped_9_1",
        "validation_fraction": validation_fraction,
        "validation_seed": seed_value,
        "validation_grouping_key": "fold_group_key",
        "validation_grouping_preserves_pairs": True,
        **label_group_counts,
    }


def _deterministic_split_official_train_samples(
    samples: list[Any],
    *,
    task_name: str,
    task_kind: str,
    validation_fraction: float,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if abs(float(validation_fraction) - 0.1) > 1e-9:
        raise ValueError(
            "deterministic_grouped_9_1 requires validation_fraction=0.1, "
            f"got {validation_fraction}."
        )

    train_samples: list[Any] = []
    validation_samples: list[Any] = []
    label_group_counts: dict[str, int] = {}
    label_buckets: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        label_buckets[label_bucket_key(sample.raw_label)].append(sample)

    for label_key, bucket_samples in sorted(label_buckets.items()):
        grouped_samples = _group_manifest_train_samples_for_validation(bucket_samples)
        current_validation_samples = 0
        for group_key, group in sorted(grouped_samples.items(), key=lambda item: stable_sample_key(item[0])):
            bucket_index = _validation_bucket_for_group_key(group_key)
            if bucket_index == OFFICIAL_TRAIN_VALIDATION_REMAINDER:
                validation_samples.extend(group)
                current_validation_samples += len(group)
            else:
                train_samples.extend(group)
        label_group_counts[f"validation_label_bucket_{label_key}_samples"] = current_validation_samples

    train_samples.sort(key=lambda sample: stable_sample_key(sample.sample_id, sample.sequence_hash, sample.raw_label))
    validation_samples.sort(key=lambda sample: stable_sample_key(sample.sample_id, sample.sequence_hash, sample.raw_label))
    return train_samples, validation_samples, {
        "validation_split_method": "deterministic_hash_grouped_9_1",
        "validation_fraction": validation_fraction,
        "validation_grouping_key": "label_bucket+fold_group_key",
        "validation_grouping_preserves_pairs": True,
        "validation_bucket_modulus": OFFICIAL_TRAIN_VALIDATION_MODULUS,
        "validation_bucket_remainder": OFFICIAL_TRAIN_VALIDATION_REMAINDER,
        **label_group_counts,
    }


def _normalize_manifest_validation_policy(raw_value: str) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"", "deterministic", "deterministic_grouped_9_1", "grouped_hash_9_1"}:
        return "deterministic_grouped_9_1"
    if normalized in {"random", "random_grouped_9_1"}:
        return "random_grouped_9_1"
    raise ValueError(
        f"Unsupported manifest_validation_policy {raw_value!r}. "
        "Expected deterministic_grouped_9_1 or random_grouped_9_1."
    )


def _group_manifest_train_samples_for_validation(samples: list[Any]) -> dict[str, list[Any]]:
    grouped_samples: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped_samples[_manifest_validation_group_key(sample)].append(sample)
    return grouped_samples


def _manifest_validation_group_key(sample: Any) -> str:
    return f"{label_bucket_key(sample.raw_label)}::{_fold_group_key(sample)}"


def _validation_bucket_for_group_key(group_key: str) -> int:
    digest = stable_sample_key(group_key)
    return int(digest, 16) % OFFICIAL_TRAIN_VALIDATION_MODULUS


def stable_sample_key(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
