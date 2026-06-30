from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any

from src.datasets.downstream_dataset import DownstreamDataset


DEFAULT_MMSEQS_BINARY = "mmseqs"
DEFAULT_CLUSTER_MIN_SEQ_ID = 0.4
DEFAULT_CLUSTER_COVERAGE = 0.8
DEFAULT_CLUSTER_COV_MODE = 0
DEFAULT_CLUSTER_TRAIN_FRACTION = 0.8
DEFAULT_CLUSTER_VALIDATION_FRACTION = 0.1
DEFAULT_CLUSTER_SEED: int | None = None


SUPPORTED_CLUSTER_SEQUENCE_FIELDS = {"sequence", "peptide_sequence"}


def split_dataset_by_mmseqs_clusters(
    dataset: DownstreamDataset,
    *,
    cluster_sequence_field: str,
    mmseqs_binary: str = DEFAULT_MMSEQS_BINARY,
    cluster_min_seq_id: float = DEFAULT_CLUSTER_MIN_SEQ_ID,
    cluster_coverage: float = DEFAULT_CLUSTER_COVERAGE,
    cluster_cov_mode: int = DEFAULT_CLUSTER_COV_MODE,
    cluster_train_fraction: float = DEFAULT_CLUSTER_TRAIN_FRACTION,
    cluster_validation_fraction: float = DEFAULT_CLUSTER_VALIDATION_FRACTION,
    cluster_seed: int | None = DEFAULT_CLUSTER_SEED,
    split_source_name: str = "mmseqs_clustered",
    cluster_balance_labels: bool = False,
    cluster_shuffle_order: bool = False,
) -> dict[str, Any]:
    resolved_cluster_field = _normalize_cluster_sequence_field(cluster_sequence_field)
    if not 0.0 < cluster_train_fraction < 1.0:
        raise ValueError(f"cluster_train_fraction must be between 0 and 1, got {cluster_train_fraction}.")
    if not 0.0 < cluster_validation_fraction < 1.0:
        raise ValueError(
            f"cluster_validation_fraction must be between 0 and 1, got {cluster_validation_fraction}."
        )
    cluster_test_fraction = 1.0 - float(cluster_train_fraction) - float(cluster_validation_fraction)
    if cluster_test_fraction <= 0.0:
        raise ValueError(
            "cluster_train_fraction + cluster_validation_fraction must be less than 1, "
            f"got train={cluster_train_fraction}, val={cluster_validation_fraction}."
        )
    if not dataset.samples:
        raise ValueError(f"Cannot split an empty {dataset.task_name} dataset.")

    samples = list(dataset.samples)
    sequence_records = _unique_cluster_sequence_records(samples, cluster_sequence_field=resolved_cluster_field)
    cluster_by_record_id = _run_mmseqs_cluster(
        sequence_records,
        mmseqs_binary=mmseqs_binary,
        cluster_min_seq_id=cluster_min_seq_id,
        cluster_coverage=cluster_coverage,
        cluster_cov_mode=cluster_cov_mode,
        split_source_name=split_source_name,
    )
    cluster_by_sample_id = _cluster_by_sample_id(
        samples,
        cluster_sequence_field=resolved_cluster_field,
        cluster_by_record_id=cluster_by_record_id,
    )
    split_by_cluster_id, split_sizes = _assign_cluster_splits(
        samples,
        cluster_by_sample_id=cluster_by_sample_id,
        cluster_train_fraction=cluster_train_fraction,
        cluster_validation_fraction=cluster_validation_fraction,
        cluster_seed=cluster_seed,
        cluster_balance_labels=cluster_balance_labels,
        cluster_shuffle_order=cluster_shuffle_order,
    )

    train_samples: list[Any] = []
    validation_samples: list[Any] = []
    test_samples: list[Any] = []
    cluster_ids_by_split: dict[str, set[str]] = defaultdict(set)
    sequence_splits: dict[str, set[str]] = defaultdict(set)

    for sample in samples:
        cluster_id = cluster_by_sample_id[sample.sample_id]
        split_name = split_by_cluster_id[cluster_id]
        cluster_ids_by_split[split_name].add(cluster_id)
        sequence_splits[_cluster_record_id(sample, cluster_sequence_field=resolved_cluster_field)].add(split_name)
        assigned_sample = replace(sample, split=split_name)
        if split_name == "train":
            train_samples.append(assigned_sample)
        elif split_name == "validation":
            validation_samples.append(assigned_sample)
        elif split_name == "test":
            test_samples.append(assigned_sample)
        else:
            raise ValueError(f"Unsupported split assignment {split_name!r} for sample {sample.sample_id!r}.")

    if not train_samples or not validation_samples or not test_samples:
        raise ValueError(
            "MMseqs clustered split produced an empty subset. "
            f"train={len(train_samples)}, val={len(validation_samples)}, test={len(test_samples)}."
        )

    max_cluster_leakage = max(
        (len(cluster_splits) for cluster_splits in _invert_split_map(split_by_cluster_id).values()),
        default=1,
    )
    max_sequence_leakage = max((len(split_names) for split_names in sequence_splits.values()), default=1)
    if max_cluster_leakage != 1:
        raise ValueError(f"Cluster leakage detected across {split_source_name} splits.")
    if max_sequence_leakage != 1:
        raise ValueError(
            f"{resolved_cluster_field} leakage detected across {split_source_name} splits."
        )

    summary = {
        "split_source": split_source_name,
        "cluster_sequence_field": resolved_cluster_field,
        "cluster_min_seq_id": cluster_min_seq_id,
        "cluster_coverage": cluster_coverage,
        "cluster_cov_mode": cluster_cov_mode,
        "cluster_train_fraction": cluster_train_fraction,
        "cluster_validation_fraction": cluster_validation_fraction,
        "cluster_test_fraction": cluster_test_fraction,
        "cluster_seed": cluster_seed,
        "cluster_balance_labels": cluster_balance_labels,
        "cluster_balance_labels_applied": bool(cluster_balance_labels),
        "cluster_shuffle_order": cluster_shuffle_order,
        "mmseqs_binary": mmseqs_binary,
        "original_official_splits_reused": False,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "test_samples": len(test_samples),
        "num_clustered_sequences": len(sequence_records),
        "num_clusters": len(set(cluster_by_record_id.values())),
        "train_clusters": len(cluster_ids_by_split.get("train", set())),
        "validation_clusters": len(cluster_ids_by_split.get("validation", set())),
        "test_clusters": len(cluster_ids_by_split.get("test", set())),
        "cluster_leakage_check_passed": True,
        "cluster_sequence_leakage_check_passed": True,
        "sequence_leakage_check_passed": True,
        "max_cluster_split_membership": max_cluster_leakage,
        "max_sequence_split_membership": max_sequence_leakage,
        **_label_counts_by_split(("train", train_samples), ("validation", validation_samples), ("test", test_samples)),
        **split_sizes,
    }

    return {
        "train": DownstreamDataset(train_samples, task_name=dataset.task_name, task_kind=dataset.task_kind),
        "validation": DownstreamDataset(
            validation_samples,
            task_name=dataset.task_name,
            task_kind=dataset.task_kind,
        ),
        "test": DownstreamDataset(test_samples, task_name=dataset.task_name, task_kind=dataset.task_kind),
        "summary": summary,
    }


def _normalize_cluster_sequence_field(raw_field: str) -> str:
    normalized = str(raw_field or "").strip().lower()
    if normalized in {"protein_sequence", "tcr_sequence", "main_sequence"}:
        normalized = "sequence"
    if normalized not in SUPPORTED_CLUSTER_SEQUENCE_FIELDS:
        raise ValueError(
            f"Unsupported cluster_sequence_field {raw_field!r}. "
            f"Expected one of {sorted(SUPPORTED_CLUSTER_SEQUENCE_FIELDS)}."
        )
    return normalized


def _unique_cluster_sequence_records(
    samples: list[Any],
    *,
    cluster_sequence_field: str,
) -> dict[str, str]:
    records: dict[str, str] = {}
    for sample in samples:
        record_id = _cluster_record_id(sample, cluster_sequence_field=cluster_sequence_field)
        sequence = _cluster_sequence(sample, cluster_sequence_field=cluster_sequence_field)
        if record_id not in records:
            records[record_id] = sequence
        elif records[record_id] != sequence:
            raise ValueError(f"Conflicting sequences observed for cluster record {record_id!r}.")
    if not records:
        raise ValueError(f"No sequences available for MMseqs clustering field {cluster_sequence_field!r}.")
    return records


def _cluster_by_sample_id(
    samples: list[Any],
    *,
    cluster_sequence_field: str,
    cluster_by_record_id: dict[str, str],
) -> dict[str, str]:
    cluster_by_sample_id: dict[str, str] = {}
    for sample in samples:
        record_id = _cluster_record_id(sample, cluster_sequence_field=cluster_sequence_field)
        cluster_by_sample_id[sample.sample_id] = cluster_by_record_id[record_id]
    return cluster_by_sample_id


def _cluster_record_id(sample: Any, *, cluster_sequence_field: str) -> str:
    if cluster_sequence_field == "sequence":
        return str(sample.sequence_hash)
    if cluster_sequence_field == "peptide_sequence":
        peptide_hash = str(getattr(sample, "peptide_sequence_hash", "") or "").strip()
        if not peptide_hash:
            raise ValueError(f"Sample {sample.sample_id!r} is missing peptide_sequence_hash for peptide clustering.")
        return peptide_hash
    raise ValueError(f"Unsupported cluster_sequence_field: {cluster_sequence_field}")


def _cluster_sequence(sample: Any, *, cluster_sequence_field: str) -> str:
    sequence = getattr(sample, cluster_sequence_field, None)
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError(f"Sample {sample.sample_id!r} is missing {cluster_sequence_field} for MMseqs clustering.")
    return sequence.strip()


def _run_mmseqs_cluster(
    sequence_records: dict[str, str],
    *,
    mmseqs_binary: str,
    cluster_min_seq_id: float,
    cluster_coverage: float,
    cluster_cov_mode: int,
    split_source_name: str,
) -> dict[str, str]:
    prefix = _safe_prefix(split_source_name)
    with tempfile.TemporaryDirectory(prefix=f"{prefix}_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        fasta_path = tmp_root / "all_sequences.fasta"
        cluster_prefix = tmp_root / "clustered"
        cluster_tsv_path = tmp_root / "clustered_cluster.tsv"
        tmp_workdir = tmp_root / "work"

        with fasta_path.open("w", encoding="utf-8") as handle:
            for record_id, sequence in sorted(sequence_records.items()):
                handle.write(f">{record_id}\n{sequence}\n")

        command = [
            mmseqs_binary,
            "easy-cluster",
            str(fasta_path),
            str(cluster_prefix),
            str(tmp_workdir),
            "--min-seq-id",
            str(cluster_min_seq_id),
            "-c",
            str(cluster_coverage),
            "--cov-mode",
            str(cluster_cov_mode),
        ]
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not cluster_tsv_path.exists():
            raise FileNotFoundError(
                "MMseqs2 clustering completed without producing the expected cluster TSV: "
                f"{cluster_tsv_path}. stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )

        cluster_by_record_id: dict[str, str] = {}
        with cluster_tsv_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                representative, member = stripped.split("\t")
                cluster_by_record_id[member] = representative

    missing_record_ids = sorted(record_id for record_id in sequence_records if record_id not in cluster_by_record_id)
    if missing_record_ids:
        preview = ", ".join(missing_record_ids[:5])
        raise ValueError(
            f"MMseqs2 clustering missed {len(missing_record_ids)} sequence(s). "
            f"First missing record id(s): {preview}"
        )
    return cluster_by_record_id


def _assign_cluster_splits(
    samples: list[Any],
    *,
    cluster_by_sample_id: dict[str, str],
    cluster_train_fraction: float,
    cluster_validation_fraction: float,
    cluster_seed: int | None,
    cluster_balance_labels: bool = False,
    cluster_shuffle_order: bool = False,
) -> tuple[dict[str, str], dict[str, int]]:
    total_samples = len(samples)
    target_train = int(round(total_samples * cluster_train_fraction))
    target_validation = int(round(total_samples * cluster_validation_fraction))
    target_test = total_samples - target_train - target_validation

    cluster_to_samples: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        cluster_to_samples[cluster_by_sample_id[sample.sample_id]].append(sample)

    cluster_ids = list(cluster_to_samples)
    rng = random.Random(cluster_seed) if cluster_seed is not None else random.SystemRandom()
    if cluster_shuffle_order:
        rng.shuffle(cluster_ids)
    else:
        random_tiebreak = {cluster_id: rng.random() for cluster_id in cluster_ids}
        cluster_ids.sort(
            key=lambda cluster_id: (
                len(cluster_to_samples[cluster_id]),
                random_tiebreak[cluster_id],
            ),
            reverse=True,
        )

    split_sizes = {
        "train_target_samples": target_train,
        "validation_target_samples": target_validation,
        "test_target_samples": target_test,
    }
    assigned_sizes = {"train": 0, "validation": 0, "test": 0}
    targets = {"train": target_train, "validation": target_validation, "test": target_test}
    fractions = {
        "train": float(cluster_train_fraction),
        "validation": float(cluster_validation_fraction),
        "test": 1.0 - float(cluster_train_fraction) - float(cluster_validation_fraction),
    }
    labels = sorted({str(getattr(sample, "raw_label", "")).strip() for sample in samples if str(getattr(sample, "raw_label", "")).strip()})
    global_label_counts = Counter(str(getattr(sample, "raw_label", "")).strip() for sample in samples)
    label_targets = {
        split_name: {
            label: global_label_counts[label] * fractions[split_name]
            for label in labels
        }
        for split_name in assigned_sizes
    }
    assigned_label_counts = {split_name: Counter() for split_name in assigned_sizes}
    cluster_label_counts = {
        cluster_id: Counter(
            str(getattr(sample, "raw_label", "")).strip()
            for sample in cluster_to_samples[cluster_id]
            if str(getattr(sample, "raw_label", "")).strip()
        )
        for cluster_id in cluster_ids
    }
    split_by_cluster_id: dict[str, str] = {}

    for cluster_id in cluster_ids:
        cluster_size = len(cluster_to_samples[cluster_id])
        current_cluster_label_counts = cluster_label_counts[cluster_id]
        candidate_order = sorted(
            assigned_sizes,
            key=lambda split_name: _cluster_split_score(
                split_name,
                assigned_sizes=assigned_sizes,
                targets=targets,
                cluster_size=cluster_size,
                assigned_label_counts=assigned_label_counts,
                cluster_label_counts=current_cluster_label_counts,
                label_targets=label_targets,
                labels=labels,
                cluster_balance_labels=cluster_balance_labels,
            ),
        )
        chosen_split = candidate_order[0]
        split_by_cluster_id[cluster_id] = chosen_split
        assigned_sizes[chosen_split] += cluster_size
        assigned_label_counts[chosen_split].update(current_cluster_label_counts)

    split_sizes.update(
        {
            "train_assigned_samples": assigned_sizes["train"],
            "validation_assigned_samples": assigned_sizes["validation"],
            "test_assigned_samples": assigned_sizes["test"],
        }
    )
    for split_name, label_counts in assigned_label_counts.items():
        for label in labels:
            split_sizes[f"{split_name}_label_{label}_target_samples"] = round(label_targets[split_name][label], 3)
            split_sizes[f"{split_name}_label_{label}_assigned_samples"] = int(label_counts[label])
    return split_by_cluster_id, split_sizes


def _cluster_split_score(
    split_name: str,
    *,
    assigned_sizes: dict[str, int],
    targets: dict[str, int],
    cluster_size: int,
    assigned_label_counts: dict[str, Counter],
    cluster_label_counts: Counter,
    label_targets: dict[str, dict[str, float]],
    labels: list[str],
    cluster_balance_labels: bool,
) -> tuple[Any, ...]:
    next_size = assigned_sizes[split_name] + cluster_size
    size_deviation = abs(next_size - targets[split_name])
    overflow = assigned_sizes[split_name] >= targets[split_name]
    label_deviation = 0.0
    if cluster_balance_labels and labels:
        label_deviation = sum(
            abs(
                assigned_label_counts[split_name][label]
                + cluster_label_counts[label]
                - label_targets[split_name][label]
            )
            for label in labels
        )
    return (
        overflow,
        size_deviation,
        label_deviation,
        assigned_sizes[split_name],
        split_name,
    )


def _invert_split_map(split_by_cluster_id: dict[str, str]) -> dict[str, set[str]]:
    split_sets_by_cluster: dict[str, set[str]] = defaultdict(set)
    for cluster_id, split_name in split_by_cluster_id.items():
        split_sets_by_cluster[cluster_id].add(split_name)
    return split_sets_by_cluster


def _label_counts_by_split(*split_items: tuple[str, list[Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name, samples in split_items:
        for sample in samples:
            label = str(getattr(sample, "raw_label", "")).strip()
            if not label:
                continue
            counts[f"{split_name}_label_{label}_samples"] = counts.get(f"{split_name}_label_{label}_samples", 0) + 1
    return counts


def _safe_prefix(raw_value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in raw_value.strip().lower()).strip("_") or "mmseqs"
