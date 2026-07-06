from __future__ import annotations

import csv
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SequenceClusterRecord:
    record_id: str
    sequence: str
    label: str | None = None


@dataclass(frozen=True)
class ClusterSplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    seed: int | None = 42
    shuffle_order: bool = False


FASTA_WRAP = 80


def read_sequence_records_from_csv(
    path: str | Path,
    *,
    id_column: str,
    sequence_column: str,
    label_column: str | None = None,
) -> list[SequenceClusterRecord]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    records: list[SequenceClusterRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in (id_column, sequence_column) if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required column(s): {missing_columns}")
        if label_column is not None and label_column not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing label column: {label_column}")
        for row_number, row in enumerate(reader, start=2):
            record_id = str(row.get(id_column) or "").strip()
            sequence = normalize_cluster_sequence(row.get(sequence_column))
            if not record_id or not sequence:
                raise ValueError(f"Missing record id or sequence at {csv_path}:{row_number}")
            label = None if label_column is None else str(row.get(label_column) or "").strip() or None
            records.append(SequenceClusterRecord(record_id=record_id, sequence=sequence, label=label))
    return dedupe_sequence_records(records)


def dedupe_sequence_records(records: Iterable[SequenceClusterRecord]) -> list[SequenceClusterRecord]:
    by_record_id: dict[str, SequenceClusterRecord] = {}
    for record in records:
        existing = by_record_id.get(record.record_id)
        if existing is None:
            by_record_id[record.record_id] = record
            continue
        if existing.sequence != record.sequence:
            raise ValueError(f"Conflicting sequences for record_id={record.record_id!r}.")
    return [by_record_id[key] for key in sorted(by_record_id)]


def write_sequence_fasta(records: Iterable[SequenceClusterRecord], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.record_id):
            handle.write(f">{record.record_id}\n")
            for start in range(0, len(record.sequence), FASTA_WRAP):
                handle.write(f"{record.sequence[start:start + FASTA_WRAP]}\n")
    return path


def run_mmseqs_easy_cluster(
    fasta_path: str | Path,
    *,
    output_prefix: str | Path,
    tmp_dir: str | Path,
    mmseqs_binary: str = "mmseqs",
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    cov_mode: int = 0,
) -> dict[str, str]:
    fasta = Path(fasta_path)
    prefix = Path(output_prefix)
    work_dir = Path(tmp_dir)
    if not fasta.exists():
        raise FileNotFoundError(f"MMseqs input FASTA does not exist: {fasta}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    command = [
        mmseqs_binary,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(work_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    cluster_tsv = prefix.with_name(f"{prefix.name}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(
            "MMseqs2 completed without producing the expected cluster TSV: "
            f"{cluster_tsv}. stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return read_mmseqs_cluster_tsv(cluster_tsv)


def read_mmseqs_cluster_tsv(path: str | Path) -> dict[str, str]:
    cluster_tsv = Path(path)
    cluster_by_record_id: dict[str, str] = {}
    with cluster_tsv.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid MMseqs cluster row at {cluster_tsv}:{line_number}: {stripped!r}")
            representative, member = parts
            cluster_by_record_id[member] = representative
    return cluster_by_record_id


def assign_clusters_to_splits(
    records: Sequence[SequenceClusterRecord],
    cluster_by_record_id: dict[str, str],
    *,
    config: ClusterSplitConfig,
) -> dict[str, str]:
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {config.train_fraction}.")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be between 0 and 1, got {config.validation_fraction}.")
    test_fraction = 1.0 - config.train_fraction - config.validation_fraction
    if test_fraction <= 0.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")

    cluster_to_records: dict[str, list[SequenceClusterRecord]] = {}
    for record in records:
        cluster_id = cluster_by_record_id.get(record.record_id)
        if cluster_id is None:
            raise ValueError(f"Record {record.record_id!r} is absent from MMseqs cluster assignments.")
        cluster_to_records.setdefault(cluster_id, []).append(record)

    rng = random.Random(config.seed) if config.seed is not None else random.SystemRandom()
    cluster_ids = list(cluster_to_records)
    if config.shuffle_order:
        rng.shuffle(cluster_ids)
    else:
        tie_break = {cluster_id: rng.random() for cluster_id in cluster_ids}
        cluster_ids.sort(key=lambda cluster_id: (len(cluster_to_records[cluster_id]), tie_break[cluster_id]), reverse=True)

    total_records = len(records)
    targets = {
        "train": int(round(total_records * config.train_fraction)),
        "validation": int(round(total_records * config.validation_fraction)),
    }
    targets["test"] = total_records - targets["train"] - targets["validation"]
    assigned_sizes = {"train": 0, "validation": 0, "test": 0}
    split_by_cluster: dict[str, str] = {}
    for cluster_id in cluster_ids:
        cluster_size = len(cluster_to_records[cluster_id])
        chosen_split = min(
            assigned_sizes,
            key=lambda split_name: (
                assigned_sizes[split_name] >= targets[split_name],
                abs((assigned_sizes[split_name] + cluster_size) - targets[split_name]),
                assigned_sizes[split_name],
                split_name,
            ),
        )
        split_by_cluster[cluster_id] = chosen_split
        assigned_sizes[chosen_split] += cluster_size
    return split_by_cluster


def write_cluster_assignments(
    records: Sequence[SequenceClusterRecord],
    cluster_by_record_id: dict[str, str],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "cluster_id", "label"])
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.record_id):
            writer.writerow({
                "record_id": record.record_id,
                "cluster_id": cluster_by_record_id[record.record_id],
                "label": record.label or "",
            })
    return path


def write_split_assignments(
    records: Sequence[SequenceClusterRecord],
    cluster_by_record_id: dict[str, str],
    split_by_cluster_id: dict[str, str],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "cluster_id", "split", "label"])
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.record_id):
            cluster_id = cluster_by_record_id[record.record_id]
            writer.writerow({
                "record_id": record.record_id,
                "cluster_id": cluster_id,
                "split": split_by_cluster_id[cluster_id],
                "label": record.label or "",
            })
    return path


def write_split_summary(
    records: Sequence[SequenceClusterRecord],
    cluster_by_record_id: dict[str, str],
    split_by_cluster_id: dict[str, str],
    output_path: str | Path,
) -> Path:
    split_counts = {"train": 0, "validation": 0, "test": 0}
    cluster_counts = {"train": set(), "validation": set(), "test": set()}
    for record in records:
        cluster_id = cluster_by_record_id[record.record_id]
        split = split_by_cluster_id[cluster_id]
        split_counts[split] = split_counts.get(split, 0) + 1
        cluster_counts.setdefault(split, set()).add(cluster_id)
    summary = {
        "num_records": len(records),
        "num_clusters": len(set(cluster_by_record_id.values())),
        "split_counts": split_counts,
        "cluster_counts": {split: len(ids) for split, ids in sorted(cluster_counts.items())},
        "cluster_leakage_check_passed": _cluster_leakage_check(cluster_by_record_id, split_by_cluster_id),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _cluster_leakage_check(cluster_by_record_id: dict[str, str], split_by_cluster_id: dict[str, str]) -> bool:
    return all(cluster_id in split_by_cluster_id for cluster_id in set(cluster_by_record_id.values()))


def normalize_cluster_sequence(raw_sequence: object) -> str:
    return "".join(str(raw_sequence or "").strip().upper().split())
