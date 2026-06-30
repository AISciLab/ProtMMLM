from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from typing import Any

from src.datasets.md_pairwise_terms_dataset import MD_PAIRWISE_MANIFEST_FIELDS, load_md_pairwise_manifest


def split_md_pairwise_manifest_by_mmseqs(
    *,
    manifest_path: str | Path,
    output_manifest_path: str | Path,
    summary_path: str | Path,
    mmseqs_binary: str = "mmseqs",
    cluster_min_seq_id: float = 0.4,
    cluster_coverage: float = 0.8,
    cluster_cov_mode: int = 0,
    cluster_train_fraction: float = 0.8,
    cluster_validation_fraction: float = 0.1,
    cluster_seed: int = 42,
    reuse_clusters_from: str | Path | None = None,
) -> dict[str, Any]:
    source_manifest_path = reuse_clusters_from or manifest_path
    samples = load_md_pairwise_manifest(source_manifest_path)
    if not samples:
        raise ValueError(f"No MD pairwise samples were loaded from {source_manifest_path}.")
    unique_by_hash = {sample.sequence_hash: sample for sample in samples}

    if reuse_clusters_from is None:
        if shutil.which(mmseqs_binary) is None:
            raise FileNotFoundError(f"MMseqs binary was not found: {mmseqs_binary}")
        hash_to_cluster = _run_mmseqs_cluster(
            unique_by_hash,
            mmseqs_binary=mmseqs_binary,
            cluster_min_seq_id=cluster_min_seq_id,
            cluster_coverage=cluster_coverage,
            cluster_cov_mode=cluster_cov_mode,
        )
    else:
        hash_to_cluster = _load_existing_cluster_assignments(reuse_clusters_from)
        missing = sorted(set(unique_by_hash) - set(hash_to_cluster))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Existing cluster manifest misses {len(missing)} sequence hash(es). First missing: {preview}"
            )

    cluster_ids = sorted(set(hash_to_cluster.values()))
    rng = random.Random(int(cluster_seed))
    rng.shuffle(cluster_ids)
    train_target = int(round(len(cluster_ids) * float(cluster_train_fraction)))
    validation_target = int(round(len(cluster_ids) * float(cluster_validation_fraction)))
    train_clusters = set(cluster_ids[:train_target])
    validation_clusters = set(cluster_ids[train_target : train_target + validation_target])
    test_clusters = set(cluster_ids[train_target + validation_target :])

    split_by_cluster: dict[str, str] = {}
    for cluster_id in train_clusters:
        split_by_cluster[cluster_id] = "train"
    for cluster_id in validation_clusters:
        split_by_cluster[cluster_id] = "validation"
    for cluster_id in test_clusters:
        split_by_cluster[cluster_id] = "test"

    rows: list[dict[str, str]] = []
    seen_hash_split: dict[str, str] = {}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for sample in samples:
        cluster_id = hash_to_cluster.get(sample.sequence_hash, sample.sequence_hash)
        split = split_by_cluster.get(cluster_id, "test")
        previous_split = seen_hash_split.get(sample.sequence_hash)
        if previous_split is not None and previous_split != split:
            raise ValueError("Exact sequence leakage detected across MD pairwise splits.")
        seen_hash_split[sample.sequence_hash] = split
        split_counts[split] += 1
        rows.append(
            {
                "sample_id": sample.sample_id,
                "protein_id": sample.protein_id,
                "sequence": sample.sequence,
                "sequence_hash": sample.sequence_hash,
                "nature_path": sample.nature_path,
                "md_path": sample.md_path,
                "interaction_tsv_path": sample.interaction_tsv_path,
                "map_path": sample.map_path,
                "length": str(sample.length),
                "total_frames": str(sample.total_frames),
                "split": split,
                "cluster_id": cluster_id,
            }
        )

    out_manifest = Path(output_manifest_path)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MD_PAIRWISE_MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest_path": str(source_manifest_path),
        "output_manifest_path": str(out_manifest),
        "summary_path": str(summary_path),
        "num_samples": len(samples),
        "num_unique_sequences": len(unique_by_hash),
        "num_clusters": len(cluster_ids),
        "split_counts": split_counts,
        "cluster_min_seq_id": cluster_min_seq_id,
        "cluster_coverage": cluster_coverage,
        "cluster_cov_mode": cluster_cov_mode,
        "cluster_seed": cluster_seed,
        "reuse_clusters_from": None if reuse_clusters_from is None else str(reuse_clusters_from),
    }
    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _run_mmseqs_cluster(
    unique_by_hash: dict[str, Any],
    *,
    mmseqs_binary: str,
    cluster_min_seq_id: float,
    cluster_coverage: float,
    cluster_cov_mode: int,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="md_pairwise_mmseqs_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        fasta_path = tmp_dir / "all_sequences.fasta"
        with fasta_path.open("w", encoding="utf-8") as handle:
            for sequence_hash, sample in sorted(unique_by_hash.items()):
                handle.write(f">{sequence_hash}\n{sample.sequence}\n")
        cluster_prefix = tmp_dir / "clustered"
        work_dir = tmp_dir / "work"
        command = [
            mmseqs_binary,
            "easy-cluster",
            str(fasta_path),
            str(cluster_prefix),
            str(work_dir),
            "--min-seq-id",
            str(cluster_min_seq_id),
            "-c",
            str(cluster_coverage),
            "--cov-mode",
            str(cluster_cov_mode),
        ]
        try:
            completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "MMseqs easy-cluster failed.\n"
                f"command={' '.join(command)}\n"
                f"returncode={exc.returncode}\n"
                f"stdout={exc.stdout}\n"
                f"stderr={exc.stderr}"
            ) from exc
        cluster_tsv = cluster_prefix.with_name(cluster_prefix.name + "_cluster.tsv")
        if not cluster_tsv.exists():
            raise FileNotFoundError(
                f"MMseqs did not create cluster file: {cluster_tsv}\n"
                f"stdout={completed.stdout}\nstderr={completed.stderr}"
            )
        hash_to_cluster: dict[str, str] = {}
        with cluster_tsv.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.strip().split("\t")
                if len(parts) >= 2:
                    representative, member = parts[0], parts[1]
                    hash_to_cluster[member] = representative
    for sequence_hash in unique_by_hash:
        hash_to_cluster.setdefault(sequence_hash, sequence_hash)
    return hash_to_cluster


def _load_existing_cluster_assignments(manifest_path: str | Path) -> dict[str, str]:
    hash_to_cluster: dict[str, str] = {}
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence_hash = str(row.get("sequence_hash") or "").strip()
            cluster_id = str(row.get("cluster_id") or "").strip()
            if sequence_hash and cluster_id:
                previous = hash_to_cluster.get(sequence_hash)
                if previous is not None and previous != cluster_id:
                    raise ValueError(
                        f"Conflicting cluster_id for sequence_hash={sequence_hash}: {previous!r} vs {cluster_id!r}"
                    )
                hash_to_cluster[sequence_hash] = cluster_id
    if not hash_to_cluster:
        raise ValueError(f"No cluster_id assignments were found in {manifest_path}.")
    return hash_to_cluster
