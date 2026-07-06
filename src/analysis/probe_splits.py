from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from typing import Any, Callable

from src.utils.sequence import sequence_hash


def resolve_or_build_probe_split_file(
    *,
    split_file: str | Path | None,
    split_source_manifest: str | Path | None,
    pretrain_manifest_path: str | Path,
    output_dir: str | Path,
    split_output_dir: str | Path | None,
    repo_root: str | Path,
    mmseqs_splitter: Callable[..., dict[str, Any]],
    mmseqs_binary: str = "mmseqs",
    cluster_min_seq_id: float = 0.4,
    cluster_coverage: float = 0.8,
    cluster_cov_mode: int = 0,
    cluster_train_fraction: float = 0.8,
    cluster_validation_fraction: float = 0.1,
    cluster_seed: int = 42,
) -> Path:
    """Resolve a protein split for probing without requiring pairwise preprocessing.

    The original analysis scripts built splits from the MD pairwise-term manifest.
    Release users often only have the pretraining manifest, so this helper falls
    back to a deterministic protein-level split when that pairwise manifest is
    absent.
    """
    root = Path(repo_root)
    if split_file is not None:
        return _resolve_existing_path(split_file, root)

    out_dir = _resolve_output_path(split_output_dir, root) if split_output_dir is not None else Path(output_dir) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    identity_label = _format_identity_label(float(cluster_min_seq_id))
    pairwise_source = _resolve_candidate_path(split_source_manifest, root) if split_source_manifest is not None else None
    if pairwise_source is not None and pairwise_source.exists():
        mmseqs_split = out_dir / f"protein_mmseqs{identity_label}_seed{cluster_seed}.csv"
        mmseqs_summary = out_dir / f"protein_mmseqs{identity_label}_seed{cluster_seed}_summary.json"
        try:
            summary = mmseqs_splitter(
                manifest_path=pairwise_source,
                output_manifest_path=mmseqs_split,
                summary_path=mmseqs_summary,
                mmseqs_binary=str(mmseqs_binary),
                cluster_min_seq_id=float(cluster_min_seq_id),
                cluster_coverage=float(cluster_coverage),
                cluster_cov_mode=int(cluster_cov_mode),
                cluster_train_fraction=float(cluster_train_fraction),
                cluster_validation_fraction=float(cluster_validation_fraction),
                cluster_seed=int(cluster_seed),
            )
            print(f"[split] {json.dumps(summary, indent=2, sort_keys=True)}", flush=True)
            return mmseqs_split
        except FileNotFoundError as exc:
            print(f"[split] MMseqs split unavailable ({exc}); falling back to pretrain manifest split.", flush=True)

    fallback_split = out_dir / f"protein_manifest_seed{cluster_seed}.csv"
    fallback_summary = out_dir / f"protein_manifest_seed{cluster_seed}_summary.json"
    summary = build_manifest_protein_split(
        manifest_path=pretrain_manifest_path,
        output_manifest_path=fallback_split,
        summary_path=fallback_summary,
        repo_root=root,
        train_fraction=float(cluster_train_fraction),
        validation_fraction=float(cluster_validation_fraction),
        seed=int(cluster_seed),
        missing_pairwise_manifest=None if pairwise_source is None else str(pairwise_source),
    )
    print(f"[split] {json.dumps(summary, indent=2, sort_keys=True)}", flush=True)
    return fallback_split


def build_manifest_protein_split(
    *,
    manifest_path: str | Path,
    output_manifest_path: str | Path,
    summary_path: str | Path,
    repo_root: str | Path,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    missing_pairwise_manifest: str | None = None,
) -> dict[str, Any]:
    resolved_manifest = _resolve_existing_path(manifest_path, Path(repo_root))
    samples_by_id = _load_split_candidates_from_pretrain_manifest(resolved_manifest)
    protein_ids = sorted(samples_by_id)
    if not protein_ids:
        raise ValueError(
            f"No protein records were loaded from {resolved_manifest}. "
            "Please check that the manifest contains protein_id and sequence columns."
        )

    rng = random.Random(int(seed))
    rng.shuffle(protein_ids)
    train_target = int(round(len(protein_ids) * float(train_fraction)))
    validation_target = int(round(len(protein_ids) * float(validation_fraction)))
    train_ids = set(protein_ids[:train_target])
    validation_ids = set(protein_ids[train_target : train_target + validation_target])

    rows: list[dict[str, str]] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for protein_id in sorted(protein_ids):
        sample = samples_by_id[protein_id]
        if protein_id in train_ids:
            split = "train"
        elif protein_id in validation_ids:
            split = "validation"
        else:
            split = "test"
        split_counts[split] += 1
        rows.append(
            {
                "protein_id": sample["protein_id"],
                "sequence_hash": sample["sequence_hash"],
                "nature_path": sample["nature_path"],
                "md_path": sample["md_path"],
                "split": split,
                "cluster_id": sample["sequence_hash"] or sample["protein_id"],
            }
        )

    out_manifest = Path(output_manifest_path)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "sequence_hash", "nature_path", "md_path", "split", "cluster_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest_path": str(resolved_manifest),
        "output_manifest_path": str(out_manifest),
        "summary_path": str(summary_path),
        "split_strategy": "pretrain_manifest_protein_random",
        "missing_pairwise_manifest": missing_pairwise_manifest,
        "num_samples": len(rows),
        "split_counts": split_counts,
        "cluster_train_fraction": train_fraction,
        "cluster_validation_fraction": validation_fraction,
        "cluster_seed": seed,
    }
    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def _load_split_candidates_from_pretrain_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    samples_by_id: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = str(row.get("protein_id") or row.get("sample_id") or "").strip()
            sequence = str(row.get("sequence") or "").strip()
            if not protein_id:
                continue
            sequence_hash_value = str(row.get("sequence_hash") or "").strip()
            if not sequence_hash_value and sequence:
                try:
                    sequence_hash_value = sequence_hash(sequence)
                except ValueError:
                    sequence_hash_value = protein_id
            samples_by_id.setdefault(
                protein_id,
                {
                    "protein_id": protein_id,
                    "sequence_hash": sequence_hash_value or protein_id,
                    "nature_path": str(row.get("nature_path") or "").strip(),
                    "md_path": str(row.get("md_path") or "").strip(),
                },
            )
    return samples_by_id


def _resolve_existing_path(path: str | Path, repo_root: Path) -> Path:
    resolved = _resolve_candidate_path(path, repo_root)
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def _resolve_candidate_path(path: str | Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _resolve_output_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _format_identity_label(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")
