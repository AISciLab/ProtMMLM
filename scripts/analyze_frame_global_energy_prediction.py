#!/usr/bin/env python3
from __future__ import annotations

# RELEASE_IMPORT_BOOTSTRAP: allow running scripts directly from the repository root.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np

from src.analysis.trajectory_embeddings import extract_trajectory_embeddings
from src.analysis.probe_splits import resolve_or_build_probe_split_file
from src.datasets.md_pairwise_terms_splits import split_md_pairwise_manifest_by_mmseqs
from src.datasets.pretrain_dataset import PretrainDataset
from src.datasets.structure_io import CAFrame, _extract_md_ca_frames, _sample_ca_frames
from src.training.pretrain_trainer import PretrainTrainerConfig, build_pretrain_trainer


TARGETS = ("total_score",)
PROBES = ("linear", "svm", "mlp")
RESULT_FIELDS = [
    "target",
    "probe",
    "pca_components",
    "n_train_frames",
    "n_validation_frames",
    "n_test_frames",
    "n_train_proteins",
    "n_validation_proteins",
    "n_test_proteins",
    "mae",
    "mse",
    "rmse",
    "pearson",
    "spearman",
    "r2",
]
PREDICTION_FIELDS = [
    "protein_id",
    "frame_id",
    "split",
    "target",
    "probe",
    "pca_components",
    "true_value",
    "pred_value",
]
FRAME_TARGET_FIELDS = ["protein_id", "frame_id", "split", "total_score"]


def _tqdm(iterable: Any, **kwargs: Any) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:  # noqa: BLE001
        return iterable


def _normalize_thread_env() -> None:
    for variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        raw_value = os.environ.get(variable_name)
        if raw_value is None:
            continue
        stripped_value = raw_value.strip()
        os.environ[variable_name] = stripped_value if stripped_value.isdigit() and int(stripped_value) > 0 else "1"


_normalize_thread_env()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen Fusion global embedding probes for frame-level total_score prediction."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain/pretrain.yaml"))
    parser.add_argument("--config-override", type=Path, action="append", default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--nature-dir", type=Path, default=None, help="Optional native-structure directory used to remap manifest nature_path values by protein_id.")
    parser.add_argument("--md-dir", type=Path, default=None, help="Optional MD directory used to remap manifest md_path values by protein_id.")
    parser.add_argument("--energy-csv", type=Path, default=Path("datasets/downstream/Thermodynamic/thermodynamic.csv"), help="Frame-level thermodynamic CSV with sample, frame, and total_score columns.")
    parser.add_argument(
        "--energy-frame-offset",
        type=int,
        default=1,
        help="Offset added to zero-based MD frame indices before matching the energy CSV frame column.",
    )
    parser.add_argument("--split-file", type=Path, default=None, help="Existing protein-level split CSV. If omitted, the script uses the pairwise manifest for an MMseqs split or falls back to a pretrain-manifest split.")
    parser.add_argument("--split-source-manifest", type=Path, default=Path("results/md_pairwise_terms/md_pairwise_terms_manifest.csv"))
    parser.add_argument("--split-output-dir", type=Path, default=None)
    parser.add_argument("--mmseqs-binary", type=str, default="mmseqs")
    parser.add_argument("--cluster-min-seq-id", type=float, default=0.4)
    parser.add_argument("--cluster-coverage", type=float, default=0.8)
    parser.add_argument("--cluster-cov-mode", type=int, default=0)
    parser.add_argument("--cluster-train-fraction", type=float, default=0.8)
    parser.add_argument("--cluster-validation-fraction", type=float, default=0.1)
    parser.add_argument("--cluster-seed", type=int, default=None, help="Seed for assigning MMseqs clusters to train/validation/test. Defaults to --seed.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--protein-id", type=str, default=None)
    parser.add_argument("--max-residues", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-embedding-mode", choices=("token_aggregate", "per_frame"), default="token_aggregate")
    parser.add_argument("--fusion-embedding-source", choices=("fused_pooled", "fused_cls"), default="fused_pooled")
    parser.add_argument(
        "--pca-components",
        type=str,
        default="none,50,100,200",
        help="Comma-separated PCA dimensions. Use 'none' or 'raw' to train probes on standardized full embeddings.",
    )
    parser.add_argument(
        "--probe",
        type=str,
        default="all",
        help="Probe to run: linear, svm, mlp, all, or comma-separated values such as linear,mlp.",
    )
    parser.add_argument("--mlp-hidden-dims", type=str, default="512,128")
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-epochs", type=int, default=200)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-lr", type=float, default=1.0e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--mlp-patience", type=int, default=5)
    parser.add_argument("--mlp-early-stop-metric", choices=("spearman", "pearson", "r2", "rmse", "mae", "mse", "val_loss"), default="spearman")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-epsilon", type=float, default=0.0)
    parser.add_argument("--svm-max-iter", type=int, default=5000)
    parser.add_argument("--target-min", type=float, default=None, help="Optional lower bound for total_score; frames below this value are filtered out before caching/probing.")
    parser.add_argument("--target-max", type=float, default=None, help="Optional upper bound for total_score; frames above this value are filtered out before caching/probing.")
    parser.add_argument("--target-quantile-low", type=float, default=None, help="Optional lower quantile in [0,1] for total_score filtering, computed on the loaded labeled frames.")
    parser.add_argument("--target-quantile-high", type=float, default=None, help="Optional upper quantile in [0,1] for total_score filtering, computed on the loaded labeled frames.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Optional .npz cache for labeled frame-level frozen Fusion embeddings and total_score targets.",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Load --embedding-cache and skip frozen encoder/Fusion embedding extraction.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract/write frame targets and optional --embedding-cache, then skip PCA/probe fitting.",
    )
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--failure-policy", choices=("skip", "raise"), default="skip")
    parser.add_argument("--log-interval", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = _resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_embeddings:
        (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    embedding_cache_path = _resolve_output_path(args.embedding_cache) if args.embedding_cache is not None else None

    config_dict = _load_config_dict(args.config) if args.config is not None else {}
    for override_path in args.config_override or []:
        config_dict.update(_load_config_dict(override_path))
    if args.manifest_path is not None:
        config_dict["manifest_path"] = str(args.manifest_path)
    config_dict["checkpoint_path"] = str(args.checkpoint)
    if args.device is not None:
        config_dict["device"] = args.device
    if args.max_residues is not None:
        config_dict["max_residues"] = args.max_residues
    if args.max_frames is not None:
        config_dict["max_frames"] = args.max_frames

    config = _build_config(config_dict)
    checkpoint_path = Path(config.checkpoint_path or str(args.checkpoint))
    if not checkpoint_path.is_absolute():
        checkpoint_path = _repo_root() / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    split_file = resolve_or_build_split_file(args, output_dir, config.manifest_path)
    split_by_protein = load_split_map(split_file)
    energy_csv_path = _resolve_repo_path_required(args.energy_csv)
    energy_targets = None if args.reuse_embeddings else load_energy_targets(energy_csv_path)

    if args.reuse_embeddings:
        if embedding_cache_path is None:
            raise ValueError("--reuse-embeddings requires --embedding-cache.")
        frame_rows, embedding_rows = load_embedding_cache(
            embedding_cache_path,
            split_by_protein=split_by_protein,
            protein_id_filter=args.protein_id,
            sample_limit=args.sample_limit,
        )
        failures: list[dict[str, Any]] = []
        print(
            f"[cache] loaded frame embeddings path={embedding_cache_path} frames={len(frame_rows)}",
            flush=True,
        )
        num_samples_requested = len(set(row["protein_id"] for row in frame_rows))
    else:
        dataset = PretrainDataset.from_manifest(
            config.manifest_path,
            sample_limit=None,
            full_only=True,
            nature_dir=args.nature_dir,
            md_dir=args.md_dir,
        )
        samples = [sample for sample in dataset.samples if sample.protein_id in split_by_protein]
        if args.protein_id:
            samples = [sample for sample in samples if sample.protein_id == args.protein_id]
        if args.sample_limit is not None:
            samples = samples[: args.sample_limit]
        if not samples:
            raise ValueError(
                "No samples matched manifest/split filters. Check that --manifest-path points to a pretrain_manifest.csv "
                "whose nature_path and md_path files/directories exist on this machine. If you downloaded the released "
                "dataset, extract datasets/pretrain/nature.tar.gz and datasets/pretrain/md.tar.gz before running analysis."
            )

        trainer = build_pretrain_trainer(config)
        _load_pretrain_weights_for_inference(trainer, checkpoint_path)
        _freeze_for_inference(trainer)

        frame_rows, embedding_rows, failures = extract_dataset_rows(
            samples=samples,
            split_by_protein=split_by_protein,
            trainer=trainer,
            energy_targets=energy_targets or {},
            args=args,
            config=config,
            output_dir=output_dir,
        )
        if failures:
            write_csv(output_dir / "failures.csv", failures, ["protein_id", "frame_id", "stage", "error_type", "error_message"])
        num_samples_requested = len(samples)
    if not frame_rows:
        raise ValueError("No labeled frame rows were extracted after matching energy targets.")
    filter_summary = filter_embedding_rows_by_target(embedding_rows, args=args)
    if filter_summary["num_filtered"] > 0:
        frame_rows = rows_from_embedding_rows(embedding_rows)
        print(
            "[filter] "
            f"target=total_score before={filter_summary['num_before']} after={filter_summary['num_after']} "
            f"filtered={filter_summary['num_filtered']} lower={filter_summary['lower_bound']} upper={filter_summary['upper_bound']}",
            flush=True,
        )
    if not frame_rows:
        raise ValueError("No labeled frame rows remain after total_score filtering.")
    if embedding_cache_path is not None and not args.reuse_embeddings:
        save_embedding_cache(embedding_cache_path, embedding_rows)
        print(
            f"[cache] saved frame embeddings path={embedding_cache_path} frames={len(embedding_rows)}",
            flush=True,
        )

    write_csv(output_dir / "frame_targets.csv", frame_rows, FRAME_TARGET_FIELDS)
    if args.extract_only:
        resolved = {
            "config": config_dict,
            "checkpoint": str(checkpoint_path),
            "split_file": str(split_file),
            "energy_csv": str(energy_csv_path),
            "energy_frame_offset": int(args.energy_frame_offset),
            "output_dir": str(output_dir),
            "embedding_cache": str(embedding_cache_path) if embedding_cache_path is not None else None,
            "reuse_embeddings": bool(args.reuse_embeddings),
            "extract_only": True,
            "num_samples_requested": num_samples_requested,
            "num_frames": len(frame_rows),
            "num_labeled_frames": len(frame_rows),
            "num_energy_labels": None if energy_targets is None else len(energy_targets),
            "num_missing_energy_labels": _count_failures(failures, "MissingEnergyTarget") if "failures" in locals() else 0,
            "target_filter": filter_summary,
            "targets": TARGETS,
            "note": "Frozen representation extraction only; PCA/probe fitting was skipped.",
        }
        _write_json(output_dir / "resolved_config.json", resolved)
        print(json.dumps({"num_frames": len(frame_rows), "output_dir": str(output_dir), "extract_only": True}, indent=2), flush=True)
        return 0
    probes = parse_probes(args.probe)
    feature_transforms = parse_feature_transforms(args.pca_components)
    results, prediction_rows = run_probe_suite(
        embedding_rows=embedding_rows,
        probes=probes,
        feature_transforms=feature_transforms,
        args=args,
    )
    write_csv(output_dir / "prediction_results.csv", results, RESULT_FIELDS)
    if not args.no_predictions:
        write_csv(output_dir / "per_frame_predictions.csv", prediction_rows, PREDICTION_FIELDS)

    config_summary = trainer.config_summary() if "trainer" in locals() else config_dict
    resolved = {
        "config": config_summary,
        "checkpoint": str(checkpoint_path),
        "split_file": str(split_file),
        "energy_csv": str(energy_csv_path),
        "energy_frame_offset": int(args.energy_frame_offset),
        "output_dir": str(output_dir),
        "embedding_cache": str(embedding_cache_path) if embedding_cache_path is not None else None,
        "reuse_embeddings": bool(args.reuse_embeddings),
        "num_samples_requested": num_samples_requested,
        "num_frames": len(frame_rows),
        "num_labeled_frames": len(frame_rows),
        "num_energy_labels": None if energy_targets is None else len(energy_targets),
        "num_missing_energy_labels": _count_failures(failures, "MissingEnergyTarget") if "failures" in locals() else 0,
        "target_filter": filter_summary,
        "pca_components": [feature_transform_label(value) for value in feature_transforms],
        "probes": probes,
        "targets": TARGETS,
        "note": "Frozen representation probing only; ProtMMLM weights are not updated; only probes are trained.",
    }
    _write_json(output_dir / "resolved_config.json", resolved)
    print(json.dumps({"num_frames": len(frame_rows), "num_results": len(results), "output_dir": str(output_dir)}, indent=2), flush=True)
    return 0


def resolve_or_build_split_file(args: argparse.Namespace, output_dir: Path, manifest_path: str | Path) -> Path:
    cluster_seed = int(args.cluster_seed if args.cluster_seed is not None else args.seed)
    return resolve_or_build_probe_split_file(
        split_file=args.split_file,
        split_source_manifest=args.split_source_manifest,
        pretrain_manifest_path=manifest_path,
        output_dir=output_dir,
        split_output_dir=args.split_output_dir,
        repo_root=_repo_root(),
        mmseqs_splitter=split_md_pairwise_manifest_by_mmseqs,
        mmseqs_binary=str(args.mmseqs_binary),
        cluster_min_seq_id=float(args.cluster_min_seq_id),
        cluster_coverage=float(args.cluster_coverage),
        cluster_cov_mode=int(args.cluster_cov_mode),
        cluster_train_fraction=float(args.cluster_train_fraction),
        cluster_validation_fraction=float(args.cluster_validation_fraction),
        cluster_seed=cluster_seed,
    )


def extract_dataset_rows(
    *,
    samples: list[Any],
    split_by_protein: dict[str, str],
    energy_targets: dict[tuple[str, int], float],
    trainer: Any,
    args: argparse.Namespace,
    config: Any,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    frame_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            try:
                frames = load_md_ca_frames_preserve_indices(sample.md_path, max_residues=config.max_residues, max_frames=config.max_frames)
                if len(frames) < 2:
                    raise ValueError(f"Need at least 2 frames, got {len(frames)}.")
                embeddings = extract_trajectory_embeddings(
                    sample=sample,
                    frames=frames,
                    sequence_encoder=trainer.sequence_encoder,
                    structure_encoder=trainer.structure_encoder,
                    fusion_transformer=trainer.fusion_transformer,
                    frame_embedding_mode=args.frame_embedding_mode,
                    fusion_embedding_source=args.fusion_embedding_source,
                )
                z = np.asarray(embeddings.fusion_embeddings, dtype=np.float32)
                if z.ndim != 2 or z.shape[0] != len(frames):
                    raise ValueError(f"Expected frame embeddings [T,d], got {z.shape}.")
                expected_frame_indices = np.asarray([int(frame.frame_index) for frame in frames], dtype=np.int64)
                observed_frame_indices = np.asarray(embeddings.frame_indices, dtype=np.int64)
                if not np.array_equal(observed_frame_indices, expected_frame_indices):
                    raise ValueError(
                        "Embedding frame indices do not match loaded MD frame indices: "
                        f"observed={observed_frame_indices.tolist()} expected={expected_frame_indices.tolist()}"
                    )
                split = normalize_split(split_by_protein[sample.protein_id])
                matched_embeddings: list[np.ndarray] = []
                matched_frame_indices: list[int] = []
                matched_scores: list[float] = []
                for row_index, frame in enumerate(frames):
                    md_frame_id = int(getattr(frame, "frame_index", row_index))
                    energy_frame_id = md_frame_id + int(args.energy_frame_offset)
                    key = (sample.protein_id, energy_frame_id)
                    if key not in energy_targets:
                        failures.append(
                            {
                                "protein_id": sample.protein_id,
                                "frame_id": energy_frame_id,
                                "stage": "match_energy_target",
                                "error_type": "MissingEnergyTarget",
                                "error_message": "No total_score for (sample, frame).",
                            }
                        )
                        continue
                    total_score = float(energy_targets[key])
                    frame_rows.append(
                        {
                            "protein_id": sample.protein_id,
                            "frame_id": energy_frame_id,
                            "split": split,
                            "total_score": total_score,
                        }
                    )
                    embedding = z[row_index]
                    embedding_rows.append(
                        {
                            "protein_id": sample.protein_id,
                            "frame_id": energy_frame_id,
                            "split": split,
                            "embedding": embedding,
                            "targets": {"total_score": total_score},
                        }
                    )
                    matched_embeddings.append(embedding)
                    matched_frame_indices.append(energy_frame_id)
                    matched_scores.append(total_score)
                if args.save_embeddings and matched_embeddings:
                    np.savez_compressed(
                        output_dir / "embeddings" / f"{_safe_name(sample.protein_id)}.npz",
                        fusion_embeddings=np.stack(matched_embeddings, axis=0).astype(np.float32),
                        frame_indices=np.asarray(matched_frame_indices, dtype=np.int64),
                        total_score=np.asarray(matched_scores, dtype=np.float32),
                    )
            except Exception as exc:  # noqa: BLE001
                if args.failure_policy == "raise":
                    raise
                failures.append(
                    {
                        "protein_id": sample.protein_id,
                        "frame_id": "",
                        "stage": "extract_dataset_rows",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            if args.log_interval > 0 and (index + 1) % args.log_interval == 0:
                print(f"processed={index + 1}/{len(samples)} labeled_frames={len(frame_rows)} failures={len(failures)}", flush=True)
    return frame_rows, embedding_rows, failures


def load_md_ca_frames_preserve_indices(
    path: str | Path,
    *,
    max_residues: int = 100,
    max_frames: int = 160,
) -> list[CAFrame]:
    md_path = Path(path)
    frames = _extract_md_ca_frames(md_path, max_residues=max_residues)
    return list(_sample_ca_frames(frames, seed_path=md_path, max_frames=max_frames))


def load_energy_targets(path: Path) -> dict[tuple[str, int], float]:
    if not path.exists():
        raise FileNotFoundError(f"Energy CSV does not exist: {path}")
    targets: dict[tuple[str, int], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Energy CSV has no header: {path}")
        required_columns = {"sample", "frame", "total_score"}
        missing_columns = sorted(required_columns.difference(fieldnames))
        if missing_columns:
            raise ValueError(f"Energy CSV is missing required columns {missing_columns}: {path}; columns={fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            sample = str(row.get("sample", "")).strip()
            if not sample:
                raise ValueError(f"Energy CSV has empty sample at row {row_number}: {path}")
            frame_raw = str(row.get("frame", "")).strip()
            score_raw = str(row.get("total_score", "")).strip()
            try:
                frame_value = float(frame_raw)
            except ValueError as exc:
                raise ValueError(f"Energy CSV has invalid frame {frame_raw!r} at row {row_number}: {path}") from exc
            if not math.isfinite(frame_value) or not frame_value.is_integer():
                raise ValueError(f"Energy CSV has non-integer frame {frame_raw!r} at row {row_number}: {path}")
            frame_id = int(frame_value)
            try:
                total_score = float(score_raw)
            except ValueError as exc:
                raise ValueError(f"Energy CSV has invalid total_score {score_raw!r} at row {row_number}: {path}") from exc
            if not math.isfinite(total_score):
                raise ValueError(f"Energy CSV has non-finite total_score at row {row_number}: {path}")
            key = (sample, frame_id)
            if key in targets:
                raise ValueError(f"Energy CSV contains duplicate (sample, frame) key {key}: {path}")
            targets[key] = total_score
    if not targets:
        raise ValueError(f"Energy CSV contains no target rows: {path}")
    return targets


def save_embedding_cache(path: Path, embedding_rows: list[dict[str, Any]]) -> None:
    if not embedding_rows:
        raise ValueError("Cannot save an empty embedding cache.")
    path.parent.mkdir(parents=True, exist_ok=True)
    protein_ids = np.asarray([str(row["protein_id"]) for row in embedding_rows], dtype=object)
    frame_ids = np.asarray([int(row["frame_id"]) for row in embedding_rows], dtype=np.int64)
    embeddings = np.stack([np.asarray(row["embedding"], dtype=np.float32) for row in embedding_rows], axis=0).astype(np.float32)
    total_score = np.asarray([float(row["targets"]["total_score"]) for row in embedding_rows], dtype=np.float32)
    np.savez_compressed(
        path,
        protein_ids=protein_ids,
        frame_ids=frame_ids,
        embeddings=embeddings,
        total_score=total_score,
    )


def load_embedding_cache(
    path: Path,
    *,
    split_by_protein: dict[str, str],
    protein_id_filter: str | None,
    sample_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache does not exist: {path}")
    payload = np.load(path, allow_pickle=True)
    required = {"protein_ids", "frame_ids", "embeddings", "total_score"}
    missing = sorted(required.difference(payload.files))
    if missing:
        raise ValueError(f"Embedding cache is missing required arrays: {missing}")
    protein_ids = np.asarray(payload["protein_ids"]).astype(str)
    frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    total_score = np.asarray(payload["total_score"], dtype=np.float32)
    frame_count = int(protein_ids.shape[0])
    if not (
        frame_ids.shape[0] == frame_count
        and embeddings.shape[0] == frame_count
        and total_score.shape[0] == frame_count
    ):
        raise ValueError("Embedding cache arrays have inconsistent first dimensions.")

    selected_proteins = [protein_id for protein_id in dict.fromkeys(protein_ids.tolist()) if protein_id in split_by_protein]
    if protein_id_filter:
        selected_proteins = [protein_id for protein_id in selected_proteins if protein_id == protein_id_filter]
    if sample_limit is not None:
        selected_proteins = selected_proteins[: int(sample_limit)]
    selected = set(selected_proteins)
    if not selected:
        raise ValueError("No cached proteins matched manifest/split filters.")

    frame_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    for index in range(frame_count):
        protein_id = str(protein_ids[index])
        if protein_id not in selected:
            continue
        split = normalize_split(split_by_protein[protein_id])
        frame_id = int(frame_ids[index])
        targets = {"total_score": float(total_score[index])}
        frame_rows.append(
            {
                "protein_id": protein_id,
                "frame_id": frame_id,
                "split": split,
                **targets,
            }
        )
        embedding_rows.append(
            {
                "protein_id": protein_id,
                "frame_id": frame_id,
                "split": split,
                "embedding": embeddings[index],
                "targets": targets,
            }
        )
    return frame_rows, embedding_rows


def filter_embedding_rows_by_target(embedding_rows: list[dict[str, Any]], *, args: argparse.Namespace) -> dict[str, Any]:
    values = np.asarray([float(row["targets"]["total_score"]) for row in embedding_rows], dtype=np.float64)
    if values.size == 0:
        raise ValueError("No total_score values are available for filtering.")
    lower_bound = float(args.target_min) if args.target_min is not None else None
    upper_bound = float(args.target_max) if args.target_max is not None else None
    if args.target_quantile_low is not None:
        low_q = float(args.target_quantile_low)
        if not 0.0 <= low_q <= 1.0:
            raise ValueError(f"--target-quantile-low must be in [0, 1], got {low_q}.")
        quantile_value = float(np.quantile(values, low_q))
        lower_bound = quantile_value if lower_bound is None else max(lower_bound, quantile_value)
    if args.target_quantile_high is not None:
        high_q = float(args.target_quantile_high)
        if not 0.0 <= high_q <= 1.0:
            raise ValueError(f"--target-quantile-high must be in [0, 1], got {high_q}.")
        quantile_value = float(np.quantile(values, high_q))
        upper_bound = quantile_value if upper_bound is None else min(upper_bound, quantile_value)
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        raise ValueError(f"total_score filter lower bound {lower_bound} exceeds upper bound {upper_bound}.")

    keep_mask = np.ones(values.shape[0], dtype=bool)
    if lower_bound is not None:
        keep_mask &= values >= lower_bound
    if upper_bound is not None:
        keep_mask &= values <= upper_bound
    filtered_rows = [row for row, keep in zip(embedding_rows, keep_mask.tolist()) if keep]
    num_before = int(values.shape[0])
    num_after = int(len(filtered_rows))
    embedding_rows[:] = filtered_rows
    return {
        "target": "total_score",
        "target_min": None if args.target_min is None else float(args.target_min),
        "target_max": None if args.target_max is None else float(args.target_max),
        "target_quantile_low": None if args.target_quantile_low is None else float(args.target_quantile_low),
        "target_quantile_high": None if args.target_quantile_high is None else float(args.target_quantile_high),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "num_before": num_before,
        "num_after": num_after,
        "num_filtered": int(num_before - num_after),
    }


def rows_from_embedding_rows(embedding_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "protein_id": row["protein_id"],
            "frame_id": int(row["frame_id"]),
            "split": row["split"],
            "total_score": float(row["targets"]["total_score"]),
        }
        for row in embedding_rows
    ]


def run_probe_suite(
    *,
    embedding_rows: list[dict[str, Any]],
    probes: list[str],
    feature_transforms: list[int | None],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    splits = np.asarray([row["split"] for row in embedding_rows])
    embeddings = np.stack([row["embedding"] for row in embedding_rows], axis=0).astype(np.float32)
    protein_ids = np.asarray([row["protein_id"] for row in embedding_rows])
    frame_ids = np.asarray([row["frame_id"] for row in embedding_rows])

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Both train and test frames are required for probe evaluation.")

    results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    total_jobs = len(feature_transforms) * len(TARGETS) * len(probes)
    completed_jobs = 0
    print(
        "[probe] "
        f"frames={len(embedding_rows)} train={int(np.sum(train_mask))} "
        f"validation={int(np.sum(validation_mask))} test={int(np.sum(test_mask))} "
        f"jobs={total_jobs}",
        flush=True,
    )
    for feature_transform in feature_transforms:
        feature_label = feature_transform_label(feature_transform)
        print(f"[probe] fitting feature transform pca_components={feature_label}", flush=True)
        transformed = fit_transform_features(embeddings, train_mask=train_mask, pca_components=feature_transform)
        print(f"[probe] finished feature transform pca_components={feature_label} shape={transformed.shape}", flush=True)
        for target in TARGETS:
            y = np.asarray([row["targets"][target] for row in embedding_rows], dtype=np.float64)
            for probe in probes:
                completed_jobs += 1
                print(
                    f"[probe] start {completed_jobs}/{total_jobs} "
                    f"target={target} probe={probe} pca_components={feature_label}",
                    flush=True,
                )
                pred = fit_predict_probe(
                    probe=probe,
                    x=transformed,
                    y=y,
                    train_mask=train_mask,
                    validation_mask=validation_mask,
                    test_mask=test_mask,
                    args=args,
                )
                metrics = regression_metrics(y[test_mask], pred[test_mask])
                print(
                    f"[probe] done {completed_jobs}/{total_jobs} "
                    f"target={target} probe={probe} pca_components={feature_label} "
                    f"rmse={metrics['rmse']:.6g} spearman={metrics['spearman']:.6g} r2={metrics['r2']:.6g}",
                    flush=True,
                )
                results.append(
                    {
                        "target": target,
                        "probe": probe,
                        "pca_components": feature_label,
                        "n_train_frames": int(np.sum(train_mask)),
                        "n_validation_frames": int(np.sum(validation_mask)),
                        "n_test_frames": int(np.sum(test_mask)),
                        "n_train_proteins": int(len(set(protein_ids[train_mask]))),
                        "n_validation_proteins": int(len(set(protein_ids[validation_mask]))),
                        "n_test_proteins": int(len(set(protein_ids[test_mask]))),
                        **metrics,
                    }
                )
                for protein_id, frame_id, true_value, pred_value in zip(protein_ids[test_mask], frame_ids[test_mask], y[test_mask], pred[test_mask]):
                    prediction_rows.append(
                        {
                            "protein_id": protein_id,
                            "frame_id": int(frame_id),
                            "split": "test",
                            "target": target,
                            "probe": probe,
                            "pca_components": feature_label,
                            "true_value": float(true_value),
                            "pred_value": float(pred_value),
                        }
                    )
    return results, prediction_rows


def fit_transform_features(embeddings: np.ndarray, *, train_mask: np.ndarray, pca_components: int | None) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(embeddings[train_mask])
    x_all = scaler.transform(embeddings)
    if pca_components is None:
        return x_all.astype(np.float32)

    from sklearn.decomposition import PCA

    n_components = min(int(pca_components), int(x_train.shape[0]), int(x_train.shape[1]))
    if n_components <= 0:
        raise ValueError("PCA requires at least one component.")
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(x_train)
    return pca.transform(x_all).astype(np.float32)


def fit_predict_probe(
    *,
    probe: str,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if probe == "linear":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(x[train_mask], y[train_mask])
        return np.asarray(model.predict(x), dtype=np.float64)
    if probe == "svm":
        from sklearn.svm import LinearSVR

        model = LinearSVR(
            C=float(args.svm_c),
            epsilon=float(args.svm_epsilon),
            max_iter=int(args.svm_max_iter),
            random_state=int(args.seed),
        )
        model.fit(x[train_mask], y[train_mask])
        return np.asarray(model.predict(x), dtype=np.float64)
    if probe == "mlp":
        return fit_predict_mlp(x=x, y=y, train_mask=train_mask, validation_mask=validation_mask, args=args)
    raise ValueError(f"Unsupported probe: {probe}")


def fit_predict_mlp(*, x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, validation_mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    device = torch.device(_resolve_runtime_device(args.device or "cpu"))
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y[train_mask].reshape(-1, 1)).reshape(-1)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)
    hidden_dims = parse_int_list(args.mlp_hidden_dims)
    model = MLPRegressor(input_dim=x.shape[1], hidden_dims=hidden_dims, dropout=float(args.mlp_dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.mlp_lr), weight_decay=float(args.mlp_weight_decay))
    loss_fn = nn.MSELoss()
    train_dataset = TensorDataset(
        torch.as_tensor(x[train_mask], dtype=torch.float32),
        torch.as_tensor(y_scaled[train_mask], dtype=torch.float32).unsqueeze(-1),
    )
    train_loader = DataLoader(train_dataset, batch_size=int(args.mlp_batch_size), shuffle=True)
    validation_indices = np.where(validation_mask)[0]
    if validation_indices.size == 0:
        validation_indices = np.where(train_mask)[0]
    x_val = torch.as_tensor(x[validation_indices], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_scaled[validation_indices], dtype=torch.float32, device=device).unsqueeze(-1)
    y_val_true = y[validation_indices].astype(np.float64)
    best_state = None
    early_stop_metric = str(args.mlp_early_stop_metric)
    minimize_metric = early_stop_metric in {"rmse", "mae", "mse", "val_loss"}
    best_score = float("inf") if minimize_metric else -float("inf")
    stale_epochs = 0
    epochs = int(args.mlp_epochs)
    epoch_iter = _tqdm(
        range(epochs),
        total=epochs,
        desc="mlp epochs",
        leave=False,
        dynamic_ncols=True,
    )
    for epoch in epoch_iter:
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_x.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size
        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(x_val)
            val_loss = float(loss_fn(val_pred_scaled, y_val).item())
            val_pred = y_scaler.inverse_transform(val_pred_scaled.detach().cpu().numpy().reshape(-1, 1)).reshape(-1)
        if early_stop_metric == "val_loss":
            current_score = val_loss
        else:
            current_score = regression_metrics(y_val_true, val_pred)[early_stop_metric]
        improved = (
            np.isfinite(current_score)
            and (
                current_score < best_score - 1.0e-8
                if minimize_metric
                else current_score > best_score + 1.0e-8
            )
        )
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                train_loss=(train_loss_sum / max(1, train_count)),
                val_loss=val_loss,
                metric=early_stop_metric,
                score=current_score,
                best=best_score,
                stale=stale_epochs,
            )
        if improved:
            best_score = float(current_score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.mlp_patience):
                print(
                    f"[mlp] early_stop epoch={epoch + 1}/{epochs} "
                    f"metric={early_stop_metric} best={best_score:.6g} "
                    f"patience={int(args.mlp_patience)}",
                    flush=True,
                )
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.as_tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1).astype(np.float64)


class MLPRegressor:
    def __new__(cls, *, input_dim: int, hidden_dims: list[int], dropout: float) -> Any:
        import torch
        from torch import nn

        layers: list[Any] = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    ss_res = float(np.sum(diff * diff))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered * centered))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "r2": r2,
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(array, kind="mergesort")
    sorted_values = array[sorter]
    ranks = np.zeros(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(array.size)
    return ranks[inverse]


def load_split_map(split_file: Path) -> dict[str, str]:
    split_by_protein: dict[str, str] = {}
    with split_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            protein_id = (row.get("protein_id") or row.get("sample_id") or "").strip()
            split = normalize_split(row.get("split", ""))
            if protein_id:
                split_by_protein[protein_id] = split
    return split_by_protein


def normalize_split(raw_split: str) -> str:
    split = str(raw_split).strip().lower()
    if split in {"val", "valid", "validation"}:
        return "validation"
    if split in {"train", "test"}:
        return split
    raise ValueError(f"Unsupported split label: {raw_split!r}")


def parse_feature_transforms(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for item in str(raw).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"none", "raw", "no_pca", "nopca"}:
            values.append(None)
        else:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one PCA value or 'none'.")
    return values


def feature_transform_label(pca_components: int | None) -> str:
    return "none" if pca_components is None else str(int(pca_components))


def parse_probes(raw: str) -> list[str]:
    values: list[str] = []
    for item in str(raw).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "all":
            for probe in PROBES:
                if probe not in values:
                    values.append(probe)
            continue
        if token not in PROBES:
            raise ValueError(f"Unsupported probe: {item}. Choose from {', '.join(PROBES)} or all.")
        if token not in values:
            values.append(token)
    if not values:
        raise ValueError("At least one probe is required.")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _load_pretrain_weights_for_inference(trainer: Any, checkpoint_path: Path) -> None:
    import torch

    payload = torch.load(checkpoint_path, map_location=trainer.config.device, weights_only=False, mmap=True)
    backend_mode = payload.get("backend_mode")
    if backend_mode != trainer.backend_mode:
        raise ValueError(
            f"Checkpoint backend_mode={backend_mode!r} does not match trainer backend_mode={trainer.backend_mode!r}."
        )
    model_state = payload.get("model_state") or {}
    trainer.sequence_encoder.load_state_dict(model_state.get("sequence_encoder", {}))
    trainer.structure_encoder.load_state_dict(model_state.get("structure_encoder", {}))
    trainer.fusion_transformer.load_state_dict(model_state.get("fusion_transformer", {}))
    trainer.global_step = int(payload.get("global_step", 0))


def _freeze_for_inference(trainer: Any) -> None:
    for module in (trainer.sequence_encoder, trainer.structure_encoder, trainer.fusion_transformer):
        if hasattr(module, "eval"):
            module.eval()
        if hasattr(module, "parameters"):
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def _build_config(config_dict: Dict[str, Any]) -> PretrainTrainerConfig:
    manifest_path = _resolve_repo_path(config_dict.get("manifest_path"))
    if not manifest_path:
        raise ValueError("manifest_path is required in config or CLI arguments.")
    checkpoint_path = _resolve_repo_path(config_dict.get("checkpoint_path"))
    sequence_model_name = str(_resolve_config_alias(config_dict, primary_key="model_name", alias_key="sequence_model_name", default="esmc_600m"))
    embedding_dim = int(_resolve_config_alias(config_dict, primary_key="d_model", alias_key="embedding_dim", default=8))
    return PretrainTrainerConfig(
        manifest_path=str(manifest_path),
        batch_size_pretrain=int(config_dict.get("batch_size_pretrain", 1)),
        sequence_model_name=sequence_model_name,
        sequence_pooling=str(config_dict.get("sequence_pooling", "mean_pool")),
        structure_pooling=str(config_dict.get("structure_pooling", "mean_pool")),
        fusion_pooling=str(config_dict.get("fusion_pooling", "cls")),
        embedding_dim=embedding_dim,
        projection_dim=int(config_dict.get("projection_dim", 4)),
        lambda_align=float(config_dict.get("lambda_align", 0.1)),
        lambda_cons=float(config_dict.get("lambda_cons", 1.0)),
        lambda_recon=float(config_dict.get("lambda_recon", 0.2)),
        dyn_whole_modality_dropout_prob=float(config_dict.get("dyn_whole_modality_dropout_prob", 1.0)),
        consistency_mode=str(config_dict.get("consistency_mode", "cosine")),
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        max_residues=int(config_dict.get("max_residues", 100)),
        max_frames=int(config_dict.get("max_frames", 160)),
        backend_mode=str(config_dict.get("backend_mode", "real")),
        st_num_layers=int(config_dict.get("st_num_layers", 4)),
        st_num_heads=int(config_dict.get("st_num_heads", 8)),
        st_dropout=float(config_dict.get("st_dropout", 0.1)),
        fusion_num_layers=int(config_dict.get("fusion_num_layers", 2)),
        fusion_num_heads=int(config_dict.get("fusion_num_heads", 8)),
        fusion_dropout=float(config_dict.get("fusion_dropout", 0.1)),
        optimizer=str(config_dict.get("optimizer", "AdamW")),
        learning_rate=float(config_dict.get("learning_rate", 3e-5)),
        weight_decay=float(config_dict.get("weight_decay", 0.01)),
        grad_clip=float(config_dict.get("grad_clip", 1.0)),
        use_flash_attn=_parse_bool(config_dict.get("use_flash_attn"), default=True),
        sequence_encoder_trainable=False,
        device=_resolve_runtime_device(str(config_dict.get("device", "cpu"))),
        max_epochs_pretrain=int(config_dict.get("max_epochs_pretrain", 10)),
        validation_ratio_pretrain=float(config_dict.get("validation_ratio_pretrain", 0.0)),
        validation_seed_pretrain=int(config_dict.get("validation_seed_pretrain", 42)),
        validation_interval_pretrain=int(config_dict.get("validation_interval_pretrain", 1)),
        checkpoint_interval_pretrain=int(config_dict.get("checkpoint_interval_pretrain", 5)),
        show_progress_pretrain=False,
        progress_log_interval_pretrain=int(config_dict.get("progress_log_interval_pretrain", 50)),
        min_delta_pretrain=float(config_dict.get("min_delta_pretrain", 0.0)),
        tensorboard_log_dir=None,
    )


def _load_config_dict(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if importlib.util.find_spec("yaml") is not None:
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Top-level YAML config must be a mapping.")
        return dict(loaded)
    config: Dict[str, Any] = {}
    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            config[key.strip()] = _parse_scalar(value.strip())
    return config


def _resolve_repo_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(_repo_root() / path)


def _resolve_repo_path_required(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else _repo_root() / path


def _resolve_output_path(path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else _repo_root() / path_value


def _format_identity_label(value: float) -> str:
    percentage = value * 100.0
    if abs(percentage - round(percentage)) <= 1.0e-8:
        return str(int(round(percentage)))
    return str(percentage).replace(".", "p")


def _resolve_runtime_device(requested_device: str) -> str:
    normalized = requested_device.strip().lower()
    if not normalized.startswith("cuda"):
        return requested_device
    try:
        import torch
    except ImportError:
        return "cpu"
    return requested_device if torch.cuda.is_available() else "cpu"


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    lower_value = value.lower()
    if lower_value in {"true", "false"}:
        return lower_value == "true"
    try:
        if "." in value or "e" in lower_value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_bool(raw_value: Any, *, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_config_alias(config_dict: Dict[str, Any], *, primary_key: str, alias_key: str, default: Any) -> Any:
    primary_value = config_dict.get(primary_key)
    alias_value = config_dict.get(alias_key)
    if _has_value(primary_value) and _has_value(alias_value) and str(primary_value) != str(alias_value):
        raise ValueError(f"Conflicting config values for {primary_key!r}={primary_value!r} and {alias_key!r}={alias_value!r}.")
    if _has_value(primary_value):
        return primary_value
    if _has_value(alias_value):
        return alias_value
    return default


def _has_value(raw_value: Any) -> bool:
    return raw_value is not None and str(raw_value).strip() != ""


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return value


def _count_failures(failures: list[dict[str, Any]], error_type: str) -> int:
    return sum(1 for failure in failures if failure.get("error_type") == error_type)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
