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
import time
from typing import Any, Dict

import numpy as np

from src.analysis.distance_correlation import (
    compare_distance_matrices,
    pairwise_embedding_distance,
)
from src.analysis.md_rmsd import pairwise_rmsd_matrix
from src.analysis.trajectory_embeddings import extract_trajectory_embeddings
from src.datasets.pretrain_dataset import PretrainDataset, PretrainSample
from src.datasets.structure_io import load_md_ca_frames
from src.training.pretrain_trainer import (
    PretrainTrainerConfig,
    build_pretrain_trainer,
)


SAMPLE_METRIC_FIELDS = [
    "sample_index",
    "protein_id",
    "sequence_hash",
    "md_path",
    "num_frames",
    "num_valid_rmsd_pairs",
    "num_invalid_rmsd_pairs",
    "mean_rmsd",
    "median_rmsd",
    "std_rmsd",
    "st_distance_metric",
    "st_pearson",
    "st_spearman",
    "st_mantel_r",
    "st_mantel_p",
    "st_embedding_dim",
    "fusion_distance_metric",
    "fusion_pearson",
    "fusion_spearman",
    "fusion_mantel_r",
    "fusion_mantel_p",
    "fusion_embedding_dim",
    "fusion_minus_st_pearson",
    "fusion_minus_st_spearman",
    "st_quantile_mean_distances",
    "fusion_quantile_mean_distances",
    "quantile_pair_counts",
    "st_rank_histogram",
    "fusion_rank_histogram",
    "rank_histogram_bins",
    "status",
    "error_message",
    "elapsed_seconds",
]

FAILURE_FIELDS = [
    "sample_index",
    "protein_id",
    "md_path",
    "stage",
    "error_type",
    "error_message",
]


def _normalize_thread_env() -> None:
    for variable_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        raw_value = os.environ.get(variable_name)
        if raw_value is None:
            continue
        stripped_value = raw_value.strip()
        if stripped_value.isdigit() and int(stripped_value) > 0:
            os.environ[variable_name] = stripped_value
            continue
        os.environ[variable_name] = "1"


_normalize_thread_env()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze agreement between MD frame RMSD and frozen ProtMMLM embedding geometry."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML pretrain config path.")
    parser.add_argument("--config-override", type=Path, action="append", default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--protein-id", type=str, default=None)
    parser.add_argument("--max-residues", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-common-residues", type=int, default=3)
    parser.add_argument("--embedding-distance-metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--frame-embedding-mode", choices=("token_aggregate", "per_frame"), default="token_aggregate")
    parser.add_argument("--fusion-embedding-source", choices=("fused_pooled", "fused_cls"), default="fused_pooled")
    parser.add_argument("--mantel-permutations", type=int, default=0)
    parser.add_argument("--mantel-method", choices=("pearson", "spearman"), default="spearman")
    parser.add_argument("--save-matrices", action="store_true")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--quantile-bins", type=int, default=10, help="Number of RMSD quantile bins for summary fields.")
    parser.add_argument("--rank-histogram-bins", type=int, default=50, help="2D bins for pooled RMSD-rank vs embedding-rank summary fields.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--failure-policy", choices=("skip", "raise"), default="skip")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = _resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_matrices:
        (output_dir / "matrices").mkdir(parents=True, exist_ok=True)
    if args.save_embeddings:
        (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    config_dict = _load_config_dict(args.config) if args.config is not None else {}
    for override_path in args.config_override or []:
        config_dict.update(_load_config_dict(override_path))
    if args.manifest_path is not None:
        config_dict["manifest_path"] = str(args.manifest_path)
    if args.checkpoint is not None:
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

    dataset = PretrainDataset.from_manifest(
        config.manifest_path,
        sample_limit=args.sample_limit,
        full_only=True,
    )
    samples = list(dataset.samples)
    if args.protein_id:
        samples = [sample for sample in samples if sample.protein_id == args.protein_id]
    if args.sample_limit is not None:
        samples = samples[: args.sample_limit]
    if not samples:
        raise ValueError("No samples matched the requested manifest/sample filters.")

    trainer = build_pretrain_trainer(config)
    _load_pretrain_weights_for_inference(trainer, checkpoint_path)
    trainer.sequence_encoder.eval()
    trainer.structure_encoder.eval()
    trainer.fusion_transformer.eval()

    resolved_config = {
        "config": trainer.config_summary(),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_requested_samples": len(samples),
        "analysis": {
            "min_common_residues": args.min_common_residues,
            "embedding_distance_metric": args.embedding_distance_metric,
            "frame_embedding_mode": args.frame_embedding_mode,
            "fusion_embedding_source": args.fusion_embedding_source,
            "mantel_permutations": args.mantel_permutations,
            "mantel_method": args.mantel_method,
            "save_matrices": args.save_matrices,
            "save_embeddings": args.save_embeddings,
            "quantile_bins": args.quantile_bins,
            "rank_histogram_bins": args.rank_histogram_bins,
        },
    }
    _write_json(output_dir / "resolved_config.json", resolved_config)

    completed_ids = _read_completed_ids(output_dir / "sample_metrics.csv") if args.resume else set()
    sample_metrics_path = output_dir / "sample_metrics.csv"
    failures_path = output_dir / "failures.csv"
    progress_path = output_dir / "progress.jsonl"
    _ensure_csv_header(sample_metrics_path, SAMPLE_METRIC_FIELDS)
    _ensure_csv_header(failures_path, FAILURE_FIELDS)

    import torch

    successful_rows: list[dict[str, Any]] = []
    failed_count = 0
    with torch.inference_mode():
        for sample_index, sample in enumerate(samples):
            if args.resume and sample.protein_id in completed_ids:
                continue
            start_time = time.perf_counter()
            try:
                row = analyze_one_sample(
                    sample=sample,
                    sample_index=sample_index,
                    trainer=trainer,
                    args=args,
                    config=config,
                    output_dir=output_dir,
                )
                row["elapsed_seconds"] = time.perf_counter() - start_time
                _append_csv_row(sample_metrics_path, SAMPLE_METRIC_FIELDS, row)
                successful_rows.append(row)
                _append_jsonl(progress_path, {"event": "sample_success", "protein_id": sample.protein_id, "sample_index": sample_index})
            except Exception as exc:  # noqa: BLE001 - long-running analysis should log sample-level failures.
                failed_count += 1
                if args.failure_policy == "raise":
                    raise
                failure_row = {
                    "sample_index": sample_index,
                    "protein_id": sample.protein_id,
                    "md_path": sample.md_path,
                    "stage": "analyze_one_sample",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                _append_csv_row(failures_path, FAILURE_FIELDS, failure_row)
                _append_jsonl(progress_path, {"event": "sample_failure", **failure_row})
            if args.log_interval > 0 and (sample_index + 1) % args.log_interval == 0:
                print(f"processed={sample_index + 1}/{len(samples)} successes={len(successful_rows)} failures={failed_count}", flush=True)

    all_rows = _read_metric_rows(sample_metrics_path)
    summary = summarize_rows(all_rows, total_samples=len(samples), failed_count=failed_count)
    _write_json(output_dir / "cohort_summary.json", summary)
    _write_summary_metrics_csv(output_dir / "summary_metrics.csv", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def analyze_one_sample(
    *,
    sample: PretrainSample,
    sample_index: int,
    trainer: Any,
    args: argparse.Namespace,
    config: PretrainTrainerConfig,
    output_dir: Path,
) -> dict[str, Any]:
    frames = load_md_ca_frames(
        sample.md_path,
        max_residues=config.max_residues,
        max_frames=config.max_frames,
    )
    if len(frames) < 3:
        raise ValueError(f"Need at least 3 MD frames, got {len(frames)}.")

    rmsd_matrix, valid_pair_mask = pairwise_rmsd_matrix(
        frames,
        min_common_residues=args.min_common_residues,
    )
    embeddings = extract_trajectory_embeddings(
        sample=sample,
        frames=frames,
        sequence_encoder=trainer.sequence_encoder,
        structure_encoder=trainer.structure_encoder,
        fusion_transformer=trainer.fusion_transformer,
        frame_embedding_mode=args.frame_embedding_mode,
        fusion_embedding_source=args.fusion_embedding_source,
    )
    st_distance_matrix = pairwise_embedding_distance(
        embeddings.st_embeddings,
        metric=args.embedding_distance_metric,
    )
    fusion_distance_matrix = pairwise_embedding_distance(
        embeddings.fusion_embeddings,
        metric=args.embedding_distance_metric,
    )
    st_metrics = compare_distance_matrices(
        rmsd_matrix,
        st_distance_matrix,
        valid_pair_mask=valid_pair_mask,
        mantel_permutations=args.mantel_permutations,
        mantel_method=args.mantel_method,
        random_seed=sample_index,
    )
    fusion_metrics = compare_distance_matrices(
        rmsd_matrix,
        fusion_distance_matrix,
        valid_pair_mask=valid_pair_mask,
        mantel_permutations=args.mantel_permutations,
        mantel_method=args.mantel_method,
        random_seed=sample_index + 7919,
    )

    if args.save_matrices:
        np.savez_compressed(
            output_dir / "matrices" / f"{_safe_name(sample.protein_id)}.npz",
            rmsd_matrix=rmsd_matrix,
            st_distance_matrix=st_distance_matrix,
            fusion_distance_matrix=fusion_distance_matrix,
            valid_pair_mask=valid_pair_mask,
            frame_indices=embeddings.frame_indices,
        )
    if args.save_embeddings:
        np.savez_compressed(
            output_dir / "embeddings" / f"{_safe_name(sample.protein_id)}.npz",
            st_embeddings=embeddings.st_embeddings,
            fusion_embeddings=embeddings.fusion_embeddings,
            sequence_embedding=embeddings.sequence_embedding,
            frame_indices=embeddings.frame_indices,
        )

    upper_mask = np.triu(valid_pair_mask, k=1) & np.isfinite(rmsd_matrix)
    rmsd_values = rmsd_matrix[upper_mask]
    return {
        "sample_index": sample_index,
        "protein_id": sample.protein_id,
        "sequence_hash": sample.sequence_hash,
        "md_path": sample.md_path,
        "num_frames": len(frames),
        "num_valid_rmsd_pairs": int(rmsd_values.size),
        "num_invalid_rmsd_pairs": int(len(frames) * (len(frames) - 1) // 2 - rmsd_values.size),
        "mean_rmsd": _nan_float(np.mean(rmsd_values) if rmsd_values.size else np.nan),
        "median_rmsd": _nan_float(np.median(rmsd_values) if rmsd_values.size else np.nan),
        "std_rmsd": _nan_float(np.std(rmsd_values) if rmsd_values.size else np.nan),
        "st_distance_metric": args.embedding_distance_metric,
        "st_pearson": _nan_float(st_metrics.pearson),
        "st_spearman": _nan_float(st_metrics.spearman),
        "st_mantel_r": _nan_float(st_metrics.mantel_r),
        "st_mantel_p": _nan_float(st_metrics.mantel_p),
        "st_embedding_dim": int(embeddings.st_embeddings.shape[1]),
        "fusion_distance_metric": args.embedding_distance_metric,
        "fusion_pearson": _nan_float(fusion_metrics.pearson),
        "fusion_spearman": _nan_float(fusion_metrics.spearman),
        "fusion_mantel_r": _nan_float(fusion_metrics.mantel_r),
        "fusion_mantel_p": _nan_float(fusion_metrics.mantel_p),
        "fusion_embedding_dim": int(embeddings.fusion_embeddings.shape[1]),
        "fusion_minus_st_pearson": _nan_float(fusion_metrics.pearson - st_metrics.pearson),
        "fusion_minus_st_spearman": _nan_float(fusion_metrics.spearman - st_metrics.spearman),
        **_quantile_trend_fields(
            rmsd_matrix,
            st_distance_matrix,
            fusion_distance_matrix,
            valid_pair_mask=valid_pair_mask,
            num_bins=args.quantile_bins,
        ),
        **_rank_histogram_fields(
            rmsd_matrix,
            st_distance_matrix,
            fusion_distance_matrix,
            valid_pair_mask=valid_pair_mask,
            num_bins=args.rank_histogram_bins,
        ),
        "status": "success",
        "error_message": "",
        "elapsed_seconds": 0.0,
    }


def summarize_rows(rows: list[dict[str, Any]], *, total_samples: int, failed_count: int) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("status") == "success"]
    return {
        "num_samples_total": total_samples,
        "num_samples_success": len(valid_rows),
        "num_samples_failed": failed_count,
        "st": {
            "pearson": _summarize_metric(valid_rows, "st_pearson"),
            "spearman": _summarize_metric(valid_rows, "st_spearman"),
        },
        "fusion": {
            "pearson": _summarize_metric(valid_rows, "fusion_pearson"),
            "spearman": _summarize_metric(valid_rows, "fusion_spearman"),
        },
        "comparison": {
            "fusion_minus_st_pearson": _summarize_metric(valid_rows, "fusion_minus_st_pearson"),
            "fusion_minus_st_spearman": _summarize_metric(valid_rows, "fusion_minus_st_spearman"),
            "num_fusion_better_spearman": sum(
                1 for row in valid_rows if _finite_float(row.get("fusion_minus_st_spearman")) > 0.0
            ),
            "fraction_fusion_better_spearman": _safe_fraction(
                sum(1 for row in valid_rows if _finite_float(row.get("fusion_minus_st_spearman")) > 0.0),
                len(valid_rows),
            ),
        },
    }


def _summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([_finite_float(row.get(key)) for row in rows], dtype=np.float64)
    weights = np.asarray([_finite_float(row.get("num_valid_rmsd_pairs")) for row in rows], dtype=np.float64)
    valid_mask = np.isfinite(values)
    values = values[valid_mask]
    weights = weights[valid_mask]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "iqr": None, "weighted_mean_by_pairs": None, "fisher_z_weighted": None}
    positive_weights = weights > 0.0
    weighted_mean = None
    fisher = None
    if np.any(positive_weights):
        weighted_mean = float(np.average(values[positive_weights], weights=weights[positive_weights]))
        clipped = np.clip(values[positive_weights], -0.999999, 0.999999)
        fisher = float(np.tanh(np.average(np.arctanh(clipped), weights=weights[positive_weights])))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "weighted_mean_by_pairs": weighted_mean,
        "fisher_z_weighted": fisher,
    }


def _quantile_trend_fields(
    rmsd_matrix: np.ndarray,
    st_distance_matrix: np.ndarray,
    fusion_distance_matrix: np.ndarray,
    *,
    valid_pair_mask: np.ndarray,
    num_bins: int,
) -> dict[str, str]:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}.")
    upper_mask = np.triu(valid_pair_mask, k=1)
    upper_mask &= np.isfinite(rmsd_matrix) & np.isfinite(st_distance_matrix) & np.isfinite(fusion_distance_matrix)
    rmsd_values = rmsd_matrix[upper_mask]
    st_values = st_distance_matrix[upper_mask]
    fusion_values = fusion_distance_matrix[upper_mask]
    if rmsd_values.size == 0:
        nan_values = [float("nan")] * num_bins
        zero_counts = [0] * num_bins
        return {
            "st_quantile_mean_distances": json.dumps(nan_values),
            "fusion_quantile_mean_distances": json.dumps(nan_values),
            "quantile_pair_counts": json.dumps(zero_counts),
        }

    order = np.argsort(rmsd_values, kind="mergesort")
    split_indices = np.array_split(order, num_bins)
    st_means: list[float] = []
    fusion_means: list[float] = []
    counts: list[int] = []
    for indices in split_indices:
        counts.append(int(indices.size))
        if indices.size == 0:
            st_means.append(float("nan"))
            fusion_means.append(float("nan"))
            continue
        st_means.append(float(np.mean(st_values[indices])))
        fusion_means.append(float(np.mean(fusion_values[indices])))
    return {
        "st_quantile_mean_distances": json.dumps(st_means),
        "fusion_quantile_mean_distances": json.dumps(fusion_means),
        "quantile_pair_counts": json.dumps(counts),
    }


def _rank_histogram_fields(
    rmsd_matrix: np.ndarray,
    st_distance_matrix: np.ndarray,
    fusion_distance_matrix: np.ndarray,
    *,
    valid_pair_mask: np.ndarray,
    num_bins: int,
) -> dict[str, str]:
    if num_bins <= 1:
        raise ValueError(f"num_bins must be greater than 1, got {num_bins}.")
    upper_mask = np.triu(valid_pair_mask, k=1)
    upper_mask &= np.isfinite(rmsd_matrix) & np.isfinite(st_distance_matrix) & np.isfinite(fusion_distance_matrix)
    rmsd_values = rmsd_matrix[upper_mask]
    st_values = st_distance_matrix[upper_mask]
    fusion_values = fusion_distance_matrix[upper_mask]
    if rmsd_values.size < 2:
        empty_histogram = [[0] * num_bins for _ in range(num_bins)]
        return {
            "st_rank_histogram": json.dumps(empty_histogram),
            "fusion_rank_histogram": json.dumps(empty_histogram),
            "rank_histogram_bins": num_bins,
        }

    rmsd_ranks = _fractional_ranks(rmsd_values)
    st_ranks = _fractional_ranks(st_values)
    fusion_ranks = _fractional_ranks(fusion_values)
    st_histogram = _rank_histogram2d(rmsd_ranks, st_ranks, num_bins=num_bins)
    fusion_histogram = _rank_histogram2d(rmsd_ranks, fusion_ranks, num_bins=num_bins)
    return {
        "st_rank_histogram": json.dumps(st_histogram.astype(int).tolist()),
        "fusion_rank_histogram": json.dumps(fusion_histogram.astype(int).tolist()),
        "rank_histogram_bins": num_bins,
    }


def _fractional_ranks(values: np.ndarray) -> np.ndarray:
    ranks = _rankdata(values)
    if ranks.size <= 1:
        return np.zeros_like(ranks, dtype=np.float64)
    return (ranks - 1.0) / float(ranks.size - 1)


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(array, kind="mergesort")
    inverse = np.empty_like(sorter)
    inverse[sorter] = np.arange(array.size)
    sorted_values = array[sorter]
    ranks = np.zeros(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks[inverse]


def _rank_histogram2d(x_values: np.ndarray, y_values: np.ndarray, *, num_bins: int) -> np.ndarray:
    x_bins = np.minimum(np.floor(np.clip(x_values, 0.0, 1.0) * num_bins).astype(int), num_bins - 1)
    y_bins = np.minimum(np.floor(np.clip(y_values, 0.0, 1.0) * num_bins).astype(int), num_bins - 1)
    histogram = np.zeros((num_bins, num_bins), dtype=np.int64)
    np.add.at(histogram, (y_bins, x_bins), 1)
    return histogram



def _write_summary_metrics_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = _flatten_summary_metrics(summary)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in rows:
            writer.writerow({"metric": metric, "value": _csv_value(value)})


def _flatten_summary_metrics(summary: dict[str, Any]) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in sorted(value.items()):
                visit(f"{prefix}.{child_key}" if prefix else child_key, child_value)
            return
        if isinstance(value, list):
            return
        rows.append((prefix, value))

    visit("", summary)
    return rows


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
    epoch = payload.get("epoch")
    if epoch is not None:
        trainer.current_epoch = int(epoch)


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
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in raw_line:
                raise ValueError(f"Fallback YAML parser only supports 'key: value' lines. Invalid line at {config_path}:{line_number}")
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


def _resolve_output_path(path_value: Path) -> Path:
    if path_value.is_absolute():
        return path_value
    return _repo_root() / path_value


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
        if "." in value:
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


def _ensure_csv_header(path: Path, fields: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def _append_csv_row(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "success" and row.get("protein_id"):
                completed.add(str(row["protein_id"]))
    return completed


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return value


def _nan_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
