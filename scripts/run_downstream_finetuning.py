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
from dataclasses import asdict, replace
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List


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


def _resolve_repo_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(_repo_root() / path)


from src.datasets.downstream_adapters import load_downstream_samples, load_ppikb_samples
from src.datasets.downstream_dataset import DownstreamDataset, DownstreamStage1Sample, task_kind_for_name
from src.datasets.downstream_splits import split_downstream_dataset
from src.datasets.pretrain_dataset import PretrainDataset, build_pretrain_hash_index
from src.evaluation.evaluator import EvaluationRecord, ProtMMLMEvaluator
from src.training.downstream_trainer import (
    DownstreamTrainerConfig,
    build_downstream_trainer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ProtMMLM downstream fine-tuning.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--config-override",
        type=Path,
        action="append",
        default=None,
        help="Optional flat YAML override applied after --config. Can be passed multiple times.",
    )
    parser.add_argument(
        "--stage",
        choices=("downstream",),
        default=None,
        help="Single downstream training path.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Downstream task name.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Downstream manifest CSV path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. If it exists, it will be loaded before training resumes.",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Optional ProtMMLM pretrain checkpoint used to initialize downstream modules.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional tiny-subset sample limit override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training/evaluation batch size override.",
    )
    parser.add_argument(
        "--show-progress",
        dest="show_progress",
        action="store_true",
        default=None,
        help="Enable downstream tqdm/progress output.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable downstream tqdm/progress output.",
    )
    parser.add_argument(
        "--progress-log-interval",
        type=int,
        default=None,
        help="Fallback plain-text progress interval when tqdm is unavailable.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=None,
        help="Override the number of cross-validation folds.",
    )
    parser.add_argument(
        "--test-fold-index",
        type=int,
        default=None,
        help="Override the held-out test fold index.",
    )
    parser.add_argument(
        "--val-fold-index",
        type=int,
        default=None,
        help="Override the validation fold index.",
    )
    parser.add_argument(
        "--val-fold-offset",
        type=int,
        default=None,
        help="Override the validation fold offset when val-fold-index is unset.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional output subdirectory label, e.g. fold_00.",
    )
    parser.add_argument(
        "--data-source",
        choices=("manifest", "raw_official"),
        default=None,
        help="Optional downstream data source override. raw_official currently applies to PPIKB official run splits.",
    )
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=None,
        help="Optional raw official data root, e.g. datasets/downstream/PPIKB for PPIKB.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Optional validation fraction for manifest-based official-train random splits.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=None,
        help="Optional seed for manifest-based official-train random validation splits.",
    )
    parser.add_argument(
        "--manifest-validation-policy",
        type=str,
        default=None,
        help="Validation derivation policy for manifests with official train/test but no validation rows.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=None,
        help="Optional downstream split strategy override, e.g. prmftp_mmseqs_clustered.",
    )
    parser.add_argument(
        "--mmseqs-binary",
        type=str,
        default=None,
        help="Optional mmseqs executable path or name for clustered PrMFTP splitting.",
    )
    parser.add_argument(
        "--cluster-min-seq-id",
        type=float,
        default=None,
        help="Optional MMseqs minimum sequence identity for clustered splitting.",
    )
    parser.add_argument(
        "--cluster-coverage",
        type=float,
        default=None,
        help="Optional MMseqs coverage threshold for clustered splitting.",
    )
    parser.add_argument(
        "--cluster-cov-mode",
        type=int,
        default=None,
        help="Optional MMseqs coverage mode for clustered splitting.",
    )
    parser.add_argument(
        "--cluster-train-fraction",
        type=float,
        default=None,
        help="Optional train fraction for clustered splitting.",
    )
    parser.add_argument(
        "--cluster-validation-fraction",
        type=float,
        default=None,
        help="Optional validation fraction for clustered splitting.",
    )
    parser.add_argument(
        "--cluster-seed",
        type=int,
        default=None,
        help="Optional random seed for clustered splitting.",
    )
    parser.add_argument(
        "--cluster-sequence-field",
        type=str,
        default=None,
        help="Sequence field to cluster for generic MMseqs splits, e.g. sequence or peptide_sequence.",
    )
    parser.add_argument(
        "--cluster-balance-labels",
        dest="cluster_balance_labels",
        action="store_true",
        default=None,
        help="Try to preserve label balance while assigning MMseqs clusters to splits.",
    )
    parser.add_argument(
        "--cluster-shuffle-order",
        dest="cluster_shuffle_order",
        action="store_true",
        default=None,
        help="Shuffle MMseqs cluster assignment order with cluster_seed before assigning splits.",
    )
    parser.add_argument(
        "--no-cluster-shuffle-order",
        dest="cluster_shuffle_order",
        action="store_false",
        help="Use size-sorted MMseqs cluster assignment order with random tie-breaking only.",
    )
    parser.add_argument(
        "--no-cluster-balance-labels",
        dest="cluster_balance_labels",
        action="store_false",
        help="Disable label-aware MMseqs cluster split assignment.",
    )
    parser.add_argument(
        "--balance-binary-train-split",
        dest="balance_binary_train_split",
        action="store_true",
        default=None,
        help="Enable binary train-split class balancing after splitting.",
    )
    parser.add_argument(
        "--no-balance-binary-train-split",
        dest="balance_binary_train_split",
        action="store_false",
        help="Disable binary train-split class balancing after splitting.",
    )
    parser.add_argument(
        "--task-head-type",
        choices=("auto", "linear", "mlp"),
        default=None,
        help="Override downstream task head type.",
    )
    parser.add_argument(
        "--train-only-task-head",
        dest="train_only_task_head",
        action="store_true",
        default=None,
        help="Freeze all non-task-head modules and train only the downstream task head.",
    )
    parser.add_argument(
        "--no-train-only-task-head",
        dest="train_only_task_head",
        action="store_false",
        help="Disable task-head-only training.",
    )
    parser.add_argument(
        "--task-head-hidden-dims",
        type=str,
        default=None,
        help="Comma-separated hidden dims for MLP task heads, e.g. 512,128.",
    )
    parser.add_argument(
        "--task-head-dropout",
        type=float,
        default=None,
        help="Dropout for MLP task heads.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_dict = _load_config_dict(args.config) if args.config is not None else {}
    for override_path in args.config_override or []:
        config_dict.update(_load_config_dict(override_path))
    if args.task is not None:
        config_dict["task_name"] = args.task
    if args.manifest_path is not None:
        config_dict["manifest_path"] = str(args.manifest_path)
    if args.checkpoint is not None:
        config_dict["checkpoint_path"] = str(args.checkpoint)
    if args.pretrain_checkpoint is not None:
        config_dict["pretrain_checkpoint_path"] = str(args.pretrain_checkpoint)
    if args.sample_limit is not None:
        config_dict["sample_limit"] = args.sample_limit
    if args.batch_size is not None:
        config_dict["batch_size"] = args.batch_size
    if args.show_progress is not None:
        config_dict["show_progress_downstream"] = args.show_progress
    if args.progress_log_interval is not None:
        config_dict["progress_log_interval_downstream"] = args.progress_log_interval
    if args.num_folds is not None:
        config_dict["num_folds"] = args.num_folds
    if args.test_fold_index is not None:
        config_dict["test_fold_index"] = args.test_fold_index
    if args.val_fold_index is not None:
        config_dict["val_fold_index"] = args.val_fold_index
    if args.val_fold_offset is not None:
        config_dict["val_fold_offset"] = args.val_fold_offset
    if args.run_id is not None:
        config_dict["run_id"] = args.run_id
    if args.data_source is not None:
        config_dict["data_source"] = args.data_source
    if args.raw_data_path is not None:
        config_dict["raw_data_path"] = str(args.raw_data_path)
    if args.validation_fraction is not None:
        config_dict["validation_fraction"] = args.validation_fraction
    if args.validation_seed is not None:
        config_dict["validation_seed"] = args.validation_seed
    if args.manifest_validation_policy is not None:
        config_dict["manifest_validation_policy"] = args.manifest_validation_policy
    if args.split_strategy is not None:
        config_dict["split_strategy"] = args.split_strategy
    if args.mmseqs_binary is not None:
        config_dict["mmseqs_binary"] = args.mmseqs_binary
    if args.cluster_min_seq_id is not None:
        config_dict["cluster_min_seq_id"] = args.cluster_min_seq_id
    if args.cluster_coverage is not None:
        config_dict["cluster_coverage"] = args.cluster_coverage
    if args.cluster_cov_mode is not None:
        config_dict["cluster_cov_mode"] = args.cluster_cov_mode
    if args.cluster_train_fraction is not None:
        config_dict["cluster_train_fraction"] = args.cluster_train_fraction
    if args.cluster_validation_fraction is not None:
        config_dict["cluster_validation_fraction"] = args.cluster_validation_fraction
    if args.cluster_seed is not None:
        config_dict["cluster_seed"] = args.cluster_seed
    if args.cluster_sequence_field is not None:
        config_dict["cluster_sequence_field"] = args.cluster_sequence_field
    if args.cluster_balance_labels is not None:
        config_dict["cluster_balance_labels"] = args.cluster_balance_labels
    if args.cluster_shuffle_order is not None:
        config_dict["cluster_shuffle_order"] = args.cluster_shuffle_order
    if args.balance_binary_train_split is not None:
        config_dict["balance_binary_train_split"] = args.balance_binary_train_split
    if args.task_head_type is not None:
        config_dict["task_head_type"] = args.task_head_type
    if args.train_only_task_head is not None:
        config_dict["train_only_task_head"] = args.train_only_task_head
    if args.task_head_hidden_dims is not None:
        config_dict["task_head_hidden_dims"] = args.task_head_hidden_dims
    if args.task_head_dropout is not None:
        config_dict["task_head_dropout"] = args.task_head_dropout

    resolved_stage = str(args.stage or config_dict.get("stage", "downstream")).strip().lower()
    if resolved_stage not in {"downstream", "downstream"}:
        raise ValueError(
            f"Unsupported stage {resolved_stage!r}. "
            "Only the single downstream mixed-missing path is supported."
        )

    config = _build_downstream_config(config_dict)
    enabled_control_modes = [
        name
        for name, enabled in (
            ("sequence_only_baseline", config.sequence_only_baseline),
            ("force_missing_dynamics_for_all_samples", config.force_missing_dynamics_for_all_samples),
            ("force_natural_structure_only_for_all_samples", config.force_natural_structure_only_for_all_samples),
        )
        if enabled
    ]
    if len(enabled_control_modes) > 1:
        raise ValueError(
            "Control modes cannot be combined: "
            f"{', '.join(enabled_control_modes)}. "
            "Use sequence_only_baseline for the pure sequence baseline, "
            "force_missing_dynamics_for_all_samples for the missing-dynamics/mask-token control, "
            "or force_natural_structure_only_for_all_samples for the natural-structure-only control."
        )
    trainer = build_downstream_trainer(config)

    dataset = _load_downstream_dataset(config, config_dict)
    split_bundle = split_downstream_dataset(
        dataset,
        num_folds=int(config_dict.get("num_folds", 5)),
        test_fold_index=int(config_dict.get("test_fold_index", 0)),
        val_fold_index=_optional_int(config_dict.get("val_fold_index")),
        val_fold_offset=int(config_dict.get("val_fold_offset", 1)),
        validation_fraction=float(config_dict.get("validation_fraction", 0.1)),
        validation_seed=_optional_int(config_dict.get("validation_seed")),
        manifest_validation_policy=str(config_dict.get("manifest_validation_policy", "random_grouped_9_1")),
        split_strategy=_optional_cached_value(config_dict.get("split_strategy")),
        mmseqs_binary=str(config_dict.get("mmseqs_binary", "mmseqs")),
        cluster_min_seq_id=float(config_dict.get("cluster_min_seq_id", 0.4)),
        cluster_coverage=float(config_dict.get("cluster_coverage", 0.8)),
        cluster_cov_mode=int(config_dict.get("cluster_cov_mode", 0)),
        cluster_train_fraction=float(config_dict.get("cluster_train_fraction", 0.8)),
        cluster_validation_fraction=float(config_dict.get("cluster_validation_fraction", 0.1)),
        cluster_seed=_optional_int(config_dict.get("cluster_seed")),
        cluster_sequence_field=_optional_cached_value(config_dict.get("cluster_sequence_field")),
        cluster_balance_labels=_parse_bool(config_dict.get("cluster_balance_labels"), default=False),
        cluster_shuffle_order=_parse_bool(config_dict.get("cluster_shuffle_order"), default=False),
        balance_binary_train_split=_parse_bool(config_dict.get("balance_binary_train_split"), default=True),
    )
    training_split_bundle, target_scaler = _maybe_normalize_regression_targets(
        split_bundle,
        task_name=config.task_name,
        config_dict=config_dict,
    )
    checkpoint_path = _resolve_downstream_checkpoint_path(
        task_name=config.task_name,
        checkpoint_path=config.checkpoint_path,
        run_id=config_dict.get("run_id"),
    )
    split_paths = _write_split_csvs(split_bundle, checkpoint_root=checkpoint_path)

    _print_summary("config", trainer.config_summary())
    _print_summary("dataset", dataset.summary())
    _print_summary("split", split_bundle["summary"])
    _print_summary("split_files", {key: str(value) for key, value in split_paths.items()})
    _print_summary("train_dataset", split_bundle["train"].summary())
    _print_summary("val_dataset", split_bundle["validation"].summary())
    _print_summary("test_dataset", split_bundle["test"].summary())
    if hasattr(trainer, "component_summary"):
        for key, value in sorted(trainer.component_summary().items()):
            print(f"component.{key}={value}")
    print("stage=downstream")
    print(f"backend_mode={getattr(trainer, 'backend_mode', 'real')}")
    print(f"checkpoint_path={checkpoint_path}")
    if config_dict.get("run_id") is not None and str(config_dict.get("run_id")).strip():
        print(f"run_id={str(config_dict.get('run_id')).strip()}")
    if config.pretrain_checkpoint_path:
        print(f"pretrain_checkpoint_path={config.pretrain_checkpoint_path}")
    dataset_summary = dataset.summary()
    mixed_full_samples = int(dataset_summary["num_full_samples"])
    mixed_partial_samples = int(dataset_summary.get("num_partial_samples", 0))
    print("mixed_batch_mode=true")
    print(f"mixed_batch.full_samples={mixed_full_samples}")
    print(f"mixed_batch.partial_samples={mixed_partial_samples}")
    print(f"mixed_batch.seq_only_samples={int(dataset_summary['num_seq_only_samples'])}")
    if target_scaler is not None:
        _print_summary("target_normalization", target_scaler)

    resume_checkpoint_path = _resolve_resume_checkpoint_path(checkpoint_path)
    initialization_mode = "fresh_random_init"
    if resume_checkpoint_path is not None and config.pretrain_checkpoint_path:
        raise ValueError(
            "Both a downstream resume checkpoint and a pretrain checkpoint are available. "
            "Remove or rename the downstream checkpoint to start a fresh run from pretrain weights, "
            "or omit --pretrain-checkpoint to resume downstream training."
        )
    if resume_checkpoint_path is not None:
        initialization_mode = "resume_downstream_checkpoint"
        trainer.load_checkpoint(resume_checkpoint_path)
        print(f"loaded_checkpoint={resume_checkpoint_path}")
    elif config.pretrain_checkpoint_path:
        initialization_mode = "initialize_from_pretrain"
        pretrain_load_summary = trainer.load_pretrain_checkpoint(config.pretrain_checkpoint_path)
        print(f"loaded_pretrain_checkpoint={config.pretrain_checkpoint_path}")
        _print_summary("pretrain_init", pretrain_load_summary.get("load_summary", {}))
    print(f"initialization_mode={initialization_mode}")

    fit_summary = trainer.fit(
        training_split_bundle["train"],
        validation_dataset=training_split_bundle["validation"],
        test_dataset=training_split_bundle["test"],
        checkpoint_root=checkpoint_path,
    )
    _print_summary("train", fit_summary)

    best_checkpoint_path, best_checkpoint_kind = _resolve_best_checkpoint_path(
        fit_summary,
        checkpoint_root=checkpoint_path,
    )
    trainer.load_checkpoint(best_checkpoint_path)
    print(f"loaded_best_checkpoint={best_checkpoint_path}")
    print(f"loaded_best_checkpoint_kind={best_checkpoint_kind}")

    validation_report = trainer.evaluate_dataset(training_split_bundle["validation"])
    test_report = trainer.evaluate_dataset(training_split_bundle["test"])
    test_records: List[EvaluationRecord]
    if target_scaler is not None:
        original_validation_report = _evaluate_original_scale_regression(
            trainer,
            normalized_dataset=training_split_bundle["validation"],
            raw_dataset=split_bundle["validation"],
            target_scaler=target_scaler,
        )
        original_test_records = _predict_original_scale_regression(
            trainer,
            normalized_dataset=training_split_bundle["test"],
            raw_dataset=split_bundle["test"],
            target_scaler=target_scaler,
        )
        original_scale_subset_names = None
        if getattr(trainer.config, "sequence_only_baseline", False):
            original_scale_subset_names = ("seq_only",)
        original_test_report = ProtMMLMEvaluator(task_name=split_bundle["test"].task_name).evaluate(
            original_test_records,
            subset_names=original_scale_subset_names,
        )
        _print_report("val_best_norm", validation_report)
        _print_report("test_norm", test_report)
        _print_report("val_best", original_validation_report)
        _print_report("test", original_test_report)
        validation_report = original_validation_report
        test_report = original_test_report
        test_records = original_test_records
    else:
        _print_report("val_best", validation_report)
        _print_report("test", test_report)
        test_records = trainer.predict_dataset(training_split_bundle["test"])
    _write_per_label_csv("val_best", validation_report, checkpoint_root=checkpoint_path)
    _write_per_label_csv("test", test_report, checkpoint_root=checkpoint_path)
    misclassification_summary = _write_binary_misclassification_exports(
        checkpoint_root=checkpoint_path,
        prefix="test",
        test_records=test_records,
        test_dataset=split_bundle["test"],
    )
    if misclassification_summary.get("enabled"):
        _print_summary("test_misclassifications", misclassification_summary)
    final_metrics_path = _write_final_metrics_json(
        checkpoint_root=checkpoint_path,
        fit_summary=fit_summary,
        validation_report=validation_report,
        test_report=test_report,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_kind=best_checkpoint_kind,
    )
    print(f"final_metrics_json={final_metrics_path}")
    test_results_path = _write_test_results_json(
        checkpoint_root=checkpoint_path,
        test_report=test_report,
        test_records=test_records,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_kind=best_checkpoint_kind,
        fit_summary=fit_summary,
        misclassification_summary=misclassification_summary,
    )
    print(f"test_results_json={test_results_path}")
    return 0


def _default_downstream_data_root(task_name: str) -> str:
    normalized = str(task_name).strip().lower().replace("-", "_")
    if normalized == "toxteller":
        return "./datasets/downstream/Toxteller"
    if normalized == "conotoxin":
        return "./datasets/downstream/Conotoxin"
    if normalized == "prmftp":
        return "./datasets/downstream/PrMFTP"
    if normalized == "ppikb":
        return "./datasets/downstream/PPIKB"
    raise ValueError(f"Unsupported downstream task {task_name!r}.")


def _default_pretrain_data_root() -> str:
    return "./datasets/pretrain"


def _load_downstream_dataset(
    config: DownstreamTrainerConfig,
    config_dict: Dict[str, Any],
) -> DownstreamDataset:
    manifest_path = _resolve_repo_path(config_dict.get("manifest_path"))
    data_source = str(config_dict.get("data_source") or "").strip().lower()
    if manifest_path and data_source in {"", "manifest"}:
        return DownstreamDataset.from_manifest(
            manifest_path,
            task_name=config.task_name,
            sample_limit=config.sample_limit,
        )

    raw_data_path = _resolve_repo_path(
        config_dict.get("raw_data_path", _default_downstream_data_root(config.task_name))
    )
    if not raw_data_path:
        raise ValueError("raw_data_path is required for downstream loading.")
    pretrain_data_root = _resolve_repo_path(
        config_dict.get("pretrain_data_root", _default_pretrain_data_root())
    )
    if not pretrain_data_root:
        raise ValueError("pretrain_data_root is required for downstream loading.")

    if config.task_name == "ppikb":
        run_id = config_dict.get("run_id")
        if run_id is None or not str(run_id).strip():
            raise ValueError("run_id is required for PPIKB downstream loading.")
        run_dir = Path(raw_data_path) / str(run_id).strip()
        if not run_dir.exists():
            run_dir = Path(raw_data_path) / f"run_{str(run_id).strip()}"
        samples = load_ppikb_samples(run_dir, sample_limit=config.sample_limit)
    else:
        samples = load_downstream_samples(config.task_name, raw_data_path, sample_limit=config.sample_limit)

    pretrain_dataset = PretrainDataset.from_dataset_root(pretrain_data_root)
    pretrain_index = build_pretrain_hash_index(pretrain_dataset)
    return DownstreamDataset.from_samples_with_pretrain_index(
        samples,
        task_name=config.task_name,
        pretrain_index=pretrain_index,
    )


def _build_downstream_config(config_dict: Dict[str, Any]) -> DownstreamTrainerConfig:
    manifest_path = _resolve_repo_path(config_dict.get("manifest_path"))
    task_name = config_dict.get("task_name")
    if not task_name:
        raise ValueError("task_name is required in config or CLI arguments.")
    if not manifest_path:
        manifest_path = str(Path(_resolve_repo_path(config_dict.get("raw_data_path", _default_downstream_data_root(task_name))) or ""))

    checkpoint_path = _resolve_repo_path(config_dict.get("checkpoint_path"))
    pretrain_checkpoint_path = _resolve_repo_path(config_dict.get("pretrain_checkpoint_path"))
    sequence_model_name = str(
        _resolve_config_alias(
            config_dict,
            primary_key="model_name",
            alias_key="sequence_model_name",
            default="esmc_600m",
        )
    )
    embedding_dim = int(
        _resolve_config_alias(
            config_dict,
            primary_key="d_model",
            alias_key="embedding_dim",
            default=8,
        )
    )
    return DownstreamTrainerConfig(
        manifest_path=str(manifest_path),
        task_name=str(task_name),
        sample_limit=_optional_int(config_dict.get("sample_limit")),
        batch_size=int(config_dict.get("batch_size", config_dict.get("batch_size_downstream", 1))),
        sequence_model_name=sequence_model_name,
        sequence_pooling=str(config_dict.get("sequence_pooling", "mean_pool")),
        structure_pooling=str(config_dict.get("structure_pooling", "mean_pool")),
        fusion_pooling=str(config_dict.get("fusion_pooling", "cls")),
        embedding_dim=embedding_dim,
        projection_dim=int(config_dict.get("projection_dim", 4)),
        lambda_align=float(config_dict.get("lambda_align", 0.1)),
        lambda_cons=float(config_dict.get("lambda_cons", 1.0)),
        lambda_recon=float(config_dict.get("lambda_recon", 0.2)),
        consistency_mode=str(config_dict.get("consistency_mode", "cosine")),
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        pretrain_checkpoint_path=None if pretrain_checkpoint_path is None else str(pretrain_checkpoint_path),
        backend_mode=str(config_dict.get("backend_mode", "real")),
        regression_loss_mode=str(config_dict.get("regression_loss_mode", "huber")),
        regression_delta=float(config_dict.get("regression_delta", 1.0)),
        multilabel_pos_weight_mode=str(config_dict.get("multilabel_pos_weight_mode", "none")),
        multilabel_max_pos_weight=float(config_dict.get("multilabel_max_pos_weight", 20.0)),
        max_sequence_length=int(
            config_dict.get(
                "max_sequence_length",
                config_dict.get("protein_max_sequence_length", config_dict.get("max_residues", 100)),
            )
        ),
        protein_max_sequence_length=int(
            config_dict.get(
                "protein_max_sequence_length",
                config_dict.get("max_sequence_length", config_dict.get("max_residues", 100)),
            )
        ),
        peptide_max_sequence_length=int(config_dict.get("peptide_max_sequence_length", 100)),
        max_residues=int(config_dict.get("max_residues", 100)),
        protein_max_residues=int(config_dict.get("protein_max_residues", config_dict.get("max_residues", 100))),
        peptide_max_residues=int(config_dict.get("peptide_max_residues", config_dict.get("max_residues", 100))),
        max_frames=int(config_dict.get("max_frames", 160)),
        max_epochs_downstream=int(
            _resolve_config_alias(
                config_dict,
                primary_key="max_epochs",
                alias_key="max_epochs_downstream",
                default=30,
            )
        ),
        min_epochs_downstream=int(
            _resolve_config_alias(
                config_dict,
                primary_key="min_epochs",
                alias_key="min_epochs_downstream",
                default=5,
            )
        ),
        patience_downstream=int(
            _resolve_config_alias(
                config_dict,
                primary_key="patience",
                alias_key="patience_downstream",
                default=5,
            )
        ),
        min_delta_downstream=float(
            _resolve_config_alias(
                config_dict,
                primary_key="min_delta",
                alias_key="min_delta_downstream",
                default=0.0,
            )
        ),
        validation_interval_downstream=int(
            _resolve_config_alias(
                config_dict,
                primary_key="validation_interval",
                alias_key="validation_interval_downstream",
                default=1,
            )
        ),
        seq_guard_tolerance=float(config_dict.get("seq_guard_tolerance", 0.0)),
        st_num_layers=int(config_dict.get("st_num_layers", 4)),
        st_num_heads=int(config_dict.get("st_num_heads", 8)),
        st_dropout=float(config_dict.get("st_dropout", 0.1)),
        fusion_num_layers=int(config_dict.get("fusion_num_layers", 2)),
        fusion_num_heads=int(config_dict.get("fusion_num_heads", 8)),
        fusion_dropout=float(config_dict.get("fusion_dropout", 0.1)),
        optimizer=str(config_dict.get("optimizer", "AdamW")),
        learning_rate=float(config_dict.get("learning_rate", 2e-5)),
        weight_decay=float(config_dict.get("weight_decay", 0.01)),
        grad_clip=float(config_dict.get("grad_clip", 1.0)),
        use_flash_attn=_parse_bool(config_dict.get("use_flash_attn"), default=True),
        sequence_encoder_trainable=_parse_bool(
            config_dict.get("sequence_encoder_trainable"),
            default=True,
        ),
        structure_encoder_trainable=_parse_bool(
            config_dict.get("structure_encoder_trainable"),
            default=True,
        ),
        fusion_transformer_trainable=_parse_bool(
            config_dict.get("fusion_transformer_trainable"),
            default=True,
        ),
        show_progress_downstream=_parse_bool(
            config_dict.get("show_progress_downstream", config_dict.get("show_progress")),
            default=True,
        ),
        progress_log_interval_downstream=int(
            config_dict.get(
                "progress_log_interval_downstream",
                config_dict.get("progress_log_interval", 50),
            )
        ),
        gradient_accumulation_steps=int(config_dict.get("gradient_accumulation_steps", 1)),
        sequence_only_baseline=_parse_bool(config_dict.get("sequence_only_baseline"), default=False),
        force_missing_dynamics_for_all_samples=_parse_bool(
            config_dict.get("force_missing_dynamics_for_all_samples"),
            default=False,
        ),
        force_natural_structure_only_for_all_samples=_parse_bool(
            config_dict.get("force_natural_structure_only_for_all_samples"),
            default=False,
        ),
        mixed_task_mode=str(config_dict.get("mixed_task_mode", "coverage_aware")),
        partial_pair_handling=str(config_dict.get("partial_pair_handling", "seq_fallback")),
        aux_loss_reweight_mode=str(config_dict.get("aux_loss_reweight_mode", "batch_coverage")),
        min_full_fraction_for_aux=float(config_dict.get("min_full_fraction_for_aux", 0.1)),
        task_head_type=str(config_dict.get("task_head_type", "auto")),
        train_only_task_head=_parse_bool(config_dict.get("train_only_task_head"), default=False),
        task_head_hidden_dims=_parse_int_tuple(config_dict.get("task_head_hidden_dims"), default=(512, 128)),
        task_head_dropout=float(config_dict.get("task_head_dropout", 0.1)),
        save_best_guarded_checkpoint=_parse_bool(config_dict.get("save_best_guarded_checkpoint"), default=True),
        single_missing_task_fallback=str(config_dict.get("single_missing_task_fallback", "fused_missing")),
        single_task_feature_mode=str(config_dict.get("single_task_feature_mode", "legacy")),
        monitor_subset=_optional_cached_value(config_dict.get("monitor_subset")),
        monitor_name=_optional_cached_value(config_dict.get("monitor_name")),
        manifest_validation_policy=str(config_dict.get("manifest_validation_policy", "random_grouped_9_1")),
        shuffle_train_each_epoch=_parse_bool(config_dict.get("shuffle_train_each_epoch"), default=True),
        train_shuffle_seed=int(config_dict.get("train_shuffle_seed", 17)),
        single_full_sample_oversample_factor=int(config_dict.get("single_full_sample_oversample_factor", 1)),
        device=_resolve_runtime_device(str(config_dict.get("device", "cpu"))),
    )


def _resolve_runtime_device(requested_device: str) -> str:
    normalized = requested_device.strip().lower()
    if not normalized.startswith("cuda"):
        return requested_device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return requested_device
    return "cpu"


def _resolve_downstream_checkpoint_path(
    *,
    task_name: str,
    checkpoint_path: str | None,
    run_id: Any,
) -> Path:
    base_path = Path(checkpoint_path) if checkpoint_path else Path("outputs") / "downstream" / task_name / "downstream.ckpt"
    run_id_value = "" if run_id is None else str(run_id).strip()
    if not run_id_value:
        return base_path
    resolved_dir = base_path.parent if base_path.suffix else base_path
    target_dir = resolved_dir / run_id_value
    if base_path.suffix:
        return target_dir / base_path.name
    return target_dir


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
                raise ValueError(
                    f"Fallback YAML parser only supports 'key: value' lines. "
                    f"Invalid line at {config_path}:{line_number}"
                )
            key, value = raw_line.split(":", 1)
            config[key.strip()] = _parse_scalar(value.strip())
    return config


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


def _print_summary(prefix: str, values: Dict[str, Any]) -> None:
    for key in sorted(values):
        print(f"{prefix}.{key}={values[key]}")


def _print_report(prefix: str, report: Dict[str, Any]) -> None:
    summary = {
        "task_name": report["task_name"],
        "task_kind": report["task_kind"],
    }
    for subset_name in _ordered_report_subsets(report):
        summary[f"{subset_name}_num_samples"] = report[subset_name]["num_samples"]
    matched_full = report.get("matched_full")
    if matched_full is not None:
        summary["matched_full_num_samples"] = matched_full["num_samples"]
    _print_summary(prefix, summary)
    for subset_name in _ordered_report_subsets(report):
        metrics = report[subset_name]["metrics"]
        for metric_name, value in sorted(metrics.items()):
            print(f"{prefix}.{subset_name}.{metric_name}={value}")
    if matched_full is None:
        return
    for variant_name in ("seq_only", "full"):
        metrics = matched_full[variant_name]["metrics"]
        for metric_name, value in sorted(metrics.items()):
            print(f"{prefix}.matched_full.{variant_name}.{metric_name}={value}")
    for metric_name, value in sorted(matched_full["delta_full_minus_seq"].items()):
        print(f"{prefix}.matched_full.delta_full_minus_seq.{metric_name}={value}")


def _ordered_report_subsets(report: Dict[str, Any]) -> list[str]:
    return [
        subset_name
        for subset_name in ("overall", "seq_only", "nature_only", "partial", "full")
        if subset_name in report
    ]


def _write_per_label_csv(prefix: str, report: Dict[str, Any], *, checkpoint_root: Path) -> None:
    rows = list(report.get("per_label_metrics") or [])
    if not rows:
        return
    output_dir = checkpoint_root.parent if checkpoint_root.suffix else checkpoint_root
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_per_label_metrics.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{prefix}.per_label_csv={output_path}")


def _write_final_metrics_json(
    *,
    checkpoint_root: Path,
    fit_summary: Dict[str, Any],
    validation_report: Dict[str, Any],
    test_report: Dict[str, Any],
    best_checkpoint_path: Path,
    best_checkpoint_kind: str,
) -> Path:
    output_dir = checkpoint_root.parent if checkpoint_root.suffix else checkpoint_root
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_metrics.json"
    payload = {
        "train": fit_summary,
        "val_best": _with_protocol_aliases(validation_report, fit_summary=fit_summary),
        "test": _with_protocol_aliases(test_report, fit_summary=fit_summary),
        "loaded_best_checkpoint": str(best_checkpoint_path),
        "loaded_best_checkpoint_kind": best_checkpoint_kind,
        "protocol_view": _protocol_view(fit_summary=fit_summary),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _write_split_csvs(split_bundle: Dict[str, Any], *, checkpoint_root: Path) -> Dict[str, Path]:
    output_dir = checkpoint_root.parent if checkpoint_root.suffix else checkpoint_root
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for subset_name in ("train", "validation", "test"):
        dataset = split_bundle[subset_name]
        output_path = split_dir / f"{subset_name}.csv"
        _write_split_dataset_csv(output_path, dataset, subset_name=subset_name)
        paths[subset_name] = output_path

    summary_path = split_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value"])
        writer.writeheader()
        for key, value in sorted(dict(split_bundle["summary"]).items()):
            writer.writerow({"key": key, "value": _format_csv_value(value)})
    paths["summary"] = summary_path
    return paths


def _write_binary_misclassification_exports(
    *,
    checkpoint_root: Path,
    prefix: str,
    test_records: List[EvaluationRecord],
    test_dataset: DownstreamDataset,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if test_dataset.task_kind != "binary":
        return {"enabled": False, "reason": f"task_kind={test_dataset.task_kind}"}

    output_dir = checkpoint_root.parent if checkpoint_root.suffix else checkpoint_root
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_by_id = {sample.sample_id: sample for sample in test_dataset}
    false_positives: list[Dict[str, Any]] = []
    false_negatives: list[Dict[str, Any]] = []

    for record in test_records:
        sample = samples_by_id.get(record.sample_id)
        true_label = int(float(record.label) >= threshold)
        prediction = float(record.prediction)
        predicted_label = int(prediction >= threshold)
        if true_label == predicted_label:
            continue
        error_type = (
            "false_positive_pred_toxic_label_nontoxic"
            if predicted_label == 1
            else "false_negative_pred_nontoxic_label_toxic"
        )
        row = {
            "error_type": error_type,
            "sample_id": record.sample_id,
            "label": record.label,
            "predicted_label": predicted_label,
            "prediction": prediction,
            "threshold": threshold,
            "sequence": "" if sample is None else sample.sequence,
            "sequence_hash": "" if sample is None else sample.sequence_hash,
            "raw_label": "" if sample is None else sample.raw_label,
            "task_name": record.task_name,
            "task_kind": test_dataset.task_kind,
            "manifest_split": "" if sample is None else sample.split or "",
            "matched_pretrain_id": "" if sample is None else sample.matched_pretrain_id or "",
            "nature_path": "" if sample is None else sample.nature_path or "",
            "md_path": "" if sample is None else sample.md_path or "",
            "has_dyn": "" if sample is None else sample.has_dyn,
            "modality_subset": record.modality_subset,
            "seq_only_prediction": "" if record.seq_only_prediction is None else record.seq_only_prediction,
            "full_prediction": "" if record.full_prediction is None else record.full_prediction,
        }
        if predicted_label == 1:
            false_positives.append(row)
        else:
            false_negatives.append(row)

    fieldnames = [
        "error_type",
        "sample_id",
        "label",
        "predicted_label",
        "prediction",
        "threshold",
        "sequence",
        "sequence_hash",
        "raw_label",
        "task_name",
        "task_kind",
        "manifest_split",
        "matched_pretrain_id",
        "nature_path",
        "md_path",
        "has_dyn",
        "modality_subset",
        "seq_only_prediction",
        "full_prediction",
    ]
    false_positive_path = output_dir / f"{prefix}_false_positives.csv"
    false_negative_path = output_dir / f"{prefix}_false_negatives.csv"
    for output_path, rows in (
        (false_positive_path, false_positives),
        (false_negative_path, false_negatives),
    ):
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    json_path = output_dir / f"{prefix}_misclassifications.json"
    payload = {
        "enabled": True,
        "task_name": test_dataset.task_name,
        "task_kind": test_dataset.task_kind,
        "threshold": threshold,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives_csv": str(false_positive_path),
        "false_negatives_csv": str(false_negative_path),
        "misclassifications_json": str(json_path),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "enabled": True,
        "threshold": threshold,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives_csv": str(false_positive_path),
        "false_negatives_csv": str(false_negative_path),
        "misclassifications_json": str(json_path),
    }


def _write_test_results_json(
    *,
    checkpoint_root: Path,
    test_report: Dict[str, Any],
    test_records: List[EvaluationRecord],
    best_checkpoint_path: Path,
    best_checkpoint_kind: str,
    fit_summary: Dict[str, Any],
    misclassification_summary: Dict[str, Any] | None = None,
) -> Path:
    output_dir = checkpoint_root.parent if checkpoint_root.suffix else checkpoint_root
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_results.json"
    payload = {
        "task_name": test_report["task_name"],
        "task_kind": test_report["task_kind"],
        "loaded_best_checkpoint": str(best_checkpoint_path),
        "loaded_best_checkpoint_kind": best_checkpoint_kind,
        "protocol_view": _protocol_view(fit_summary=fit_summary),
        "report": _with_protocol_aliases(test_report, fit_summary=fit_summary),
        "misclassification_exports": misclassification_summary,
        "records": [asdict(record) for record in test_records],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _experiment_name(*, fit_summary: Dict[str, Any]) -> str:
    if bool(fit_summary.get("is_seq_only_baseline")):
        return "seq-only_baseline"
    monitor_subset = str(fit_summary.get("monitor_subset") or "").strip().lower()
    if monitor_subset == "nature_only":
        return "protmmlm_nature_only"
    return "protmmlm"


def _protocol_view(*, fit_summary: Dict[str, Any]) -> Dict[str, str]:
    monitor_subset = str(fit_summary.get("monitor_subset") or "").strip().lower()
    is_seq_only_baseline = bool(fit_summary.get("is_seq_only_baseline"))
    return {
        "experiment_name": _experiment_name(fit_summary=fit_summary),
        "primary_comparison_target": monitor_subset or "overall",
        "is_seq_only_baseline": str(is_seq_only_baseline).lower(),
        "uses_pretrain_checkpoint": str(not is_seq_only_baseline).lower(),
    }


def _with_protocol_aliases(report: Dict[str, Any], *, fit_summary: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(report))


def _write_split_dataset_csv(output_path: Path, dataset: DownstreamDataset, *, subset_name: str) -> None:
    fieldnames = [
        "subset",
        "sample_id",
        "sequence",
        "sequence_hash",
        "peptide_sequence",
        "peptide_sequence_hash",
        "pair_key",
        "target",
        "raw_label",
        "task_name",
        "task_kind",
        "manifest_split",
        "matched_pretrain_id",
        "nature_path",
        "md_path",
        "has_dyn",
        "peptide_matched_pretrain_id",
        "peptide_nature_path",
        "peptide_md_path",
        "peptide_has_dyn",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in dataset:
            writer.writerow(
                {
                    "subset": subset_name,
                    "sample_id": sample.sample_id,
                    "sequence": sample.sequence,
                    "sequence_hash": sample.sequence_hash,
                    "peptide_sequence": sample.peptide_sequence or "",
                    "peptide_sequence_hash": sample.peptide_sequence_hash or "",
                    "pair_key": sample.pair_key or "",
                    "target": _format_csv_value(sample.target),
                    "raw_label": sample.raw_label,
                    "task_name": sample.task_name,
                    "task_kind": sample.task_kind,
                    "manifest_split": sample.split or "",
                    "matched_pretrain_id": sample.matched_pretrain_id or "",
                    "nature_path": sample.nature_path or "",
                    "md_path": sample.md_path or "",
                    "has_dyn": sample.has_dyn,
                    "peptide_matched_pretrain_id": sample.peptide_matched_pretrain_id or "",
                    "peptide_nature_path": sample.peptide_nature_path or "",
                    "peptide_md_path": sample.peptide_md_path or "",
                    "peptide_has_dyn": sample.peptide_has_dyn,
                }
            )


def _format_csv_value(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(component) for component in value)
    if isinstance(value, tuple):
        return "|".join(str(component) for component in value)
    return str(value)


def _maybe_normalize_regression_targets(
    split_bundle: Dict[str, Any],
    *,
    task_name: str,
    config_dict: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    method = _resolve_target_normalization_method(
        task_name=task_name,
        raw_value=config_dict.get("normalize_regression_targets"),
    )
    if method is None:
        return split_bundle, None
    if method != "standard":
        raise ValueError(f"Unsupported normalize_regression_targets={method!r}. Expected 'standard'.")

    train_dataset = split_bundle["train"]
    train_targets = [float(sample.target) for sample in train_dataset]
    if not train_targets:
        raise ValueError("Cannot normalize regression targets without training samples.")

    mean_value = sum(train_targets) / float(len(train_targets))
    variance = sum((target - mean_value) ** 2 for target in train_targets) / float(len(train_targets))
    scale_value = math.sqrt(variance)
    if scale_value == 0.0:
        scale_value = 1.0

    target_scaler = {
        "method": method,
        "mean": mean_value,
        "scale": scale_value,
        "train_samples": len(train_targets),
    }

    normalized_bundle = dict(split_bundle)
    for subset_name in ("train", "validation", "test"):
        normalized_bundle[subset_name] = _normalize_dataset_targets(
            split_bundle[subset_name],
            target_scaler=target_scaler,
        )
    return normalized_bundle, target_scaler


def _resolve_target_normalization_method(*, task_name: str, raw_value: Any) -> str | None:
    normalized_task = str(task_name).strip().lower().replace("-", "_")
    if raw_value is None:
        return "standard" if normalized_task == "ppikb" else None

    value = str(raw_value).strip().lower()
    if value in {"", "0", "false", "no", "none", "off"}:
        return None
    if value in {"1", "true", "yes", "on", "standard"}:
        return "standard"
    raise ValueError(
        "normalize_regression_targets must be one of true/false/standard/none, "
        f"got {raw_value!r}."
    )


def _normalize_dataset_targets(
    dataset: DownstreamDataset,
    *,
    target_scaler: Dict[str, Any],
) -> DownstreamDataset:
    mean_value = float(target_scaler["mean"])
    scale_value = float(target_scaler["scale"])
    normalized_samples = [
        replace(sample, target=(float(sample.target) - mean_value) / scale_value)
        for sample in dataset
    ]
    return DownstreamDataset(
        normalized_samples,
        task_name=dataset.task_name,
        task_kind=dataset.task_kind,
    )


def _evaluate_original_scale_regression(
    trainer: Any,
    *,
    normalized_dataset: DownstreamDataset,
    raw_dataset: DownstreamDataset,
    target_scaler: Dict[str, Any],
) -> Dict[str, Any]:
    original_records = _predict_original_scale_regression(
        trainer,
        normalized_dataset=normalized_dataset,
        raw_dataset=raw_dataset,
        target_scaler=target_scaler,
    )
    original_scale_subset_names = None
    if getattr(trainer.config, "sequence_only_baseline", False):
        original_scale_subset_names = ("seq_only",)
    return ProtMMLMEvaluator(task_name=raw_dataset.task_name).evaluate(
        original_records,
        subset_names=original_scale_subset_names,
    )


def _predict_original_scale_regression(
    trainer: Any,
    *,
    normalized_dataset: DownstreamDataset,
    raw_dataset: DownstreamDataset,
    target_scaler: Dict[str, Any],
) -> List[EvaluationRecord]:
    raw_by_sample_id = {sample.sample_id: sample for sample in raw_dataset}
    original_records: List[EvaluationRecord] = []
    for record in trainer.predict_dataset(normalized_dataset):
        raw_sample = raw_by_sample_id.get(record.sample_id)
        if raw_sample is None:
            raise ValueError(f"Missing raw sample for normalized prediction record {record.sample_id!r}.")
        original_records.append(
            EvaluationRecord(
                sample_id=record.sample_id,
                label=float(raw_sample.target),
                prediction=_denormalize_scalar(record.prediction, target_scaler=target_scaler),
                task_name=record.task_name,
                has_dyn=record.has_dyn,
                modality_subset=record.modality_subset,
                seq_only_prediction=_denormalize_optional_scalar(
                    record.seq_only_prediction,
                    target_scaler=target_scaler,
                ),
                full_prediction=_denormalize_optional_scalar(
                    record.full_prediction,
                    target_scaler=target_scaler,
                ),
            )
        )
    return original_records


def _denormalize_optional_scalar(raw_value: Any, *, target_scaler: Dict[str, Any]) -> float | None:
    if raw_value is None:
        return None
    return _denormalize_scalar(raw_value, target_scaler=target_scaler)


def _denormalize_scalar(raw_value: Any, *, target_scaler: Dict[str, Any]) -> float:
    return float(raw_value) * float(target_scaler["scale"]) + float(target_scaler["mean"])


def _resolve_config_alias(
    config_dict: Dict[str, Any],
    *,
    primary_key: str,
    alias_key: str,
    default: Any,
) -> Any:
    primary_value = config_dict.get(primary_key)
    alias_value = config_dict.get(alias_key)
    if _has_value(primary_value) and _has_value(alias_value) and str(primary_value) != str(alias_value):
        raise ValueError(
            f"Conflicting config values for {primary_key!r}={primary_value!r} and "
            f"{alias_key!r}={alias_value!r}."
        )
    if _has_value(primary_value):
        return primary_value
    if _has_value(alias_value):
        return alias_value
    return default



def _resolve_best_checkpoint_path(
    fit_summary: Dict[str, Any],
    *,
    checkpoint_root: Path,
) -> tuple[Path, str]:
    checkpoint_paths = dict(fit_summary.get("checkpoint_paths", {}))
    best_overall_path = checkpoint_paths.get("best_overall")
    if best_overall_path:
        return Path(best_overall_path), "best_overall"

    best_path = checkpoint_paths.get("best")
    if best_path:
        return Path(best_path), "best"

    best_guarded_path = checkpoint_paths.get("best_guarded")
    if best_guarded_path:
        return Path(best_guarded_path), "best_guarded"

    resolved_dir = checkpoint_root.parent if checkpoint_root.suffix else checkpoint_root
    overall_fallback_path = resolved_dir / "downstream_best_overall.ckpt"
    if overall_fallback_path.exists():
        return overall_fallback_path, "best_overall_fallback"

    best_fallback_path = resolved_dir / "downstream_best.ckpt"
    if best_fallback_path.exists():
        return best_fallback_path, "best_fallback"

    guarded_fallback_path = resolved_dir / "downstream_best_guarded.ckpt"
    if guarded_fallback_path.exists():
        return guarded_fallback_path, "best_guarded_fallback"

    raise FileNotFoundError(
        "Best validation checkpoint was not found after training. "
        f"Expected {overall_fallback_path}, {best_fallback_path}, or {guarded_fallback_path}."
    )


def _resolve_resume_checkpoint_path(checkpoint_root: Path) -> Path | None:
    if checkpoint_root.exists():
        return checkpoint_root
    return None


def _parse_bool(raw_value: Any, *, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_int(raw_value: Any) -> int | None:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    return int(raw_value)


def _parse_int_tuple(raw_value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw_value is None or str(raw_value).strip() == "":
        return default
    if isinstance(raw_value, (list, tuple)):
        values = tuple(int(value) for value in raw_value)
    else:
        text = str(raw_value).strip().strip("[]()")
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("task_head_hidden_dims must contain at least one integer.")
    return values


def _optional_cached_value(raw_value: Any) -> str | None:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    return str(raw_value).strip()


def _has_value(raw_value: Any) -> bool:
    return raw_value is not None and str(raw_value).strip() != ""


if __name__ == "__main__":
    raise SystemExit(main())
