#!/usr/bin/env python3
from __future__ import annotations

# RELEASE_IMPORT_BOOTSTRAP: allow running scripts directly from the repository root.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import importlib.util
import os
from pathlib import Path
import random
from typing import Any, Dict

from src.datasets.pretrain_dataset import PretrainDataset
from src.training.pretrain_trainer import (
    PretrainTrainerConfig,
    build_pretrain_trainer,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ProtMMLM pretrain.")
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
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional pretrain manifest override.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. If it exists, it will be loaded before training resumes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional pretrain mini-batch size override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_dict = _load_config_dict(args.config) if args.config is not None else {}
    for override_path in args.config_override or []:
        config_dict.update(_load_config_dict(override_path))
    if args.manifest_path is not None:
        config_dict["manifest_path"] = str(args.manifest_path)
    if args.batch_size is not None:
        config_dict["batch_size_pretrain"] = args.batch_size
    if args.checkpoint is not None:
        config_dict["checkpoint_path"] = str(args.checkpoint)

    config = _build_config(config_dict)
    dataset = PretrainDataset.from_manifest(
        config.manifest_path,
        sample_limit=None,
        full_only=True,
    )
    train_dataset, validation_dataset = _split_pretrain_dataset(
        dataset,
        validation_ratio=config.validation_ratio_pretrain,
        seed=config.validation_seed_pretrain,
    )
    trainer = build_pretrain_trainer(config)

    _print_summary("config", trainer.config_summary())
    _print_summary("dataset", dataset.summary())
    _print_summary("train_dataset", train_dataset.summary())
    if validation_dataset is not None:
        _print_summary("validation_dataset", validation_dataset.summary())
    else:
        print("validation_dataset.num_samples=0")
    for key, value in sorted(trainer.component_summary().items()):
        print(f"component.{key}={value}")
    print(f"backend_mode={trainer.backend_mode}")
    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
    print(f"checkpoint_path={checkpoint_path if checkpoint_path is not None else 'none'}")

    if checkpoint_path is not None and checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
        print(f"loaded_checkpoint={checkpoint_path}")

    fit_summary = trainer.fit(
        train_dataset,
        validation_dataset=validation_dataset,
        checkpoint_root=checkpoint_path,
    )
    _print_summary("train", fit_summary)
    return 0


def _build_config(config_dict: Dict[str, Any]) -> PretrainTrainerConfig:
    manifest_path = _resolve_repo_path(config_dict.get("manifest_path"))
    if not manifest_path:
        raise ValueError("manifest_path is required in config or CLI arguments.")

    checkpoint_path = _resolve_repo_path(config_dict.get("checkpoint_path"))
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
        sequence_encoder_trainable=_parse_bool(
            config_dict.get("sequence_encoder_trainable"),
            default=True,
        ),
        device=_resolve_runtime_device(str(config_dict.get("device", "cpu"))),
        max_epochs_pretrain=int(config_dict.get("max_epochs_pretrain", 10)),
        validation_ratio_pretrain=float(config_dict.get("validation_ratio_pretrain", 0.0)),
        validation_seed_pretrain=int(config_dict.get("validation_seed_pretrain", 42)),
        validation_interval_pretrain=int(config_dict.get("validation_interval_pretrain", 1)),
        checkpoint_interval_pretrain=int(config_dict.get("checkpoint_interval_pretrain", 5)),
        show_progress_pretrain=_parse_bool(
            config_dict.get("show_progress_pretrain"),
            default=True,
        ),
        progress_log_interval_pretrain=int(config_dict.get("progress_log_interval_pretrain", 50)),
        min_delta_pretrain=float(config_dict.get("min_delta_pretrain", 0.0)),
        tensorboard_log_dir=None
        if not _has_value(config_dict.get("tensorboard_log_dir"))
        else str(config_dict.get("tensorboard_log_dir")),
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

    # Minimal fallback parser for flat key: value YAML when PyYAML is unavailable.
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


def _split_pretrain_dataset(
    dataset: PretrainDataset,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[PretrainDataset, PretrainDataset | None]:
    if validation_ratio <= 0.0 or len(dataset) <= 1:
        return dataset, None
    if validation_ratio >= 1.0:
        raise ValueError(
            f"validation_ratio_pretrain must be in [0, 1), got {validation_ratio}."
        )

    validation_size = min(
        len(dataset) - 1,
        max(1, int(len(dataset) * validation_ratio)),
    )
    shuffled_indices = list(range(len(dataset)))
    random.Random(seed).shuffle(shuffled_indices)
    validation_indices = set(shuffled_indices[:validation_size])

    train_samples = [
        sample
        for index, sample in enumerate(dataset.samples)
        if index not in validation_indices
    ]
    validation_samples = [
        sample
        for index, sample in enumerate(dataset.samples)
        if index in validation_indices
    ]
    return PretrainDataset(train_samples), PretrainDataset(validation_samples)


def _parse_bool(raw_value: Any, *, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


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

def _has_value(raw_value: Any) -> bool:
    return raw_value is not None and str(raw_value).strip() != ""


if __name__ == "__main__":
    raise SystemExit(main())
