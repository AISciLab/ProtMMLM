#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "toxteller": {
        "config": PROJECT_ROOT / "configs" / "downstream" / "toxteller.yaml",
        "manifest": PROJECT_ROOT / "examples" / "toxteller" / "toxteller_manifest.csv",
    },
    "prmftp": {
        "config": PROJECT_ROOT / "configs" / "downstream" / "prmftp.yaml",
        "manifest": PROJECT_ROOT / "examples" / "prmftp" / "prmftp_manifest.csv",
    },
    "ppikb": {
        "config": PROJECT_ROOT / "configs" / "downstream" / "ppikb.yaml",
        "manifest": PROJECT_ROOT / "examples" / "ppikb" / "ppikb_manifest.csv",
    },
    "conotoxin": {
        "config": PROJECT_ROOT / "configs" / "downstream" / "conotoxin.yaml",
        "manifest": PROJECT_ROOT / "examples" / "conotoxin" / "conotoxin_manifest.csv",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the included one-fold downstream ProtMMLM examples."
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASKS) + ("all",),
        default="all",
        help="Example task to run.",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pretrain" / "best.pth",
        help="ProtMMLM pretrain checkpoint used for downstream initialization.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch scripts/run_downstream_finetuning.py.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override downstream max_epochs for a short verification run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override downstream batch size.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Use only the first N manifest samples for a smoke test.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Override runtime device. Use cpu for dependency and data-flow checks.",
    )
    parser.add_argument(
        "--run-id-prefix",
        default="examples",
        help="Prefix used for output folders under outputs/downstream/<task>/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_names = list(TASKS) if args.task == "all" else [args.task]

    _check_required_files(task_names, args.pretrain_checkpoint)

    for task_name in task_names:
        command = _build_command(
            python_executable=args.python,
            task_name=task_name,
            pretrain_checkpoint=args.pretrain_checkpoint,
            run_id_prefix=args.run_id_prefix,
            batch_size=args.batch_size,
            sample_limit=args.sample_limit,
            override_path=_write_override_file(
                task_name=task_name,
                max_epochs=args.max_epochs,
                device=args.device,
            ),
        )
        print(_format_command(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    return 0


def _check_required_files(task_names: list[str], pretrain_checkpoint: Path) -> None:
    required_paths = [
        PROJECT_ROOT / "checkpoints" / "esmc-600m" / "data" / "weights" / "esmc_600m_2024_12_v0.pth",
        pretrain_checkpoint,
    ]
    for task_name in task_names:
        required_paths.extend([TASKS[task_name]["config"], TASKS[task_name]["manifest"]])

    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Required local assets are missing. Download/place them before running.\n"
            f"{formatted}"
        )


def _build_command(
    *,
    python_executable: str,
    task_name: str,
    pretrain_checkpoint: Path,
    run_id_prefix: str,
    batch_size: int | None,
    sample_limit: int | None,
    override_path: Path | None,
) -> list[str]:
    task = TASKS[task_name]
    command = [
        python_executable,
        "scripts/run_downstream_finetuning.py",
        "--config",
        str(task["config"].relative_to(PROJECT_ROOT)),
        "--manifest-path",
        str(task["manifest"].relative_to(PROJECT_ROOT)),
        "--pretrain-checkpoint",
        str(pretrain_checkpoint),
        "--run-id",
        f"{run_id_prefix}_{task_name}_from_pretrained",
    ]
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    if sample_limit is not None:
        command.extend(["--sample-limit", str(sample_limit)])
    if override_path is not None:
        command.extend(["--config-override", str(override_path.relative_to(PROJECT_ROOT))])
    return command


def _write_override_file(
    *,
    task_name: str,
    max_epochs: int | None,
    device: str | None,
) -> Path | None:
    overrides: dict[str, Any] = {}
    if max_epochs is not None:
        overrides["max_epochs"] = max_epochs
        overrides["min_epochs"] = min(max_epochs, 1)
        overrides["patience"] = max(max_epochs, 1)
    if device is not None:
        overrides["device"] = device

    if not overrides:
        return None

    override_dir = PROJECT_ROOT / "outputs" / "downstream_example_overrides"
    override_dir.mkdir(parents=True, exist_ok=True)
    override_path = override_dir / f"{task_name}_override.yaml"
    lines = [f"{key}: {_format_yaml_scalar(value)}\n" for key, value in overrides.items()]
    override_path.write_text("".join(lines), encoding="utf-8")
    return override_path


def _format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_command(command: list[str]) -> str:
    return " ".join(_quote_if_needed(part) for part in command)


def _quote_if_needed(part: str) -> str:
    if not part or any(character.isspace() for character in part):
        return f'"{part}"'
    return part


if __name__ == "__main__":
    raise SystemExit(main())
