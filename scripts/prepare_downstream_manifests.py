#!/usr/bin/env python3
from __future__ import annotations

# RELEASE_IMPORT_BOOTSTRAP: allow running scripts directly from the repository root.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from src.datasets.downstream_adapters import load_downstream_samples
from src.datasets.downstream_manifest import (
    build_downstream_manifest,
    load_pretrain_manifest,
    write_downstream_manifest,
    write_duplicates_report,
)


DEFAULT_INPUT_PATHS = {
    "toxteller": Path("./datasets/downstream/Toxteller"),
    "prmftp": Path("./datasets/downstream/PrMFTP"),
    "ppikb": Path("./datasets/downstream/PPIKB"),
    "conotoxin": Path("./datasets/downstream/Conotoxin"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ProtMMLM downstream manifests."
    )
    parser.add_argument(
        "--task",
        choices=tuple(DEFAULT_INPUT_PATHS.keys()),
        required=True,
        help="Downstream task to prepare.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional input path override for the selected task.",
    )
    parser.add_argument(
        "--pretrain-manifest",
        type=Path,
        required=True,
        help="CSV manifest produced by the pretrain data preparation step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the downstream manifest and duplicates report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = args.input_path or DEFAULT_INPUT_PATHS[args.task]

    samples = load_downstream_samples(
        args.task,
        input_path,
        sample_limit=None,
    )
    pretrain_records = load_pretrain_manifest(args.pretrain_manifest)
    manifest_records, duplicates_report = build_downstream_manifest(samples, pretrain_records)

    output_dir = args.output_dir
    manifest_path = output_dir / f"{args.task}_manifest.csv"
    duplicates_report_path = output_dir / f"{args.task}_duplicates_report.json"

    write_downstream_manifest(manifest_path, manifest_records)
    write_duplicates_report(duplicates_report_path, duplicates_report)

    print(f"Wrote {duplicates_report['total_samples']} records to {manifest_path}")
    print(f"Wrote duplicates report to {duplicates_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
