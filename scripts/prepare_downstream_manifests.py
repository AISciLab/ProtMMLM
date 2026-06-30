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
    "toxteller": Path("./data/downstream/ToxTeller"),
    "prmftp": Path("./data/downstream/PrMFTP"),
    "ppikb": Path("./data/downstream/processed/regression"),
    "conotoxin": Path("./data/downstream/conotoxin/filtered_id70"),
}
DRY_RUN_SAMPLE_LIMIT = 10


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
        default=None,
        help="Output directory for the downstream manifest and duplicates report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load only a small sample and print a summary without writing output files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = args.input_path or DEFAULT_INPUT_PATHS[args.task]
    sample_limit = DRY_RUN_SAMPLE_LIMIT if args.dry_run else None

    samples = load_downstream_samples(
        args.task,
        input_path,
        sample_limit=sample_limit,
    )
    pretrain_records = load_pretrain_manifest(args.pretrain_manifest)
    manifest_records, duplicates_report = build_downstream_manifest(samples, pretrain_records)

    if args.dry_run:
        print("Dry-run summary")
        print(f"task={args.task}")
        print(f"sample_limit={DRY_RUN_SAMPLE_LIMIT}")
        print(f"total={duplicates_report['total_samples']}")
        print(f"matched={duplicates_report['matched_samples']}")
        print(f"has_dyn={duplicates_report['has_dyn_samples']}")
        print(f"seq_only={duplicates_report['seq_only_samples']}")
        print(
            "duplicate_sequence_hashes="
            f"{duplicates_report['duplicate_sequence_hashes']}"
        )
        return 0

    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is set.")

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
