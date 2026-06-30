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

from src.datasets.pretrain_manifest import (
    build_pretrain_manifest,
    summarize_manifest,
    write_pretrain_manifest,
)


DEFAULT_INPUT_FASTA = Path("./data/pretrain/all_sequences.fasta")
DEFAULT_NATURE_DIR = Path("./data/pretrain/nature")
DEFAULT_MD_DIR = Path("./data/pretrain/MD")
DRY_RUN_SAMPLE_LIMIT = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ProtMMLM pretrain manifests."
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=DEFAULT_INPUT_FASTA,
        help="Input FASTA file for pretrain sequences.",
    )
    parser.add_argument(
        "--nature-dir",
        type=Path,
        default=DEFAULT_NATURE_DIR,
        help="Directory containing natural structure files.",
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=DEFAULT_MD_DIR,
        help="Directory containing MD trajectory files or clips.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for the generated manifest.",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only keep samples with both natural structure and MD paths.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan a small sample and print a summary without writing a manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fasta_limit = DRY_RUN_SAMPLE_LIMIT if args.dry_run else None
    records = build_pretrain_manifest(
        args.input_fasta,
        args.nature_dir,
        args.md_dir,
        full_only=args.full_only,
        fasta_limit=fasta_limit,
    )
    summary = summarize_manifest(records)

    if args.dry_run:
        print("Dry-run summary")
        print(f"sample_limit={DRY_RUN_SAMPLE_LIMIT}")
        print(f"total={summary['total']}")
        print(f"has_nature={summary['has_nature']}")
        print(f"has_md={summary['has_md']}")
        print(f"is_full={summary['is_full']}")
        return 0

    if args.output is None:
        parser.error("--output is required unless --dry-run is set.")

    write_pretrain_manifest(args.output, records)
    print(f"Wrote {summary['total']} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
