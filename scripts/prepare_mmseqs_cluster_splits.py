#!/usr/bin/env python3
from __future__ import annotations

# RELEASE_IMPORT_BOOTSTRAP: allow running scripts directly from the repository root.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.datasets.sequence_cluster_preprocessing import (
    ClusterSplitConfig,
    assign_clusters_to_splits,
    read_sequence_records_from_csv,
    run_mmseqs_easy_cluster,
    write_cluster_assignments,
    write_sequence_fasta,
    write_split_assignments,
    write_split_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MMseqs2 protein-similarity cluster splits from a sequence CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV containing sequence records.")
    parser.add_argument("--id-column", type=str, required=True, help="Column used as the stable record identifier.")
    parser.add_argument("--sequence-column", type=str, required=True, help="Column containing protein or peptide sequences.")
    parser.add_argument("--label-column", type=str, default=None, help="Optional label column copied to outputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for FASTA, cluster and split outputs.")
    parser.add_argument("--mmseqs-binary", type=str, default="mmseqs")
    parser.add_argument("--min-seq-id", type=float, default=0.4, help="MMseqs minimum sequence identity.")
    parser.add_argument("--coverage", type=float, default=0.8, help="MMseqs coverage threshold.")
    parser.add_argument("--cov-mode", type=int, default=0, help="MMseqs coverage mode.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-clusters", action="store_true", help="Shuffle cluster order instead of greedy size ordering.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_sequence_records_from_csv(
        args.input_csv,
        id_column=args.id_column,
        sequence_column=args.sequence_column,
        label_column=args.label_column,
    )
    fasta_path = write_sequence_fasta(records, output_dir / "sequences.fasta")
    cluster_by_record_id = run_mmseqs_easy_cluster(
        fasta_path,
        output_prefix=output_dir / "mmseqs_clusters",
        tmp_dir=output_dir / "tmp_mmseqs",
        mmseqs_binary=args.mmseqs_binary,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
    )
    split_by_cluster_id = assign_clusters_to_splits(
        records,
        cluster_by_record_id,
        config=ClusterSplitConfig(
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            shuffle_order=args.shuffle_clusters,
        ),
    )
    write_cluster_assignments(records, cluster_by_record_id, output_dir / "cluster_assignments.csv")
    write_split_assignments(records, cluster_by_record_id, split_by_cluster_id, output_dir / "split_assignments.csv")
    write_split_summary(records, cluster_by_record_id, split_by_cluster_id, output_dir / "split_summary.json")

    print(f"records={len(records)}")
    print(f"clusters={len(set(cluster_by_record_id.values()))}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
