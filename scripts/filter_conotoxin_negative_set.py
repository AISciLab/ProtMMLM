#!/usr/bin/env python3
"""Filter conotoxin negative samples similar to positive samples.

The script keeps the dataset in the original two-FASTA layout and removes negative
records that have an MMseqs2 hit to any positive record above the requested
sequence-identity threshold.
"""
from __future__ import annotations

# RELEASE_IMPORT_BOOTSTRAP: allow running scripts directly from the repository root.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("data/downstream/conotoxin")
DEFAULT_OUTPUT_DIR = Path("data/downstream/conotoxin/filtered_id70")


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str

    @property
    def name(self) -> str:
        return self.header[1:].split()[0] if self.header.startswith(">") else self.header.split()[0]


def read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    seq_chunks: list[str] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(FastaRecord(header=header, sequence="".join(seq_chunks).upper()))
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line.replace(" ", ""))

    if header is not None:
        records.append(FastaRecord(header=header, sequence="".join(seq_chunks).upper()))

    return records


def write_fasta(path: Path, records: Iterable[FastaRecord], line_width: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(f"{record.header}\n")
            seq = record.sequence
            for start in range(0, len(seq), line_width):
                handle.write(f"{seq[start:start + line_width]}\n")


def write_mmseqs_input(path: Path, records: list[FastaRecord], prefix: str) -> dict[str, FastaRecord]:
    id_to_record: dict[str, FastaRecord] = {}
    with path.open("w") as handle:
        for idx, record in enumerate(records):
            record_id = f"{prefix}_{idx}"
            id_to_record[record_id] = record
            handle.write(f">{record_id}\n{record.sequence}\n")
    return id_to_record


def parse_identity(value: str) -> float:
    identity = float(value)
    if identity > 1.0:
        identity /= 100.0
    return identity


def run_mmseqs_filter(
    positives: list[FastaRecord],
    negatives: list[FastaRecord],
    *,
    identity_threshold: float,
    coverage: float,
    threads: int,
    tmp_root: Path | None,
    keep_tmp: bool,
) -> tuple[set[str], dict[str, dict[str, object]], Path | None]:
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs executable was not found. Please install MMseqs2 or add it to PATH."
        )

    tmp_context = tempfile.TemporaryDirectory(dir=tmp_root)
    tmp_dir = Path(tmp_context.name)
    if keep_tmp:
        # Prevent TemporaryDirectory cleanup at function exit; the path is reported to the user.
        tmp_context.cleanup = lambda: None  # type: ignore[method-assign]

    query_fasta = tmp_dir / "negatives.query.fasta"
    target_fasta = tmp_dir / "positives.target.fasta"
    hits_tsv = tmp_dir / "neg_vs_pos.mmseqs.tsv"
    mmseqs_tmp = tmp_dir / "mmseqs_tmp"

    neg_id_to_record = write_mmseqs_input(query_fasta, negatives, "neg")
    write_mmseqs_input(target_fasta, positives, "pos")

    command = [
        "mmseqs",
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(hits_tsv),
        str(mmseqs_tmp),
        "--min-seq-id",
        str(identity_threshold),
        "-c",
        str(coverage),
        "--cov-mode",
        "2",
        "--threads",
        str(threads),
        "--format-output",
        "query,target,pident,alnlen,qlen,tlen,evalue,bits",
    ]
    subprocess.run(command, check=True)

    removed_ids: set[str] = set()
    best_hits: dict[str, dict[str, object]] = {}

    if hits_tsv.exists():
        with hits_tsv.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    continue
                query_id, target_id, pident, alnlen, qlen, tlen, evalue, bits = fields[:8]
                identity = parse_identity(pident)
                if identity <= identity_threshold:
                    continue

                current = best_hits.get(query_id)
                bit_score = float(bits)
                if current is None or bit_score > float(current["bits"]):
                    best_hits[query_id] = {
                        "query_id": query_id,
                        "target_id": target_id,
                        "identity": identity,
                        "identity_percent": identity * 100.0,
                        "alignment_length": int(alnlen),
                        "query_length": int(qlen),
                        "target_length": int(tlen),
                        "evalue": evalue,
                        "bits": bit_score,
                        "original_header": neg_id_to_record[query_id].header,
                    }
                removed_ids.add(query_id)

    tmp_path = tmp_dir if keep_tmp else None
    if not keep_tmp:
        tmp_context.cleanup()
    return removed_ids, best_hits, tmp_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove conotoxin negative FASTA records whose similarity to any "
            "positive record is greater than the selected threshold."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing pos.fasta and neg.fasta.",
    )
    parser.add_argument("--pos-fasta", type=Path, default=None, help="Positive FASTA path override.")
    parser.add_argument("--neg-fasta", type=Path, default=None, help="Negative FASTA path override.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write filtered pos.fasta, neg.fasta and reports.",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.40,
        help="Remove negatives with identity strictly greater than this value.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.80,
        help=(
            "MMseqs2 coverage cutoff. With --cov-mode 2 this requires the hit "
            "to cover this fraction of the negative/query sequence."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Keep only records with length <= max length before similarity filtering; use 0 to disable.",
    )
    parser.add_argument("--threads", type=int, default=8, help="Threads passed to MMseqs2.")
    parser.add_argument("--tmp-root", type=Path, default=None, help="Optional parent directory for temporary files.")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep MMseqs2 temporary files for debugging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 0.0 <= args.identity_threshold <= 1.0:
        raise SystemExit("--identity-threshold must be between 0 and 1.")
    if not 0.0 <= args.coverage <= 1.0:
        raise SystemExit("--coverage must be between 0 and 1.")
    if args.max_length < 0:
        raise SystemExit("--max-length must be >= 0.")

    pos_fasta = args.pos_fasta or args.input_dir / "pos.fasta"
    neg_fasta = args.neg_fasta or args.input_dir / "neg.fasta"

    positives_raw = read_fasta(pos_fasta)
    negatives_raw = read_fasta(neg_fasta)

    if args.max_length:
        positives = [record for record in positives_raw if len(record.sequence) <= args.max_length]
        negatives = [record for record in negatives_raw if len(record.sequence) <= args.max_length]
    else:
        positives = positives_raw
        negatives = negatives_raw

    removed_ids, best_hits, tmp_path = run_mmseqs_filter(
        positives,
        negatives,
        identity_threshold=args.identity_threshold,
        coverage=args.coverage,
        threads=args.threads,
        tmp_root=args.tmp_root,
        keep_tmp=args.keep_tmp,
    )

    kept_negatives = [record for idx, record in enumerate(negatives) if f"neg_{idx}" not in removed_ids]
    removed_negatives = [record for idx, record in enumerate(negatives) if f"neg_{idx}" in removed_ids]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(output_dir / "pos.fasta", positives)
    write_fasta(output_dir / "neg.fasta", kept_negatives)

    removed_tsv = output_dir / "removed_negatives.tsv"
    with removed_tsv.open("w") as handle:
        handle.write(
            "negative_index\tnegative_name\tpositive_id\tidentity_percent\t"
            "alignment_length\tquery_length\ttarget_length\tevalue\tbits\theader\n"
        )
        for query_id in sorted(removed_ids, key=lambda item: int(item.split("_")[1])):
            hit = best_hits[query_id]
            record = negatives[int(query_id.split("_")[1])]
            handle.write(
                f"{query_id}\t{record.name}\t{hit['target_id']}\t"
                f"{float(hit['identity_percent']):.3f}\t{hit['alignment_length']}\t"
                f"{hit['query_length']}\t{hit['target_length']}\t{hit['evalue']}\t"
                f"{float(hit['bits']):.1f}\t{record.header}\n"
            )

    report = {
        "pos_fasta": str(pos_fasta),
        "neg_fasta": str(neg_fasta),
        "output_dir": str(output_dir),
        "identity_threshold": args.identity_threshold,
        "coverage": args.coverage,
        "max_length": args.max_length,
        "raw_positive_count": len(positives_raw),
        "raw_negative_count": len(negatives_raw),
        "length_filtered_positive_count": len(positives),
        "length_filtered_negative_count": len(negatives),
        "removed_negative_count": len(removed_negatives),
        "kept_negative_count": len(kept_negatives),
        "removed_negatives_tsv": str(removed_tsv),
    }
    if tmp_path is not None:
        report["tmp_dir"] = str(tmp_path)

    report_path = output_dir / "filter_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Positive records: {len(positives_raw)} raw -> {len(positives)} after length filter")
    print(f"Negative records: {len(negatives_raw)} raw -> {len(negatives)} after length filter")
    print(
        "Removed negatives with identity "
        f"> {args.identity_threshold:.2%}: {len(removed_negatives)}"
    )
    print(f"Kept negatives: {len(kept_negatives)}")
    print(f"Wrote filtered FASTA files to {output_dir}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
