from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.utils.sequence import iter_fasta_records, sequence_hash


MANIFEST_FIELDNAMES = (
    "protein_id",
    "sequence",
    "sequence_hash",
    "nature_path",
    "md_path",
    "has_nature",
    "has_md",
    "is_full",
)


@dataclass(frozen=True)
class PathIndex:
    matches: dict[str, Path]
    ambiguous: dict[str, tuple[Path, ...]]


@dataclass(frozen=True)
class PretrainManifestRecord:
    protein_id: str
    sequence: str
    sequence_hash: str
    nature_path: str | None
    md_path: str | None
    has_nature: bool
    has_md: bool
    is_full: bool

    def to_row(self) -> dict[str, str | bool]:
        row = asdict(self)
        row["nature_path"] = self.nature_path or ""
        row["md_path"] = self.md_path or ""
        return row


def _normalize_protein_key(value: str) -> str:
    return value.strip().upper()


def _build_path_index(root: str | Path, *, include_parent_name: bool) -> PathIndex:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"Path is not a directory: {root_path}")

    candidates: dict[str, set[Path]] = defaultdict(set)

    for file_path in sorted(
        (
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.stat().st_size > 0
        ),
        key=lambda path: path.as_posix(),
    ):
        keys = {_normalize_protein_key(file_path.stem)}

        if include_parent_name and file_path.parent != root_path:
            # Extension point: update this rule only when a concrete MD naming schema requires it.
            keys.add(_normalize_protein_key(file_path.parent.name))

        for key in keys:
            if key:
                candidates[key].add(file_path)

    matches: dict[str, Path] = {}
    ambiguous: dict[str, tuple[Path, ...]] = {}
    for key, paths in candidates.items():
        ordered_paths = tuple(sorted(paths, key=lambda path: path.as_posix()))
        if len(ordered_paths) == 1:
            matches[key] = ordered_paths[0]
        else:
            ambiguous[key] = ordered_paths

    return PathIndex(matches=matches, ambiguous=ambiguous)


def scan_nature_paths(nature_dir: str | Path) -> PathIndex:
    return _build_path_index(nature_dir, include_parent_name=False)


def scan_md_paths(md_dir: str | Path) -> PathIndex:
    return _build_path_index(md_dir, include_parent_name=True)


def build_pretrain_manifest(
    input_fasta: str | Path,
    nature_dir: str | Path,
    md_dir: str | Path,
    *,
    full_only: bool = False,
    fasta_limit: int | None = None,
) -> list[PretrainManifestRecord]:
    nature_index = scan_nature_paths(nature_dir)
    md_index = scan_md_paths(md_dir)

    records: list[PretrainManifestRecord] = []
    for fasta_record in iter_fasta_records(input_fasta, limit=fasta_limit):
        protein_key = _normalize_protein_key(fasta_record.protein_id)
        nature_path = nature_index.matches.get(protein_key)
        md_path = md_index.matches.get(protein_key)

        manifest_record = PretrainManifestRecord(
            protein_id=fasta_record.protein_id,
            sequence=fasta_record.sequence,
            sequence_hash=sequence_hash(fasta_record.sequence),
            nature_path=str(nature_path) if nature_path is not None else None,
            md_path=str(md_path) if md_path is not None else None,
            has_nature=nature_path is not None,
            has_md=md_path is not None,
            is_full=nature_path is not None and md_path is not None,
        )

        if not full_only or manifest_record.is_full:
            records.append(manifest_record)

    return records


def summarize_manifest(records: Iterable[PretrainManifestRecord]) -> dict[str, int]:
    total = 0
    has_nature = 0
    has_md = 0
    is_full = 0

    for record in records:
        total += 1
        has_nature += int(record.has_nature)
        has_md += int(record.has_md)
        is_full += int(record.is_full)

    return {
        "total": total,
        "has_nature": has_nature,
        "has_md": has_md,
        "is_full": is_full,
    }


def write_pretrain_manifest(
    output_path: str | Path,
    records: Iterable[PretrainManifestRecord],
) -> None:
    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
