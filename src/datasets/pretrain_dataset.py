from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from src.utils.sequence import iter_fasta_records, sequence_hash


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PretrainSample:
    protein_id: str
    sequence: str
    sequence_hash: str
    nature_path: str
    md_path: str

    def to_training_fields(self) -> Dict[str, str]:
        return {
            "protein_id": self.protein_id,
            "sequence": self.sequence,
            "sequence_hash": self.sequence_hash,
            "nature_path": self.nature_path,
            "md_path": self.md_path,
        }


@dataclass(frozen=True)
class PretrainMatchRecord:
    protein_id: str
    sequence_hash: str
    nature_path: str | None
    md_path: str | None
    has_dyn: bool


@dataclass(frozen=True)
class PretrainHashIndex:
    unique_matches: dict[str, PretrainMatchRecord]
    ambiguous_matches: dict[str, tuple[PretrainMatchRecord, ...]]


class PretrainDataset:
    def __init__(self, samples: List[PretrainSample]) -> None:
        self.samples = list(samples)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        sample_limit: Optional[int] = None,
        full_only: bool = True,
        nature_dir: str | Path | None = None,
        md_dir: str | Path | None = None,
    ) -> "PretrainDataset":
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrain manifest does not exist: {path}")
        manifest_dir = path.parent.resolve()
        nature_root = Path(nature_dir).resolve() if nature_dir is not None else None
        md_root = Path(md_dir).resolve() if md_dir is not None else None
        if nature_root is not None and not nature_root.exists():
            raise FileNotFoundError(f"Pretrain nature directory does not exist: {nature_root}")
        if md_root is not None and not md_root.exists():
            raise FileNotFoundError(f"Pretrain MD directory does not exist: {md_root}")

        samples: List[PretrainSample] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                protein_id = (row.get("protein_id") or "").strip()
                nature_path = _resolve_override_or_manifest_path(
                    root=nature_root,
                    protein_id=protein_id,
                    resolver=_resolve_nature_path,
                    raw_manifest_path=row.get("nature_path"),
                    manifest_dir=manifest_dir,
                )
                md_path = _resolve_override_or_manifest_path(
                    root=md_root,
                    protein_id=protein_id,
                    resolver=_resolve_md_path,
                    raw_manifest_path=row.get("md_path"),
                    manifest_dir=manifest_dir,
                )
                is_full = bool(nature_path and md_path) if (nature_root is not None or md_root is not None) else _parse_bool(row.get("is_full"))
                if full_only and not is_full:
                    continue
                if not nature_path or not md_path:
                    continue
                if not _path_has_non_empty_file(nature_path):
                    continue
                if not _path_has_non_empty_file(md_path):
                    continue

                samples.append(
                    PretrainSample(
                        protein_id=protein_id,
                        sequence=(row.get("sequence") or "").strip(),
                        sequence_hash=(row.get("sequence_hash") or "").strip(),
                        nature_path=nature_path,
                        md_path=md_path,
                    )
                )
                if sample_limit is not None and len(samples) >= sample_limit:
                    break

        return cls(samples)

    @classmethod
    def from_dataset_root(
        cls,
        dataset_root: str | Path,
        *,
        sample_limit: Optional[int] = None,
    ) -> "PretrainDataset":
        root = Path(dataset_root)
        fasta_path = root / "all_sequences.fasta"
        nature_root = root / "nature"
        md_root = root / "md"
        if not fasta_path.exists():
            raise FileNotFoundError(f"Pretrain FASTA does not exist: {fasta_path}")
        if not nature_root.exists():
            raise FileNotFoundError(f"Pretrain nature directory does not exist: {nature_root}")
        if not md_root.exists():
            raise FileNotFoundError(f"Pretrain MD directory does not exist: {md_root}")

        samples: List[PretrainSample] = []
        for record in iter_fasta_records(fasta_path, limit=sample_limit):
            protein_id = record.protein_id.strip()
            nature_path = _resolve_nature_path(nature_root, protein_id)
            md_path = _resolve_md_path(md_root, protein_id)
            if nature_path is None or md_path is None:
                continue
            samples.append(
                PretrainSample(
                    protein_id=protein_id,
                    sequence=record.sequence,
                    sequence_hash=sequence_hash(record.sequence),
                    nature_path=str(nature_path),
                    md_path=str(md_path),
                )
            )
        return cls(samples)

    def summary(self) -> Dict[str, int]:
        return {
            "num_samples": len(self.samples),
            "num_full_samples": len(self.samples),
        }

    def limit(self, sample_limit: Optional[int]) -> "PretrainDataset":
        if sample_limit is None:
            return PretrainDataset(list(self.samples))
        return PretrainDataset(list(self.samples[:sample_limit]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PretrainSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[PretrainSample]:
        return iter(self.samples)


def build_pretrain_hash_index(
    pretrain_records: Iterable[PretrainSample | PretrainMatchRecord],
) -> PretrainHashIndex:
    grouped_records: dict[str, list[PretrainMatchRecord]] = defaultdict(list)
    for record in pretrain_records:
        if isinstance(record, PretrainSample):
            match_record = PretrainMatchRecord(
                protein_id=record.protein_id,
                sequence_hash=record.sequence_hash,
                nature_path=record.nature_path,
                md_path=record.md_path,
                has_dyn=True,
            )
        else:
            match_record = record
        grouped_records[match_record.sequence_hash].append(match_record)

    unique_matches: dict[str, PretrainMatchRecord] = {}
    ambiguous_matches: dict[str, tuple[PretrainMatchRecord, ...]] = {}
    for sequence_hash_value, records in grouped_records.items():
        if len(records) == 1:
            unique_matches[sequence_hash_value] = records[0]
        else:
            ambiguous_matches[sequence_hash_value] = tuple(records)
    return PretrainHashIndex(
        unique_matches=unique_matches,
        ambiguous_matches=ambiguous_matches,
    )


def _resolve_nature_path(nature_root: Path, protein_id: str) -> Path | None:
    candidate = nature_root / f"{protein_id}.pdb"
    if candidate.exists() and _path_has_non_empty_file(candidate):
        return candidate.resolve()
    return None


def _resolve_md_path(md_root: Path, protein_id: str) -> Path | None:
    candidate_dir = md_root / protein_id
    if candidate_dir.exists() and _path_has_non_empty_file(candidate_dir):
        return candidate_dir.resolve()
    candidate_pdb = md_root / f"{protein_id}.pdb"
    if candidate_pdb.exists() and _path_has_non_empty_file(candidate_pdb):
        return candidate_pdb.resolve()
    return None


def _resolve_override_or_manifest_path(
    *,
    root: Path | None,
    protein_id: str,
    resolver: Any,
    raw_manifest_path: object,
    manifest_dir: Path,
) -> Optional[str]:
    if root is not None and protein_id:
        resolved = resolver(root, protein_id)
        return str(resolved) if resolved is not None else None
    return _normalize_optional_path(raw_manifest_path, base_dir=manifest_dir)


def _parse_bool(raw_value: object) -> bool:
    return str(raw_value).strip().lower() in {"1", "true", "yes"}


def _normalize_optional_path(raw_value: object, *, base_dir: Path) -> Optional[str]:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((base_dir / path).resolve())
        candidates.append((_repo_root() / path).resolve())

    normalized_candidates: list[Path] = []
    for candidate in candidates:
        normalized_candidates.append(candidate)
        normalized_candidates.append(_normalize_md_segment_case(candidate))

    seen: set[str] = set()
    for candidate in normalized_candidates:
        candidate_key = candidate.as_posix()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists():
            return str(candidate)

    return str(normalized_candidates[0])


def _normalize_md_segment_case(path: Path) -> Path:
    path_text = path.as_posix()
    if "/MD/" in path_text:
        return Path(path_text.replace("/MD/", "/md/"))
    if "/md/" in path_text:
        return Path(path_text.replace("/md/", "/MD/"))
    return path


def _path_has_non_empty_file(raw_path: str | Path) -> bool:
    path = Path(raw_path)
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(
            file_path.is_file() and file_path.stat().st_size > 0
            for file_path in path.rglob("*")
        )
    return False
