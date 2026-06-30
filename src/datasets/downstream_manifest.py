from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.datasets.downstream_adapters import DownstreamSample
from src.utils.sequence import sequence_hash


DOWNSTREAM_MANIFEST_FIELDNAMES = (
    "sample_id",
    "sequence",
    "sequence_hash",
    "peptide_sequence",
    "peptide_sequence_hash",
    "pair_key",
    "label",
    "task_name",
    "split",
    "matched_pretrain_id",
    "nature_path",
    "md_path",
    "has_dyn",
    "peptide_matched_pretrain_id",
    "peptide_nature_path",
    "peptide_md_path",
    "peptide_has_dyn",
    "dyn_cache_path",
    "peptide_dyn_cache_path",
)


@dataclass(frozen=True)
class PretrainMatchRecord:
    protein_id: str
    sequence_hash: str
    nature_path: str | None
    md_path: str | None
    is_full: bool


@dataclass(frozen=True)
class PretrainHashIndex:
    unique_matches: dict[str, PretrainMatchRecord]
    ambiguous_matches: dict[str, tuple[PretrainMatchRecord, ...]]


@dataclass(frozen=True)
class DownstreamManifestRecord:
    sample_id: str
    sequence: str
    sequence_hash: str
    peptide_sequence: str | None
    peptide_sequence_hash: str | None
    pair_key: str | None
    label: str
    task_name: str
    split: str | None
    matched_pretrain_id: str | None
    nature_path: str | None
    md_path: str | None
    has_dyn: bool
    peptide_matched_pretrain_id: str | None = None
    peptide_nature_path: str | None = None
    peptide_md_path: str | None = None
    peptide_has_dyn: bool = False
    dyn_cache_path: str | None = None
    peptide_dyn_cache_path: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["peptide_sequence"] = self.peptide_sequence or ""
        row["peptide_sequence_hash"] = self.peptide_sequence_hash or ""
        row["pair_key"] = self.pair_key or ""
        row["split"] = self.split or ""
        row["matched_pretrain_id"] = self.matched_pretrain_id or ""
        row["nature_path"] = self.nature_path or ""
        row["md_path"] = self.md_path or ""
        row["peptide_matched_pretrain_id"] = self.peptide_matched_pretrain_id or ""
        row["peptide_nature_path"] = self.peptide_nature_path or ""
        row["peptide_md_path"] = self.peptide_md_path or ""
        row["dyn_cache_path"] = self.dyn_cache_path or ""
        row["peptide_dyn_cache_path"] = self.peptide_dyn_cache_path or ""
        return row


@dataclass(frozen=True)
class SideMatchVariant:
    matched_pretrain_id: str | None
    nature_path: str | None
    md_path: str | None
    has_dyn: bool


EMPTY_SIDE_VARIANT = SideMatchVariant(
    matched_pretrain_id=None,
    nature_path=None,
    md_path=None,
    has_dyn=False,
)


def load_pretrain_manifest(path: str | Path) -> list[PretrainMatchRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pretrain manifest does not exist: {manifest_path}")

    records: list[PretrainMatchRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence_hash_value = (row.get("sequence_hash") or "").strip()
            if not sequence_hash_value:
                raise ValueError(
                    f"Pretrain manifest row is missing sequence_hash: {manifest_path}"
                )
            nature_path = _normalize_optional_path(row.get("nature_path"))
            md_path = _normalize_optional_path(row.get("md_path"))
            has_valid_dyn = (
                _parse_bool(row.get("is_full"))
                and nature_path is not None
                and md_path is not None
                and _path_has_non_empty_file(nature_path)
                and _path_has_non_empty_file(md_path)
            )

            records.append(
                PretrainMatchRecord(
                    protein_id=(row.get("protein_id") or "").strip(),
                    sequence_hash=sequence_hash_value,
                    nature_path=nature_path if has_valid_dyn else None,
                    md_path=md_path if has_valid_dyn else None,
                    is_full=has_valid_dyn,
                )
            )

    return records


def build_pretrain_hash_index(
    pretrain_records: Iterable[PretrainMatchRecord],
) -> PretrainHashIndex:
    grouped_records: dict[str, list[PretrainMatchRecord]] = defaultdict(list)
    for record in pretrain_records:
        grouped_records[record.sequence_hash].append(record)

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


def _as_side_variant(record: PretrainMatchRecord) -> SideMatchVariant:
    has_dyn = bool(record.nature_path and record.md_path)
    return SideMatchVariant(
        matched_pretrain_id=record.protein_id or None,
        nature_path=record.nature_path if has_dyn else None,
        md_path=record.md_path if has_dyn else None,
        has_dyn=has_dyn,
    )


def _collect_full_ambiguous_variants(
    records: tuple[PretrainMatchRecord, ...],
) -> tuple[SideMatchVariant, ...]:
    variants: list[SideMatchVariant] = []
    seen_keys: set[tuple[str | None, str | None, str | None]] = set()

    for record in records:
        if not record.is_full:
            continue
        if record.nature_path is None or record.md_path is None:
            continue

        dedupe_key = (
            record.protein_id or None,
            record.nature_path,
            record.md_path,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        variants.append(_as_side_variant(record))

    return tuple(variants)


def _resolve_side_variants(
    sequence_hash_value: str,
    pretrain_index: PretrainHashIndex,
) -> tuple[tuple[SideMatchVariant, ...], bool]:
    unique_match = pretrain_index.unique_matches.get(sequence_hash_value)
    if unique_match is not None:
        return (_as_side_variant(unique_match),), False

    ambiguous_records = pretrain_index.ambiguous_matches.get(sequence_hash_value)
    if ambiguous_records is None:
        return (EMPTY_SIDE_VARIANT,), False

    full_variants = _collect_full_ambiguous_variants(ambiguous_records)
    if full_variants:
        return full_variants, True

    return (EMPTY_SIDE_VARIANT,), True


def build_downstream_manifest(
    samples: Iterable[DownstreamSample],
    pretrain_records: Iterable[PretrainMatchRecord],
) -> tuple[list[DownstreamManifestRecord], dict[str, Any]]:
    sample_list = list(samples)
    pretrain_index = build_pretrain_hash_index(pretrain_records)

    manifest_records: list[DownstreamManifestRecord] = []
    hash_to_sample_ids: dict[str, list[str]] = defaultdict(list)
    ambiguous_pretrain_samples = 0
    peptide_ambiguous_pretrain_samples = 0
    expanded_input_samples = 0

    for sample in sample_list:
        sample_hash = sequence_hash(sample.sequence)
        hash_to_sample_ids[sample_hash].append(sample.sample_id)

        protein_variants, protein_is_ambiguous = _resolve_side_variants(
            sample_hash,
            pretrain_index,
        )
        ambiguous_pretrain_samples += int(protein_is_ambiguous)

        peptide_sequence = sample.peptide_sequence
        peptide_hash: str | None = None
        base_pair_key: str | None = None
        peptide_variants: tuple[SideMatchVariant, ...] = (EMPTY_SIDE_VARIANT,)

        if peptide_sequence:
            peptide_hash = sequence_hash(peptide_sequence)
            base_pair_key = sample.pair_key or f"{sample_hash}::{peptide_hash}"
            peptide_variants, peptide_is_ambiguous = _resolve_side_variants(
                peptide_hash,
                pretrain_index,
            )
            peptide_ambiguous_pretrain_samples += int(peptide_is_ambiguous)

        num_combinations = len(protein_variants) * len(peptide_variants)
        if num_combinations > 1:
            expanded_input_samples += 1

        for protein_index, protein_variant in enumerate(protein_variants, start=1):
            for peptide_index, peptide_variant in enumerate(peptide_variants, start=1):
                sample_id = sample.sample_id
                pair_key = base_pair_key
                if num_combinations > 1:
                    suffix = f"::aug_p{protein_index}_t{peptide_index}"
                    sample_id = f"{sample.sample_id}{suffix}"
                    if pair_key is not None:
                        pair_key = f"{pair_key}{suffix}"

                manifest_records.append(
                    DownstreamManifestRecord(
                        sample_id=sample_id,
                        sequence=sample.sequence,
                        sequence_hash=sample_hash,
                        peptide_sequence=peptide_sequence,
                        peptide_sequence_hash=peptide_hash,
                        pair_key=pair_key,
                        label=sample.label,
                        task_name=sample.task_name,
                        split=sample.split,
                        matched_pretrain_id=protein_variant.matched_pretrain_id,
                        nature_path=protein_variant.nature_path,
                        md_path=protein_variant.md_path,
                        has_dyn=protein_variant.has_dyn,
                        peptide_matched_pretrain_id=peptide_variant.matched_pretrain_id,
                        peptide_nature_path=peptide_variant.nature_path,
                        peptide_md_path=peptide_variant.md_path,
                        peptide_has_dyn=peptide_variant.has_dyn,
                    )
                )

    duplicate_groups = {
        hash_value: sample_ids
        for hash_value, sample_ids in hash_to_sample_ids.items()
        if len(sample_ids) > 1
    }

    task_name = sample_list[0].task_name if sample_list else ""
    total_input_samples = len(sample_list)
    total_samples = len(manifest_records)

    matched_samples = sum(int(bool(record.matched_pretrain_id)) for record in manifest_records)
    has_dyn_samples = sum(int(record.has_dyn) for record in manifest_records)
    seq_only_samples = total_samples - has_dyn_samples

    pair_samples = sum(int(bool(record.peptide_sequence)) for record in manifest_records)
    peptide_matched_samples = sum(
        int(bool(record.peptide_matched_pretrain_id))
        for record in manifest_records
        if record.peptide_sequence
    )
    peptide_has_dyn_samples = sum(
        int(record.peptide_has_dyn)
        for record in manifest_records
        if record.peptide_sequence
    )
    pair_has_any_dyn_samples = sum(
        int(record.has_dyn or record.peptide_has_dyn)
        for record in manifest_records
        if record.peptide_sequence
    )
    pair_has_both_dyn_samples = sum(
        int(record.has_dyn and record.peptide_has_dyn)
        for record in manifest_records
        if record.peptide_sequence
    )

    split_counts: dict[str, int] = defaultdict(int)
    for record in manifest_records:
        if record.split:
            split_counts[record.split] += 1

    duplicates_report = {
        "task_name": task_name,
        "input_samples": total_input_samples,
        "total_samples": total_samples,
        "expanded_samples": total_samples - total_input_samples,
        "expanded_input_samples": expanded_input_samples,
        "matched_samples": matched_samples,
        "has_dyn_samples": has_dyn_samples,
        "seq_only_samples": seq_only_samples,
        "pair_samples": pair_samples,
        "peptide_matched_samples": peptide_matched_samples,
        "peptide_has_dyn_samples": peptide_has_dyn_samples,
        "pair_has_any_dyn_samples": pair_has_any_dyn_samples,
        "pair_has_both_dyn_samples": pair_has_both_dyn_samples,
        "unique_sequence_hashes": len(hash_to_sample_ids),
        "duplicate_sequence_hashes": len(duplicate_groups),
        "duplicate_samples": sum(len(sample_ids) - 1 for sample_ids in duplicate_groups.values()),
        "duplicate_groups": duplicate_groups,
        "ambiguous_pretrain_hashes": len(pretrain_index.ambiguous_matches),
        "ambiguous_pretrain_samples": ambiguous_pretrain_samples,
        "peptide_ambiguous_pretrain_samples": peptide_ambiguous_pretrain_samples,
        "split_counts": dict(sorted(split_counts.items())),
    }

    return manifest_records, duplicates_report


def write_downstream_manifest(
    output_path: str | Path,
    records: Iterable[DownstreamManifestRecord],
) -> None:
    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOWNSTREAM_MANIFEST_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def write_duplicates_report(output_path: str | Path, report: dict[str, Any]) -> None:
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _normalize_optional_path(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


def _parse_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _path_has_non_empty_file(raw_path: str | Path) -> bool:
    path = Path(raw_path)
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(file_path.is_file() and file_path.stat().st_size > 0 for file_path in path.rglob("*"))
    return False
