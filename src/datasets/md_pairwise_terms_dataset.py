from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ENERGY_TERMS: tuple[str, ...] = (
    "vdw",
    "hbbb",
    "hbsb",
    "hbss",
    "hp",
    "sb",
    "pc",
    "ps",
    "ts",
)
ENERGY_TERM_TO_INDEX: dict[str, int] = {term: index for index, term in enumerate(ENERGY_TERMS)}
MD_PAIRWISE_MANIFEST_FIELDS: tuple[str, ...] = (
    "sample_id",
    "protein_id",
    "sequence",
    "sequence_hash",
    "nature_path",
    "md_path",
    "interaction_tsv_path",
    "map_path",
    "length",
    "total_frames",
    "split",
    "cluster_id",
)

_ATOM_PATTERN = re.compile(r"^(?P<chain>[^:]+):(?P<resname>[^:]+):(?P<resid>[^:]+):(?P<atom>[^:]+)$")
_INT_PATTERN = re.compile(r"[-+]?\d+")


@dataclass(frozen=True)
class ParsedAtom:
    chain: str
    resname: str
    resid: str
    atom_name: str

    @property
    def residue_key(self) -> tuple[str, str, str]:
        return (self.chain, self.resname, self.resid)

    @property
    def residue_id(self) -> str:
        return f"{self.chain}:{self.resname}:{self.resid}"


@dataclass(frozen=True)
class BuildMapSummary:
    protein_id: str
    map_path: str
    length: int
    total_frames: int
    known_events: int
    skipped_unknown_terms: int
    skipped_malformed_rows: int
    skipped_unmapped_residues: int
    skipped_self_pairs: int


@dataclass(frozen=True)
class MDPairwiseSample:
    sample_id: str
    protein_id: str
    sequence: str
    sequence_hash: str
    nature_path: str
    md_path: str
    interaction_tsv_path: str
    map_path: str
    length: int
    total_frames: int
    split: str
    cluster_id: str


@dataclass(frozen=True)
class MDPairwiseExample:
    sample_id: str
    protein_id: str
    sequence: str
    nature_path: str
    md_path: str
    y: Any
    m_all: Any
    residue_mask: Any
    pair_mask: Any
    length: int


@dataclass(frozen=True)
class MDPairwiseBatch:
    sample_ids: list[str]
    protein_ids: list[str]
    sequences: list[str]
    nature_paths: list[str]
    md_paths: list[str]
    y: Any
    m_all: Any
    residue_mask: Any
    pair_mask: Any
    lengths: Any

    def to(self, device: str) -> "MDPairwiseBatch":
        return MDPairwiseBatch(
            sample_ids=self.sample_ids,
            protein_ids=self.protein_ids,
            sequences=self.sequences,
            nature_paths=self.nature_paths,
            md_paths=self.md_paths,
            y=self.y.to(device),
            m_all=self.m_all.to(device),
            residue_mask=self.residue_mask.to(device),
            pair_mask=self.pair_mask.to(device),
            lengths=self.lengths.to(device),
        )


def stable_sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def parse_atom_id(raw_atom: str) -> ParsedAtom:
    match = _ATOM_PATTERN.match(raw_atom.strip())
    if match is None:
        raise ValueError(f"Malformed atom identifier: {raw_atom!r}")
    return ParsedAtom(
        chain=match.group("chain"),
        resname=match.group("resname"),
        resid=match.group("resid"),
        atom_name=match.group("atom"),
    )


def parse_total_frames_from_header(tsv_path: str | Path) -> int | None:
    path = Path(tsv_path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                return None
            if "total_frames:" in stripped:
                for token in stripped.lstrip("#").split():
                    if token.startswith("total_frames:"):
                        raw_value = token.split(":", 1)[1]
                        try:
                            return int(raw_value)
                        except ValueError as exc:
                            raise ValueError(f"Invalid total_frames value in {path}: {raw_value!r}") from exc
    return None


def _residue_sort_key(key: tuple[str, str, str]) -> tuple[str, int, str, str]:
    chain, resname, resid = key
    match = _INT_PATTERN.search(resid)
    numeric = int(match.group(0)) if match is not None else 10**9
    return (chain, numeric, resid, resname)


def _scan_residue_keys(tsv_path: Path) -> list[tuple[str, str, str]]:
    residue_keys: set[tuple[str, str, str]] = set()
    with tsv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                atom_1 = parse_atom_id(parts[2])
                atom_2 = parse_atom_id(parts[3])
            except ValueError:
                continue
            residue_keys.add(atom_1.residue_key)
            residue_keys.add(atom_2.residue_key)
    return sorted(residue_keys, key=_residue_sort_key)


def _residue_numeric_position(resid: str) -> int | None:
    match = _INT_PATTERN.search(resid)
    if match is None:
        return None
    return int(match.group(0))


def _build_residue_to_index(residue_keys: Sequence[tuple[str, str, str]], *, length: int) -> dict[tuple[str, str, str], int]:
    numeric_positions = [_residue_numeric_position(key[2]) for key in residue_keys]
    if numeric_positions and all(position is not None for position in numeric_positions):
        min_position = min(int(position) for position in numeric_positions if position is not None)
        if min_position in {0, 1}:
            offset = min_position
            by_resid: dict[tuple[str, str, str], int] = {}
            for key, position in zip(residue_keys, numeric_positions):
                assert position is not None
                index = int(position) - offset
                if 0 <= index < length:
                    by_resid[key] = index
            if by_resid:
                return by_resid
    return {key: index for index, key in enumerate(residue_keys[:length])}


def build_md_pairwise_map(
    *,
    protein_id: str,
    sequence: str,
    interaction_tsv_path: str | Path,
    output_path: str | Path,
) -> BuildMapSummary:
    tsv_path = Path(interaction_tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"MD interaction TSV does not exist: {tsv_path}")
    length = len(sequence)
    if length <= 0:
        raise ValueError(f"Protein {protein_id} has empty sequence.")

    residue_keys = _scan_residue_keys(tsv_path)
    residue_to_index = _build_residue_to_index(residue_keys, length=length)
    residue_ids_by_index = [f"UNK:UNK:{index + 1}" for index in range(length)]
    for key, index in residue_to_index.items():
        chain, resname, resid = key
        residue_ids_by_index[index] = f"{chain}:{resname}:{resid}"
    residue_ids = residue_ids_by_index

    frame_term_pairs: set[tuple[int, int, int, int]] = set()
    frame_any_pairs: set[tuple[int, int, int]] = set()
    observed_frames: set[int] = set()
    known_events = 0
    skipped_unknown_terms = 0
    skipped_malformed_rows = 0
    skipped_unmapped_residues = 0
    skipped_self_pairs = 0

    with tsv_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                skipped_malformed_rows += 1
                continue
            try:
                frame = int(parts[0])
                interaction_type = parts[1].strip()
                atom_1 = parse_atom_id(parts[2])
                atom_2 = parse_atom_id(parts[3])
            except ValueError:
                skipped_malformed_rows += 1
                continue
            observed_frames.add(frame)
            term_index = ENERGY_TERM_TO_INDEX.get(interaction_type)
            if term_index is None:
                skipped_unknown_terms += 1
                continue
            index_1 = residue_to_index.get(atom_1.residue_key)
            index_2 = residue_to_index.get(atom_2.residue_key)
            if index_1 is None or index_2 is None:
                skipped_unmapped_residues += 1
                continue
            if index_1 == index_2:
                skipped_self_pairs += 1
                continue
            i, j = sorted((index_1, index_2))
            frame_term_pairs.add((frame, i, j, term_index))
            frame_any_pairs.add((frame, i, j))
            known_events += 1

    header_total_frames = parse_total_frames_from_header(tsv_path)
    if header_total_frames is not None:
        total_frames = header_total_frames
    elif observed_frames:
        total_frames = max(observed_frames) + 1
    else:
        raise ValueError(f"No frames could be inferred from {tsv_path}")
    if total_frames <= 0:
        raise ValueError(f"Protein {protein_id} has non-positive total_frames={total_frames}.")

    counts = np.zeros((length, length, len(ENERGY_TERMS)), dtype=np.float32)
    any_counts = np.zeros((length, length), dtype=np.float32)
    for _frame, i, j, term_index in frame_term_pairs:
        counts[i, j, term_index] += 1.0
    for _frame, i, j in frame_any_pairs:
        any_counts[i, j] += 1.0

    y = counts / float(total_frames)
    m_all = any_counts / float(total_frames)
    y = y + np.transpose(y, (1, 0, 2))
    m_all = m_all + m_all.T

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        y=y.astype(np.float32, copy=False),
        m_all=m_all.astype(np.float32, copy=False),
        energy_terms=np.asarray(ENERGY_TERMS, dtype="U8"),
        residue_ids=np.asarray(residue_ids, dtype="U32"),
        total_frames=np.asarray(total_frames, dtype=np.int64),
    )
    return BuildMapSummary(
        protein_id=protein_id,
        map_path=str(out_path),
        length=length,
        total_frames=total_frames,
        known_events=known_events,
        skipped_unknown_terms=skipped_unknown_terms,
        skipped_malformed_rows=skipped_malformed_rows,
        skipped_unmapped_residues=skipped_unmapped_residues,
        skipped_self_pairs=skipped_self_pairs,
    )


def iter_pretrain_manifest_rows(manifest_path: str | Path) -> Iterable[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key: (value or "") for key, value in row.items()}


def build_md_pairwise_manifest(
    *,
    pretrain_manifest_path: str | Path,
    interaction_dir: str | Path,
    output_manifest_path: str | Path,
    maps_dir: str | Path,
    sample_limit: int | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    interaction_root = Path(interaction_dir)
    manifest_path = Path(output_manifest_path)
    maps_root = Path(maps_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    summaries: list[BuildMapSummary] = []
    skipped_missing_tsv = 0
    skipped_no_md = 0
    skipped_errors: list[dict[str, str]] = []
    source_rows = list(iter_pretrain_manifest_rows(pretrain_manifest_path))
    processed = 0
    total = len(source_rows)

    def emit_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "processed": processed,
                "total": total,
                "built": len(rows),
                "missing_tsv": skipped_missing_tsv,
                "no_md": skipped_no_md,
                "errors": len(skipped_errors),
            }
        )

    for source_row in source_rows:
        processed += 1
        protein_id = source_row.get("protein_id", "").strip()
        sequence = source_row.get("sequence", "").strip()
        md_path = source_row.get("md_path", "").strip()
        nature_path = source_row.get("nature_path", "").strip()
        if not protein_id or not sequence:
            emit_progress()
            continue
        if not md_path:
            skipped_no_md += 1
            emit_progress()
            continue
        tsv_path = interaction_root / protein_id / f"{protein_id}.tsv"
        if not tsv_path.exists():
            skipped_missing_tsv += 1
            emit_progress()
            continue
        map_path = maps_root / f"{protein_id}.npz"
        try:
            summary = build_md_pairwise_map(
                protein_id=protein_id,
                sequence=sequence,
                interaction_tsv_path=tsv_path,
                output_path=map_path,
            )
        except Exception as exc:  # noqa: BLE001 - collect build errors in summary.
            skipped_errors.append({"protein_id": protein_id, "error": str(exc)})
            emit_progress()
            continue
        summaries.append(summary)
        sequence_hash = source_row.get("sequence_hash", "").strip() or stable_sequence_hash(sequence)
        rows.append(
            {
                "sample_id": protein_id,
                "protein_id": protein_id,
                "sequence": sequence,
                "sequence_hash": sequence_hash,
                "nature_path": nature_path,
                "md_path": md_path,
                "interaction_tsv_path": str(tsv_path),
                "map_path": str(map_path),
                "length": str(summary.length),
                "total_frames": str(summary.total_frames),
                "split": "",
                "cluster_id": "",
            }
        )
        emit_progress()
        if sample_limit is not None and len(rows) >= int(sample_limit):
            break

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MD_PAIRWISE_MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "manifest_path": str(manifest_path),
        "maps_dir": str(maps_root),
        "num_samples": len(rows),
        "skipped_no_md": skipped_no_md,
        "skipped_missing_tsv": skipped_missing_tsv,
        "skipped_errors": skipped_errors,
        "known_events": sum(item.known_events for item in summaries),
        "skipped_unknown_terms": sum(item.skipped_unknown_terms for item in summaries),
        "skipped_malformed_rows": sum(item.skipped_malformed_rows for item in summaries),
        "skipped_unmapped_residues": sum(item.skipped_unmapped_residues for item in summaries),
        "skipped_self_pairs": sum(item.skipped_self_pairs for item in summaries),
    }


class MDPairwiseTermsDataset:
    def __init__(
        self,
        samples: Sequence[MDPairwiseSample],
        *,
        min_seq_sep: int = 6,
    ) -> None:
        self.samples = list(samples)
        self.min_seq_sep = int(min_seq_sep)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        sample_limit: int | None = None,
        min_seq_sep: int = 6,
    ) -> "MDPairwiseTermsDataset":
        samples = load_md_pairwise_manifest(manifest_path, split=split, sample_limit=sample_limit)
        return cls(samples, min_seq_sep=min_seq_sep)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MDPairwiseExample:
        sample = self.samples[index]
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("MDPairwiseTermsDataset requires torch to load examples.") from exc
        payload = np.load(sample.map_path, allow_pickle=False)
        y_np = payload["y"].astype(np.float32, copy=False)
        m_all_np = payload["m_all"].astype(np.float32, copy=False)
        length = int(sample.length)
        if y_np.shape != (length, length, len(ENERGY_TERMS)):
            raise ValueError(f"Map {sample.map_path} y shape {y_np.shape} does not match length={length}.")
        if m_all_np.shape != (length, length):
            raise ValueError(f"Map {sample.map_path} m_all shape {m_all_np.shape} does not match length={length}.")
        residue_mask = torch.ones((length,), dtype=torch.bool)
        pair_mask = build_pair_mask(length, min_seq_sep=self.min_seq_sep, torch_module=torch)
        return MDPairwiseExample(
            sample_id=sample.sample_id,
            protein_id=sample.protein_id,
            sequence=sample.sequence,
            nature_path=sample.nature_path,
            md_path=sample.md_path,
            y=torch.from_numpy(y_np),
            m_all=torch.from_numpy(m_all_np),
            residue_mask=residue_mask,
            pair_mask=pair_mask,
            length=length,
        )


def load_md_pairwise_manifest(
    manifest_path: str | Path,
    *,
    split: str | None = None,
    sample_limit: int | None = None,
) -> list[MDPairwiseSample]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"MD pairwise manifest does not exist: {path}")
    samples: list[MDPairwiseSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_split = (row.get("split") or "").strip()
            if split is not None and row_split != split:
                continue
            sample = MDPairwiseSample(
                sample_id=(row.get("sample_id") or row.get("protein_id") or "").strip(),
                protein_id=(row.get("protein_id") or "").strip(),
                sequence=(row.get("sequence") or "").strip(),
                sequence_hash=(row.get("sequence_hash") or "").strip(),
                nature_path=(row.get("nature_path") or "").strip(),
                md_path=(row.get("md_path") or "").strip(),
                interaction_tsv_path=(row.get("interaction_tsv_path") or "").strip(),
                map_path=(row.get("map_path") or "").strip(),
                length=int(row.get("length") or 0),
                total_frames=int(row.get("total_frames") or 0),
                split=row_split,
                cluster_id=(row.get("cluster_id") or "").strip(),
            )
            if not sample.sample_id or not sample.protein_id or not sample.sequence or not sample.map_path:
                continue
            samples.append(sample)
            if sample_limit is not None and len(samples) >= int(sample_limit):
                break
    return samples


def build_pair_mask(length: int, *, min_seq_sep: int = 6, torch_module: Any | None = None) -> Any:
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    indices = torch_module.arange(int(length))
    row = indices.unsqueeze(1)
    col = indices.unsqueeze(0)
    return (col > row) & ((col - row) > int(min_seq_sep))


def md_pairwise_collate_fn(examples: Sequence[MDPairwiseExample]) -> MDPairwiseBatch:
    if not examples:
        raise ValueError("md_pairwise_collate_fn requires at least one example.")
    import torch

    batch_size = len(examples)
    max_length = max(example.length for example in examples)
    y = torch.zeros((batch_size, max_length, max_length, len(ENERGY_TERMS)), dtype=torch.float32)
    m_all = torch.zeros((batch_size, max_length, max_length), dtype=torch.float32)
    residue_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    pair_mask = torch.zeros((batch_size, max_length, max_length), dtype=torch.bool)
    lengths = torch.zeros((batch_size,), dtype=torch.long)

    for batch_index, example in enumerate(examples):
        length = int(example.length)
        y[batch_index, :length, :length, :] = example.y
        m_all[batch_index, :length, :length] = example.m_all
        residue_mask[batch_index, :length] = example.residue_mask
        pair_mask[batch_index, :length, :length] = example.pair_mask
        lengths[batch_index] = length

    return MDPairwiseBatch(
        sample_ids=[example.sample_id for example in examples],
        protein_ids=[example.protein_id for example in examples],
        sequences=[example.sequence for example in examples],
        nature_paths=[example.nature_path for example in examples],
        md_paths=[example.md_path for example in examples],
        y=y,
        m_all=m_all,
        residue_mask=residue_mask,
        pair_mask=pair_mask,
        lengths=lengths,
    )


def iter_batches(indices: Sequence[int], batch_size: int) -> Iterable[list[int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    for start in range(0, len(indices), batch_size):
        yield list(indices[start : start + batch_size])
