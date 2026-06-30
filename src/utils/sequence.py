from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator


# Conservative amino-acid alphabet for research data ingestion.
# Extension point: extend only when a concrete upstream dataset requires extra symbols.
VALID_SEQUENCE_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass(frozen=True)
class FastaRecord:
    header: str
    protein_id: str
    sequence: str


def normalize_sequence(raw_sequence: str) -> str:
    normalized = "".join(raw_sequence.split()).upper()
    if not normalized:
        raise ValueError("Sequence is empty after normalization.")

    invalid_chars = sorted({char for char in normalized if char not in VALID_SEQUENCE_CHARS})
    if invalid_chars:
        invalid_display = "".join(invalid_chars)
        raise ValueError(
            f"Sequence contains invalid residue characters: {invalid_display}"
        )

    return normalized


def sequence_hash(sequence: str) -> str:
    normalized = normalize_sequence(sequence)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def iter_fasta_records(path: str | Path, limit: int | None = None) -> Iterator[FastaRecord]:
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file does not exist: {fasta_path}")
    if not fasta_path.is_file():
        raise ValueError(f"FASTA path is not a file: {fasta_path}")

    current_header: str | None = None
    current_sequence_lines: list[str] = []
    yielded_records = 0

    def flush_record() -> FastaRecord | None:
        nonlocal current_header, current_sequence_lines
        if current_header is None:
            return None

        protein_id = current_header.split(None, 1)[0]
        if not protein_id:
            raise ValueError(f"FASTA record has an empty header: {fasta_path}")

        record = FastaRecord(
            header=current_header,
            protein_id=protein_id,
            sequence=normalize_sequence("".join(current_sequence_lines)),
        )
        current_header = None
        current_sequence_lines = []
        return record

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith(">"):
                record = flush_record()
                if record is not None:
                    yield record
                    yielded_records += 1
                    if limit is not None and yielded_records >= limit:
                        return

                current_header = stripped[1:].strip()
                current_sequence_lines = []
                continue

            if current_header is None:
                raise ValueError(
                    f"FASTA sequence line appears before a header at "
                    f"{fasta_path}:{line_number}"
                )

            current_sequence_lines.append(stripped)

    record = flush_record()
    if record is not None:
        if limit is None or yielded_records < limit:
            yield record
