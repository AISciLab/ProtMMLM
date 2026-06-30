from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from src.datasets.downstream_adapters import DownstreamSample
from src.datasets.pretrain_dataset import PretrainHashIndex, PretrainMatchRecord
from src.utils.sequence import sequence_hash


TASK_KIND_BY_NAME = {
    "toxteller": "binary",
    "conotoxin": "binary",
    "prmftp": "multilabel",
    "ppikb": "regression",
}


TargetType = Union[float, List[float]]


@dataclass(frozen=True)
class DownstreamStage1Sample:
    sample_id: str
    sequence: str
    sequence_hash: str
    target: TargetType
    raw_label: str
    task_name: str
    task_kind: str
    peptide_sequence: Optional[str] = None
    peptide_sequence_hash: Optional[str] = None
    pair_key: Optional[str] = None
    split: Optional[str] = None
    matched_pretrain_id: Optional[str] = None
    nature_path: Optional[str] = None
    md_path: Optional[str] = None
    has_dyn: bool = False
    dyn_cache_path: Optional[str] = None
    peptide_matched_pretrain_id: Optional[str] = None
    peptide_nature_path: Optional[str] = None
    peptide_md_path: Optional[str] = None
    peptide_has_dyn: bool = False
    peptide_dyn_cache_path: Optional[str] = None

    def to_training_fields(self) -> Dict[str, Union[str, float, List[float]]]:
        return {
            "sample_id": self.sample_id,
            "sequence": self.sequence,
            "sequence_hash": self.sequence_hash,
            "peptide_sequence": self.peptide_sequence or "",
            "peptide_sequence_hash": self.peptide_sequence_hash or "",
            "pair_key": self.pair_key or "",
            "target": self.target,
            "task_name": self.task_name,
            "task_kind": self.task_kind,
            "split": self.split or "",
            "matched_pretrain_id": self.matched_pretrain_id or "",
            "nature_path": self.nature_path or "",
            "md_path": self.md_path or "",
            "has_dyn": self.has_dyn,
            "dyn_cache_path": self.dyn_cache_path or "",
            "peptide_matched_pretrain_id": self.peptide_matched_pretrain_id or "",
            "peptide_nature_path": self.peptide_nature_path or "",
            "peptide_md_path": self.peptide_md_path or "",
            "peptide_has_dyn": self.peptide_has_dyn,
            "peptide_dyn_cache_path": self.peptide_dyn_cache_path or "",
        }

    @property
    def is_pair_sample(self) -> bool:
        return bool(self.peptide_sequence)

    @property
    def has_any_dyn(self) -> bool:
        return bool(self.has_dyn or self.peptide_has_dyn)


class DownstreamDataset:
    def __init__(self, samples: List[DownstreamStage1Sample], *, task_name: str, task_kind: str) -> None:
        self.samples = list(samples)
        self.task_name = task_name
        self.task_kind = task_kind

    @classmethod
    def from_samples_with_pretrain_index(
        cls,
        samples: List[DownstreamSample],
        *,
        task_name: str,
        pretrain_index: PretrainHashIndex,
    ) -> "DownstreamDataset":
        normalized_task_name = _normalize_task_name(task_name)
        task_kind = task_kind_for_name(normalized_task_name)
        dataset_samples = [
            _build_stage1_sample(sample, task_kind=task_kind, pretrain_index=pretrain_index)
            for sample in samples
        ]
        return cls(dataset_samples, task_name=normalized_task_name, task_kind=task_kind)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        task_name: Optional[str] = None,
        sample_limit: Optional[int] = None,
    ) -> "DownstreamDataset":
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Downstream manifest does not exist: {path}")
        manifest_dir = path.resolve().parent

        samples: List[DownstreamStage1Sample] = []
        normalized_task_name = None if task_name is None else _normalize_task_name(task_name)
        inferred_task_name: Optional[str] = None

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_task_name = _normalize_task_name(row.get("task_name"))
                if normalized_task_name is not None and row_task_name != normalized_task_name:
                    continue

                if inferred_task_name is None:
                    inferred_task_name = row_task_name
                elif inferred_task_name != row_task_name:
                    raise ValueError(
                        "DownstreamDataset.from_manifest expected a single task per manifest. "
                        f"Observed {inferred_task_name!r} and {row_task_name!r}."
                    )

                task_kind = task_kind_for_name(row_task_name)
                raw_label = (row.get("label") or "").strip()
                samples.append(
                    DownstreamStage1Sample(
                        sample_id=(row.get("sample_id") or "").strip(),
                        sequence=(row.get("sequence") or "").strip(),
                        sequence_hash=(row.get("sequence_hash") or "").strip(),
                        target=parse_target(raw_label, task_kind=task_kind),
                        raw_label=raw_label,
                        task_name=row_task_name,
                        task_kind=task_kind,
                        peptide_sequence=_normalize_optional_value(row.get("peptide_sequence")),
                        peptide_sequence_hash=_normalize_optional_value(row.get("peptide_sequence_hash")),
                        pair_key=_normalize_optional_value(row.get("pair_key")),
                        split=_normalize_split_name(row.get("split")),
                        matched_pretrain_id=_normalize_optional_value(row.get("matched_pretrain_id")),
                        nature_path=_normalize_optional_path(row.get("nature_path"), base_dir=manifest_dir),
                        md_path=_normalize_optional_path(row.get("md_path"), base_dir=manifest_dir),
                        has_dyn=_parse_bool(row.get("has_dyn")),
                        dyn_cache_path=_normalize_optional_path(row.get("dyn_cache_path"), base_dir=manifest_dir),
                        peptide_matched_pretrain_id=_normalize_optional_value(
                            row.get("peptide_matched_pretrain_id")
                        ),
                        peptide_nature_path=_normalize_optional_path(row.get("peptide_nature_path"), base_dir=manifest_dir),
                        peptide_md_path=_normalize_optional_path(row.get("peptide_md_path"), base_dir=manifest_dir),
                        peptide_has_dyn=_parse_bool(row.get("peptide_has_dyn")),
                        peptide_dyn_cache_path=_normalize_optional_path(
                            row.get("peptide_dyn_cache_path"), base_dir=manifest_dir
                        ),
                    )
                )
                if sample_limit is not None and len(samples) >= sample_limit:
                    break

        if not samples:
            raise ValueError(f"No samples were loaded from {path} for task {normalized_task_name or '<inferred>'}.")

        resolved_task_name = normalized_task_name or inferred_task_name
        assert resolved_task_name is not None
        return cls(
            samples,
            task_name=resolved_task_name,
            task_kind=task_kind_for_name(resolved_task_name),
        )

    def summary(self) -> Dict[str, Union[str, int]]:
        num_pair_samples = sum(int(sample.is_pair_sample) for sample in self.samples)
        num_pair_any_dyn_samples = sum(int(sample.has_any_dyn) for sample in self.samples if sample.is_pair_sample)
        num_pair_both_dyn_samples = sum(
            int(sample.has_dyn and sample.peptide_has_dyn)
            for sample in self.samples
            if sample.is_pair_sample
        )
        num_pair_partial_dyn_samples = sum(
            int(sample.has_any_dyn and not (sample.has_dyn and sample.peptide_has_dyn))
            for sample in self.samples
            if sample.is_pair_sample
        )
        num_single_full_samples = sum(int(sample.has_dyn) for sample in self.samples if not sample.is_pair_sample)
        num_single_seq_only_samples = sum(int(not sample.has_dyn) for sample in self.samples if not sample.is_pair_sample)
        num_full_samples = num_pair_both_dyn_samples + num_single_full_samples
        num_partial_samples = num_pair_partial_dyn_samples
        num_seq_only_samples = num_pair_samples - num_pair_any_dyn_samples + num_single_seq_only_samples
        summary: Dict[str, Union[str, int]] = {
            "num_samples": len(self.samples),
            "task_name": self.task_name,
            "task_kind": self.task_kind,
            "num_full_samples": num_full_samples,
            "num_partial_samples": num_partial_samples,
            "num_seq_only_samples": num_seq_only_samples,
        }
        if num_pair_samples:
            summary["num_pair_samples"] = num_pair_samples
            summary["num_pair_any_dyn_samples"] = num_pair_any_dyn_samples
            summary["num_pair_both_dyn_samples"] = num_pair_both_dyn_samples
            summary["num_pair_partial_dyn_samples"] = num_pair_partial_dyn_samples
        split_counts: Dict[str, int] = {}
        for sample in self.samples:
            if not sample.split:
                continue
            split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        for split_name, count in sorted(split_counts.items()):
            summary[f"split_{split_name}_samples"] = count
        return summary

    # def limit(self, sample_limit: Optional[int]) -> "DownstreamDataset":
    #     if sample_limit is None:
    #         return DownstreamDataset(list(self.samples), task_name=self.task_name, task_kind=self.task_kind)
    #     return DownstreamDataset(
    #         list(self.samples[:sample_limit]),
    #         task_name=self.task_name,
    #         task_kind=self.task_kind,
    #     )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DownstreamStage1Sample:
        return self.samples[index]

    def __iter__(self) -> Iterator[DownstreamStage1Sample]:
        return iter(self.samples)


def task_kind_for_name(task_name: str | None) -> str:
    normalized = _normalize_task_name(task_name)
    if normalized not in TASK_KIND_BY_NAME:
        raise ValueError(
            f"Unsupported downstream task {normalized!r}. Expected one of {sorted(TASK_KIND_BY_NAME)}."
        )
    return TASK_KIND_BY_NAME[normalized]


def parse_target(raw_label: str, *, task_kind: str) -> TargetType:
    if task_kind == "binary":
        return float(raw_label)
    if task_kind == "regression":
        return float(raw_label)
    if task_kind == "multilabel":
        return _parse_multilabel_target(raw_label)
    raise ValueError(f"Unsupported task_kind: {task_kind}")


def _parse_multilabel_target(raw_label: str) -> List[float]:
    label = raw_label.strip()
    if not label:
        raise ValueError("Multilabel target cannot be empty.")

    if "," in label:
        return [float(value.strip()) for value in label.split(",") if value.strip()]
    if " " in label:
        return [float(value) for value in label.split() if value]
    if set(label).issubset({"0", "1"}):
        return [float(character) for character in label]

    # Extension point: add richer multilabel parsing only if a retained downstream task requires a schema beyond binary indicator strings or flat numeric lists.
    raise ValueError(f"Unsupported multilabel target format: {raw_label}")


def _match_pretrain_record(
    sequence_hash_value: str,
    pretrain_index: PretrainHashIndex,
) -> PretrainMatchRecord | None:
    unique_match = pretrain_index.unique_matches.get(sequence_hash_value)
    if unique_match is not None:
        return unique_match
    ambiguous_matches = pretrain_index.ambiguous_matches.get(sequence_hash_value)
    if not ambiguous_matches:
        return None
    full_matches = [match for match in ambiguous_matches if match.has_dyn and match.nature_path and match.md_path]
    if len(full_matches) == 1:
        return full_matches[0]
    return None


def _build_stage1_sample(
    sample: DownstreamSample,
    *,
    task_kind: str,
    pretrain_index: PretrainHashIndex,
) -> DownstreamStage1Sample:
    protein_sequence_hash = sequence_hash(sample.sequence)
    protein_match = _match_pretrain_record(protein_sequence_hash, pretrain_index)
    peptide_sequence = sample.peptide_sequence
    peptide_sequence_hash = sequence_hash(peptide_sequence) if peptide_sequence else None
    peptide_match = _match_pretrain_record(peptide_sequence_hash, pretrain_index) if peptide_sequence_hash else None
    return DownstreamStage1Sample(
        sample_id=sample.sample_id,
        sequence=sample.sequence,
        sequence_hash=protein_sequence_hash,
        target=parse_target(sample.label, task_kind=task_kind),
        raw_label=sample.label,
        task_name=_normalize_task_name(sample.task_name),
        task_kind=task_kind,
        peptide_sequence=peptide_sequence,
        peptide_sequence_hash=peptide_sequence_hash,
        pair_key=sample.pair_key,
        split=_normalize_split_name(sample.split),
        matched_pretrain_id=None if protein_match is None else protein_match.protein_id,
        nature_path=None if protein_match is None else protein_match.nature_path,
        md_path=None if protein_match is None else protein_match.md_path,
        has_dyn=bool(protein_match and protein_match.has_dyn),
        peptide_matched_pretrain_id=None if peptide_match is None else peptide_match.protein_id,
        peptide_nature_path=None if peptide_match is None else peptide_match.nature_path,
        peptide_md_path=None if peptide_match is None else peptide_match.md_path,
        peptide_has_dyn=bool(peptide_match and peptide_match.has_dyn),
    )


def _normalize_task_name(task_name: str | None) -> str:
    if task_name is None:
        raise ValueError("task_name is required.")
    normalized = str(task_name).strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("task_name cannot be empty.")
    return normalized


def _normalize_optional_value(raw_value: object) -> Optional[str]:
    value = str(raw_value or "").strip()
    return value or None


def _normalize_optional_path(raw_value: object, *, base_dir: Path) -> Optional[str]:
    value = _normalize_optional_value(raw_value)
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    manifest_candidate = (base_dir / path).resolve()
    if manifest_candidate.exists():
        return str(manifest_candidate)
    repo_candidate = (_repo_root() / path).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    return str(repo_candidate)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_split_name(raw_value: object) -> Optional[str]:
    value = str(raw_value or "").strip().lower()
    if not value:
        return None
    if value in {"train", "training"}:
        return "train"
    if value in {"val", "valid", "validation", "dev"}:
        return "validation"
    if value == "test":
        return "test"
    raise ValueError(f"Unsupported downstream split value: {raw_value!r}")


def _parse_bool(raw_value: object) -> bool:
    return str(raw_value).strip().lower() in {"1", "true", "yes"}
