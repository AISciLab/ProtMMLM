from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.datasets.downstream_dataset import parse_target, task_kind_for_name
from src.evaluation.metrics import (
    compute_task_metrics,
    multilabel_per_label_metrics,
    _normalize_classification_scores_2d,
)


PRMFTP_LABEL_NAMES = (
    "AAP",
    "ABP",
    "ACP",
    "ACVP",
    "ADP",
    "AEP",
    "AFP",
    "AHIVP",
    "AHP",
    "AIP",
    "AMRSAP",
    "APP",
    "ATP",
    "AVP",
    "BBP",
    "BIP",
    "CPP",
    "DPPIP",
    "QSP",
    "SBP",
    "THP",
)


MODALITY_SUBSETS = frozenset({"seq_only", "nature_only", "partial", "full"})


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    label: Any
    prediction: Any
    task_name: str
    has_dyn: bool = False
    modality_subset: str | None = None
    seq_only_prediction: Any | None = None
    full_prediction: Any | None = None

    def __post_init__(self) -> None:
        modality_subset = _normalize_modality_subset(self.modality_subset, has_dyn=self.has_dyn)
        object.__setattr__(self, "modality_subset", modality_subset)
        object.__setattr__(self, "has_dyn", modality_subset == "full")

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, Any],
        *,
        task_name: Optional[str] = None,
    ) -> "EvaluationRecord":
        resolved_task_name = _normalize_task_name(task_name or mapping.get("task_name"))
        task_kind = task_kind_for_name(resolved_task_name)
        label = _coerce_target(mapping.get("label"), task_kind=task_kind)
        prediction = _coerce_prediction(
            mapping.get("prediction", mapping.get("full_prediction", mapping.get("seq_only_prediction"))),
            task_kind=task_kind,
        )

        seq_only_prediction = None
        if mapping.get("seq_only_prediction") is not None:
            seq_only_prediction = _coerce_prediction(mapping.get("seq_only_prediction"), task_kind=task_kind)

        full_prediction = None
        if mapping.get("full_prediction") is not None:
            full_prediction = _coerce_prediction(mapping.get("full_prediction"), task_kind=task_kind)

        return cls(
            sample_id=str(mapping.get("sample_id") or "").strip(),
            label=label,
            prediction=prediction,
            task_name=resolved_task_name,
            has_dyn=_parse_bool(mapping.get("has_dyn")),
            modality_subset=mapping.get("modality_subset"),
            seq_only_prediction=seq_only_prediction,
            full_prediction=full_prediction,
        )


class ProtMMLMEvaluator:
    def __init__(
        self,
        *,
        task_name: str,
        threshold: float = 0.5,
        prediction_score_space: str = "auto",
    ) -> None:
        self.task_name = _normalize_task_name(task_name)
        self.task_kind = task_kind_for_name(self.task_name)
        self.threshold = threshold
        self.prediction_score_space = prediction_score_space

    def evaluate(
        self,
        records: Sequence[EvaluationRecord],
        *,
        subset_names: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        normalized_records = list(records)
        if not normalized_records:
            raise ValueError("evaluate requires at least one evaluation record.")
        for record in normalized_records:
            if _normalize_task_name(record.task_name) != self.task_name:
                raise ValueError(
                    f"Record task_name={record.task_name!r} does not match evaluator task_name={self.task_name!r}."
                )

        requested_subsets = _resolve_subset_names(subset_names)
        seq_only_records = [record for record in normalized_records if record.modality_subset == "seq_only"]
        nature_only_records = [record for record in normalized_records if record.modality_subset == "nature_only"]
        partial_records = [record for record in normalized_records if record.modality_subset == "partial"]
        full_records = [record for record in normalized_records if record.modality_subset == "full"]
        matched_full_records = [
            record
            for record in full_records
            if record.seq_only_prediction is not None and (record.full_prediction is not None or record.prediction is not None)
        ]

        report = {
            "task_name": self.task_name,
            "task_kind": self.task_kind,
        }
        if "overall" in requested_subsets:
            report["overall"] = self._evaluate_subset(normalized_records, prediction_field="prediction")
        if "seq_only" in requested_subsets:
            report["seq_only"] = self._evaluate_subset(seq_only_records, prediction_field="prediction")
        if "nature_only" in requested_subsets:
            report["nature_only"] = self._evaluate_subset(nature_only_records, prediction_field="prediction")
        if "partial" in requested_subsets:
            report["partial"] = self._evaluate_subset(partial_records, prediction_field="prediction")
        if "full" in requested_subsets:
            report["full"] = self._evaluate_subset(full_records, prediction_field="prediction")
        if "matched_full" in requested_subsets:
            matched_full_seq_predictions = [record.seq_only_prediction for record in matched_full_records]
            matched_full_predictions = [
                record.full_prediction if record.full_prediction is not None else record.prediction
                for record in matched_full_records
            ]
            matched_full_labels = [record.label for record in matched_full_records]
            matched_full_seq_metrics = self._compute_metrics(matched_full_labels, matched_full_seq_predictions)
            matched_full_full_metrics = self._compute_metrics(matched_full_labels, matched_full_predictions)
            report["matched_full"] = {
                "num_samples": len(matched_full_records),
                "seq_only": {
                    "num_samples": len(matched_full_records),
                    "metrics": matched_full_seq_metrics,
                },
                "full": {
                    "num_samples": len(matched_full_records),
                    "metrics": matched_full_full_metrics,
                },
                "delta_full_minus_seq": _metric_delta(
                    matched_full_full_metrics,
                    matched_full_seq_metrics,
                ),
            }
        if self.task_kind == "multilabel":
            per_label_metrics = self._compute_per_label_metrics(
                [record.label for record in normalized_records],
                [record.prediction for record in normalized_records],
            )
            report["per_label_metrics"] = per_label_metrics
            report["multilabel_summary"] = self._summarize_multilabel_predictions(
                [record.prediction for record in normalized_records],
                per_label_metrics=per_label_metrics,
            )
        return report

    def save_report(self, report: Dict[str, Any], output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def format_report(self, report: Dict[str, Any]) -> str:
        lines = [
            f"task={report['task_name']}",
            f"task_kind={report['task_kind']}",
        ]
        for subset_name in _ordered_subset_names(report):
            subset = report[subset_name]
            lines.append(f"{subset_name}.num_samples={subset['num_samples']}")
            for metric_name in sorted(subset["metrics"]):
                lines.append(f"{subset_name}.{metric_name}={subset['metrics'][metric_name]}")
        matched_full = report.get("matched_full")
        if matched_full is not None:
            lines.append(f"matched_full.num_samples={matched_full['num_samples']}")
            for variant_name in ("seq_only", "full"):
                variant = matched_full[variant_name]
                for metric_name in sorted(variant["metrics"]):
                    lines.append(f"matched_full.{variant_name}.{metric_name}={variant['metrics'][metric_name]}")
            for metric_name in sorted(matched_full["delta_full_minus_seq"]):
                lines.append(
                    "matched_full.delta_full_minus_seq."
                    f"{metric_name}={matched_full['delta_full_minus_seq'][metric_name]}"
                )
        return "\n".join(lines)

    def _evaluate_subset(
        self,
        records: Sequence[EvaluationRecord],
        *,
        prediction_field: str,
    ) -> Dict[str, Any]:
        if not records:
            return {"num_samples": 0, "metrics": {}}

        labels = [record.label for record in records]
        predictions = [getattr(record, prediction_field) for record in records]
        metrics = self._compute_metrics(labels, predictions)
        if self.task_kind == "multilabel":
            per_label_metrics = self._compute_per_label_metrics(labels, predictions)
            metrics.update(self._multilabel_scalar_summary(predictions, per_label_metrics=per_label_metrics))
        return {
            "num_samples": len(records),
            "metrics": metrics,
        }

    def _compute_metrics(
        self,
        labels: Sequence[Any],
        predictions: Sequence[Any],
    ) -> Dict[str, float]:
        if not labels:
            return {}
        return compute_task_metrics(
            task_kind=self.task_kind,
            labels=labels,
            predictions=predictions,
            threshold=self.threshold,
            prediction_score_space=self.prediction_score_space,
        )

    def _compute_per_label_metrics(
        self,
        labels: Sequence[Any],
        predictions: Sequence[Any],
    ) -> List[Dict[str, float | str | int]]:
        if self.task_kind != "multilabel" or not labels:
            return []
        num_labels = len(labels[0])
        label_names = (
            PRMFTP_LABEL_NAMES
            if self.task_name == "prmftp" and len(PRMFTP_LABEL_NAMES) == num_labels
            else tuple(f"label_{index}" for index in range(num_labels))
        )
        return multilabel_per_label_metrics(
            labels,
            predictions,
            threshold=self.threshold,
            label_names=label_names,
            prediction_score_space=self.prediction_score_space,
        )

    def _summarize_multilabel_predictions(
        self,
        predictions: Sequence[Any],
        *,
        per_label_metrics: Sequence[Dict[str, float | str | int]],
    ) -> Dict[str, Any]:
        if not predictions:
            return {}
        score_rows = _normalize_classification_scores_2d(
            predictions,
            prediction_score_space=self.prediction_score_space,
        )
        scalar_summary = self._multilabel_scalar_summary(
            predictions,
            per_label_metrics=per_label_metrics,
        )
        per_label_fraction_ge_threshold = [
            {
                "label_index": label_index,
                "label_name": str(per_label_metrics[label_index].get("label_name", f"label_{label_index}"))
                if label_index < len(per_label_metrics)
                else f"label_{label_index}",
                "fraction_pred_ge_threshold": sum(
                    1 for row in score_rows if row[label_index] >= self.threshold
                ) / float(len(score_rows)),
            }
            for label_index in range(len(score_rows[0]))
        ]
        return {
            **scalar_summary,
            "per_label_fraction_pred_ge_0.5": per_label_fraction_ge_threshold,
        }

    def _multilabel_scalar_summary(
        self,
        predictions: Sequence[Any],
        *,
        per_label_metrics: Sequence[Dict[str, float | str | int]],
    ) -> Dict[str, float]:
        if not predictions:
            return {}
        score_rows = _normalize_classification_scores_2d(
            predictions,
            prediction_score_space=self.prediction_score_space,
        )
        flattened_scores = [score for row in score_rows for score in row]
        if not flattened_scores:
            return {}
        above_threshold_count = sum(1 for score in flattened_scores if score >= self.threshold)
        sorted_scores = sorted(flattened_scores)
        midpoint = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 1:
            median_score = sorted_scores[midpoint]
        else:
            median_score = 0.5 * (sorted_scores[midpoint - 1] + sorted_scores[midpoint])
        mean_label_aupr = 0.0 if not per_label_metrics else sum(float(row.get("aupr", 0.0)) for row in per_label_metrics) / float(len(per_label_metrics))
        mean_label_auc = 0.0 if not per_label_metrics else sum(float(row.get("roc_auc", 0.0)) for row in per_label_metrics) / float(len(per_label_metrics))
        return {
            "mean_pred_score": sum(flattened_scores) / float(len(flattened_scores)),
            "median_pred_score": median_score,
            "fraction_pred_ge_0.5": above_threshold_count / float(len(flattened_scores)),
            "mean_label_aupr": mean_label_aupr,
            "mean_label_auc": mean_label_auc,
        }


def load_evaluation_records(
    predictions_path: str | Path,
    *,
    task_name: Optional[str] = None,
) -> List[EvaluationRecord]:
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file does not exist: {path}")

    raw_items: List[Dict[str, Any]]
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload_task_name = payload.get("task_name")
            task_name = task_name or payload_task_name
            raw_records = payload.get("records", [])
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raise ValueError(f"Unsupported JSON payload in {path}: expected list or object.")
        raw_items = [dict(item) for item in raw_records]
    elif path.suffix.lower() == ".jsonl":
        raw_items = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                loaded = json.loads(stripped)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Invalid JSONL object at {path}:{line_number}")
                raw_items.append(dict(loaded))
    else:
        raise ValueError(f"Unsupported predictions file suffix for {path}. Expected .json or .jsonl")

    if not raw_items:
        raise ValueError(f"No evaluation records were loaded from {path}")
    return [
        EvaluationRecord.from_mapping(item, task_name=task_name)
        for item in raw_items
    ]


def _resolve_subset_names(subset_names: Sequence[str] | None) -> tuple[str, ...]:
    if subset_names is None:
        return ("overall", "seq_only", "nature_only", "partial", "full", "matched_full")
    resolved: list[str] = []
    for raw_name in subset_names:
        normalized = str(raw_name).strip().lower()
        if normalized not in {"overall", "seq_only", "nature_only", "partial", "full", "matched_full"}:
            raise ValueError(f"Unsupported report subset {raw_name!r}.")
        if normalized not in resolved:
            resolved.append(normalized)
    if not resolved:
        raise ValueError("subset_names must contain at least one subset.")
    return tuple(resolved)


def _ordered_subset_names(report: Dict[str, Any]) -> list[str]:
    return [
        subset_name
        for subset_name in ("overall", "seq_only", "nature_only", "partial", "full")
        if subset_name in report
    ]


def _normalize_modality_subset(raw_value: Any, *, has_dyn: bool) -> str:
    if raw_value is None:
        return "full" if has_dyn else "seq_only"
    normalized = str(raw_value).strip().lower()
    if normalized not in MODALITY_SUBSETS:
        raise ValueError(
            f"Unsupported modality_subset {raw_value!r}. Expected one of {sorted(MODALITY_SUBSETS)}."
        )
    return normalized


def _metric_delta(full_metrics: Dict[str, float], seq_metrics: Dict[str, float]) -> Dict[str, float]:
    shared_metrics = sorted(set(full_metrics) & set(seq_metrics))
    return {
        metric_name: full_metrics[metric_name] - seq_metrics[metric_name]
        for metric_name in shared_metrics
    }


def _coerce_target(raw_value: Any, *, task_kind: str) -> Any:
    if task_kind in {"binary", "regression"}:
        return float(raw_value)
    if isinstance(raw_value, list):
        return [float(value) for value in raw_value]
    return parse_target(str(raw_value), task_kind=task_kind)


def _coerce_prediction(raw_value: Any, *, task_kind: str) -> Any:
    if task_kind in {"binary", "regression"}:
        return float(raw_value)
    if isinstance(raw_value, list):
        return [float(value) for value in raw_value]
    return _parse_prediction_vector(str(raw_value))


def _parse_prediction_vector(raw_value: str) -> List[float]:
    value = raw_value.strip()
    if not value:
        raise ValueError("Prediction vector cannot be empty.")
    if "," in value:
        return [float(component.strip()) for component in value.split(",") if component.strip()]
    if " " in value:
        return [float(component) for component in value.split() if component]
    if set(value).issubset({"0", "1"}):
        return [float(character) for character in value]
    raise ValueError(f"Unsupported prediction vector format: {raw_value}")


def _normalize_task_name(task_name: str | None) -> str:
    if task_name is None:
        raise ValueError("task_name is required.")
    normalized = str(task_name).strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("task_name cannot be empty.")
    return normalized


def _parse_bool(raw_value: Any) -> bool:
    return str(raw_value).strip().lower() in {"1", "true", "yes"}
