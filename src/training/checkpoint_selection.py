from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_LOWER_IS_BETTER = frozenset({
    "loss",
    "mae",
    "mse",
    "rmse",
    "huber",
    "abs_false",
})

_HIGHER_IS_BETTER = frozenset({
    "accuracy",
    "acc",
    "precision",
    "recall",
    "f1",
    "f1_score",
    "auc",
    "roc_auc",
    "aupr",
    "r2",
    "pearson",
    "spearman",
    "coverage",
    "abs_true",
})


@dataclass
class EarlyStoppingState:
    monitor_name: str
    monitor_mode: str
    best_metric: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    should_stop: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EarlyStoppingState":
        return cls(
            monitor_name=str(payload.get("monitor_name", "loss")),
            monitor_mode=str(payload.get("monitor_mode", "min")),
            best_metric=_optional_float(payload.get("best_metric")),
            best_epoch=_optional_int(payload.get("best_epoch")),
            bad_epochs=int(payload.get("bad_epochs", 0)),
            should_stop=bool(payload.get("should_stop", False)),
            stop_reason=None if payload.get("stop_reason") is None else str(payload.get("stop_reason")),
        )


@dataclass
class Stage1SelectionState:
    early_stopping: EarlyStoppingState
    last_epoch: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "early_stopping": self.early_stopping.to_dict(),
            "last_epoch": self.last_epoch,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage1SelectionState":
        early_stopping_payload = payload.get("early_stopping")
        if isinstance(early_stopping_payload, Mapping):
            early_stopping = EarlyStoppingState.from_dict(early_stopping_payload)
        else:
            early_stopping = EarlyStoppingState(monitor_name="loss", monitor_mode="min")
        return cls(
            early_stopping=early_stopping,
            last_epoch=_optional_int(payload.get("last_epoch")),
        )


@dataclass
class DownstreamSelectionState:
    early_stopping: EarlyStoppingState
    last_epoch: int | None = None
    best_guarded_metric: float | None = None
    best_guarded_epoch: int | None = None
    best_guarded_seq_only_metric: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "early_stopping": self.early_stopping.to_dict(),
            "last_epoch": self.last_epoch,
            "best_guarded_metric": self.best_guarded_metric,
            "best_guarded_epoch": self.best_guarded_epoch,
            "best_guarded_seq_only_metric": self.best_guarded_seq_only_metric,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DownstreamSelectionState":
        early_stopping_payload = payload.get("early_stopping")
        if isinstance(early_stopping_payload, Mapping):
            early_stopping = EarlyStoppingState.from_dict(early_stopping_payload)
        else:
            early_stopping = EarlyStoppingState(monitor_name="loss", monitor_mode="min")
        return cls(
            early_stopping=early_stopping,
            last_epoch=_optional_int(payload.get("last_epoch")),
            best_guarded_metric=_optional_float(payload.get("best_guarded_metric")),
            best_guarded_epoch=_optional_int(payload.get("best_guarded_epoch")),
            best_guarded_seq_only_metric=_optional_float(payload.get("best_guarded_seq_only_metric")),
        )


def primary_monitor_name_for_task(task_kind: str) -> str:
    normalized = str(task_kind).strip().lower()
    if normalized == "binary":
        return "aupr"
    if normalized == "regression":
        return "RMSE"
    if normalized == "multilabel":
        return "Accuracy"
    raise ValueError(f"Unsupported task_kind for monitor selection: {task_kind!r}.")



def monitor_mode_for_metric(monitor_name: str) -> str:
    normalized = _normalize_metric_name(monitor_name)
    if normalized in _LOWER_IS_BETTER:
        return "min"
    if normalized in _HIGHER_IS_BETTER:
        return "max"
    return "max"



def extract_metric(
    report: Mapping[str, Any] | None,
    *,
    subset: str,
    monitor_name: str,
) -> float | None:
    if report is None:
        return None
    subset_payload = report.get(subset)
    if not isinstance(subset_payload, Mapping):
        return None
    metrics_payload = subset_payload.get("metrics")
    if not isinstance(metrics_payload, Mapping):
        return None
    return _lookup_metric(metrics_payload, monitor_name)



def is_better(
    candidate: float | None,
    best: float | None,
    *,
    mode: str,
    min_delta: float = 0.0,
) -> bool:
    if candidate is None:
        return False
    if best is None:
        return True
    if str(mode).strip().lower() == "min":
        return candidate < (best - min_delta)
    return candidate > (best + min_delta)



def significantly_worse(
    candidate: float | None,
    reference: float | None,
    *,
    mode: str,
    tolerance: float = 0.0,
) -> bool:
    if candidate is None or reference is None:
        return False
    if str(mode).strip().lower() == "min":
        return candidate > (reference + tolerance)
    return candidate < (reference - tolerance)



def resolve_checkpoint_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path.parent if path.suffix else path



def _lookup_metric(metrics: Mapping[str, Any], monitor_name: str) -> float | None:
    direct_value = metrics.get(monitor_name)
    if direct_value is not None:
        return _optional_float(direct_value)

    normalized_target = _normalize_metric_name(monitor_name)
    for key, value in metrics.items():
        if _normalize_metric_name(str(key)) == normalized_target:
            return _optional_float(value)
    return None



def _normalize_metric_name(metric_name: str) -> str:
    return "_".join(str(metric_name).strip().lower().replace("-", " ").split())



def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)



def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
