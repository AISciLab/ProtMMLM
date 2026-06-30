from __future__ import annotations

import importlib.util
import math
from typing import Any, Dict, List, Sequence


def has_sklearn_metrics() -> bool:
    try:
        return importlib.util.find_spec("sklearn.metrics") is not None
    except ModuleNotFoundError:
        return False


def has_scipy_stats() -> bool:
    try:
        return importlib.util.find_spec("scipy.stats") is not None
    except ModuleNotFoundError:
        return False


def binary_classification_metrics(
    labels: Sequence[float] | Sequence[int],
    predictions: Sequence[float] | Sequence[int],
    *,
    threshold: float = 0.5,
    prediction_score_space: str = "auto",
) -> Dict[str, float]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    if not labels:
        return {}

    true_values = [_to_binary_scalar(value, threshold=0.5) for value in labels]
    score_values = _normalize_classification_scores_1d(predictions, prediction_score_space=prediction_score_space)
    predicted_values = [_to_binary_scalar(value, threshold=threshold) for value in score_values]

    tp = sum(1 for label, prediction in zip(true_values, predicted_values) if label == 1 and prediction == 1)
    fp = sum(1 for label, prediction in zip(true_values, predicted_values) if label == 0 and prediction == 1)
    fn = sum(1 for label, prediction in zip(true_values, predicted_values) if label == 1 and prediction == 0)
    tn = sum(1 for label, prediction in zip(true_values, predicted_values) if label == 0 and prediction == 0)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, len(true_values))
    return {
        "roc_auc": _binary_roc_auc_score(true_values, score_values),
        "aupr": _binary_pr_auc_score(true_values, score_values),
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def multilabel_classification_metrics(
    labels: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
    prediction_score_space: str = "auto",
) -> Dict[str, float]:
    label_matrix = _ensure_matrix(labels, name="labels")
    prediction_matrix = _ensure_matrix(predictions, name="predictions")
    if len(label_matrix) != len(prediction_matrix):
        raise ValueError("labels and predictions must have the same number of rows.")
    if not label_matrix:
        return {}

    num_labels = len(label_matrix[0])
    for row in label_matrix:
        if len(row) != num_labels:
            raise ValueError("All multilabel label rows must have the same width.")
    for row in prediction_matrix:
        if len(row) != num_labels:
            raise ValueError("All multilabel prediction rows must have the same width.")

    normalized_prediction_matrix = _normalize_classification_scores_2d(
        prediction_matrix,
        prediction_score_space=prediction_score_space,
    )
    thresholded_labels = [
        [_to_binary_scalar(value, threshold=0.5) for value in row]
        for row in label_matrix
    ]
    thresholded_predictions = [
        [_to_binary_scalar(value, threshold=threshold) for value in row]
        for row in normalized_prediction_matrix
    ]

    reference_metrics = _multilabel_reference_metrics(
        thresholded_labels,
        thresholded_predictions,
    )
    return {
        "Precision": reference_metrics["Precision"],
        "Coverage": reference_metrics["Coverage"],
        "Accuracy": reference_metrics["Accuracy"],
        "Abs True": reference_metrics["Abs True"],
        "Abs False": reference_metrics["Abs False"],
    }


def multilabel_per_label_metrics(
    labels: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
    label_names: Sequence[str] | None = None,
    prediction_score_space: str = "auto",
) -> List[Dict[str, float | str | int]]:
    label_matrix = _ensure_matrix(labels, name="labels")
    prediction_matrix = _ensure_matrix(predictions, name="predictions")
    if len(label_matrix) != len(prediction_matrix):
        raise ValueError("labels and predictions must have the same number of rows.")
    if not label_matrix:
        return []

    num_labels = len(label_matrix[0])
    if label_names is None:
        resolved_label_names = [f"label_{index}" for index in range(num_labels)]
    else:
        resolved_label_names = [str(label_name) for label_name in label_names]
        if len(resolved_label_names) != num_labels:
            raise ValueError(
                f"label_names must contain {num_labels} names, got {len(resolved_label_names)}."
            )

    normalized_prediction_matrix = _normalize_classification_scores_2d(
        prediction_matrix,
        prediction_score_space=prediction_score_space,
    )
    thresholded_labels = [
        [_to_binary_scalar(value, threshold=0.5) for value in row]
        for row in label_matrix
    ]
    thresholded_predictions = [
        [_to_binary_scalar(value, threshold=threshold) for value in row]
        for row in normalized_prediction_matrix
    ]

    rows: List[Dict[str, float | str | int]] = []
    for label_index, label_name in enumerate(resolved_label_names):
        label_targets = [row[label_index] for row in thresholded_labels]
        label_predictions = [row[label_index] for row in thresholded_predictions]
        label_scores = [row[label_index] for row in normalized_prediction_matrix]
        tp = sum(1 for label, prediction in zip(label_targets, label_predictions) if label == 1 and prediction == 1)
        fp = sum(1 for label, prediction in zip(label_targets, label_predictions) if label == 0 and prediction == 1)
        fn = sum(1 for label, prediction in zip(label_targets, label_predictions) if label == 1 and prediction == 0)
        tn = sum(1 for label, prediction in zip(label_targets, label_predictions) if label == 0 and prediction == 0)

        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        precision = _safe_divide(tp, tp + fp)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        accuracy = _safe_divide(tp + tn, len(label_targets))
        roc_auc = _binary_roc_auc_score(label_targets, label_scores)
        aupr = _binary_pr_auc_score(label_targets, label_scores)
        rows.append(
            {
                "label_index": label_index,
                "label_name": label_name,
                "support": sum(label_targets),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "Recall": recall,
                "SN": recall,
                "SP": specificity,
                "MCC": _matthews_correlation_coefficient(tp=tp, fp=fp, fn=fn, tn=tn),
                "Precision": precision,
                "F1": f1,
                "Acc": accuracy,
                "AUC": roc_auc,
                "AUPR": aupr,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "acc": accuracy,
                "roc_auc": roc_auc,
                "aupr": aupr,
            }
        )
    return rows


def regression_metrics(
    labels: Sequence[float] | Sequence[int],
    predictions: Sequence[float] | Sequence[int],
) -> Dict[str, float]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    if not labels:
        return {}

    label_values = [float(value) for value in labels]
    prediction_values = [float(value) for value in predictions]
    absolute_errors = [abs(prediction - label) for label, prediction in zip(label_values, prediction_values)]
    squared_errors = [(prediction - label) ** 2 for label, prediction in zip(label_values, prediction_values)]
    errors = [prediction - label for label, prediction in zip(label_values, prediction_values)]
    mse = sum(squared_errors) / float(len(squared_errors))
    return {
        "MSE": mse,
        "MAE": sum(absolute_errors) / float(len(absolute_errors)),
        "RMSE": math.sqrt(mse),
        "R2": _r2_score(label_values, prediction_values),
        "Pearson": _pearson_correlation(label_values, prediction_values),
        "Spearman": _spearman_correlation(label_values, prediction_values),
    }


def compute_task_metrics(
    *,
    task_kind: str,
    labels: Sequence[Any],
    predictions: Sequence[Any],
    threshold: float = 0.5,
    prediction_score_space: str = "auto",
) -> Dict[str, float]:
    if task_kind == "binary":
        return binary_classification_metrics(
            labels,
            predictions,
            threshold=threshold,
            prediction_score_space=prediction_score_space,
        )
    if task_kind == "multilabel":
        return multilabel_classification_metrics(
            labels,
            predictions,
            threshold=threshold,
            prediction_score_space=prediction_score_space,
        )
    if task_kind == "regression":
        return regression_metrics(labels, predictions)
    raise ValueError(f"Unsupported task_kind: {task_kind}")


def _multilabel_reference_metrics(
    labels: Sequence[Sequence[int]],
    predictions: Sequence[Sequence[int]],
) -> Dict[str, float]:
    aiming_values: List[float] = []
    coverage_values: List[float] = []
    accuracy_values: List[float] = []
    absolute_true_values: List[float] = []
    absolute_false_values: List[float] = []
    for label_row, prediction_row in zip(labels, predictions):
        intersection = sum(
            1
            for label_value, prediction_value in zip(label_row, prediction_row)
            if label_value == 1 and prediction_value == 1
        )
        pred_count = sum(prediction_row)
        true_count = sum(label_row)
        union = sum(
            1
            for label_value, prediction_value in zip(label_row, prediction_row)
            if label_value == 1 or prediction_value == 1
        )
        mismatches = sum(
            1
            for label_value, prediction_value in zip(label_row, prediction_row)
            if label_value != prediction_value
        )
        aiming_values.append(_safe_divide(intersection, pred_count))
        coverage_values.append(_safe_divide(intersection, true_count))
        accuracy_values.append(1.0 if union == 0 else intersection / float(union))
        absolute_true_values.append(1.0 if mismatches == 0 else 0.0)
        absolute_false_values.append(_safe_divide(mismatches, len(label_row)))
    return {
        "Precision": sum(aiming_values) / float(len(aiming_values)),
        "Coverage": sum(coverage_values) / float(len(coverage_values)),
        "Accuracy": sum(accuracy_values) / float(len(accuracy_values)),
        "Abs True": sum(absolute_true_values) / float(len(absolute_true_values)),
        "Abs False": sum(absolute_false_values) / float(len(absolute_false_values)),
    }


def _ensure_matrix(values: Sequence[Sequence[float]], *, name: str) -> List[List[float]]:
    matrix = [[float(component) for component in row] for row in values]
    if not matrix:
        return []
    if not isinstance(matrix[0], list):
        raise ValueError(f"{name} must be a 2D sequence of floats.")
    return matrix


def _normalize_classification_scores_1d(
    values: Sequence[float] | Sequence[int],
    *,
    prediction_score_space: str,
) -> List[float]:
    score_values = [float(value) for value in values]
    if prediction_score_space == "probability":
        return score_values
    if prediction_score_space == "logit":
        return [_sigmoid(value) for value in score_values]
    if prediction_score_space != "auto":
        raise ValueError(
            f"Unsupported prediction_score_space={prediction_score_space!r}. "
            "Expected 'auto', 'probability', or 'logit'."
        )
    if _looks_like_probability_space(score_values):
        return score_values
    return [_sigmoid(value) for value in score_values]


def _normalize_classification_scores_2d(
    values: Sequence[Sequence[float]],
    *,
    prediction_score_space: str,
) -> List[List[float]]:
    matrix = _ensure_matrix(values, name="predictions")
    if not matrix:
        return []
    if prediction_score_space == "probability":
        return matrix
    if prediction_score_space == "logit":
        return [[_sigmoid(component) for component in row] for row in matrix]
    if prediction_score_space != "auto":
        raise ValueError(
            f"Unsupported prediction_score_space={prediction_score_space!r}. "
            "Expected 'auto', 'probability', or 'logit'."
        )
    flattened = [component for row in matrix for component in row]
    if _looks_like_probability_space(flattened):
        return matrix
    return [[_sigmoid(component) for component in row] for row in matrix]


def _looks_like_probability_space(values: Sequence[float]) -> bool:
    tolerance = 1e-6
    return all((-tolerance) <= value <= (1.0 + tolerance) for value in values)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _to_binary_scalar(value: float | int, *, threshold: float) -> int:
    return 1 if float(value) >= threshold else 0


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / float(denominator)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return 0.5 * (sorted_values[midpoint - 1] + sorted_values[midpoint])


def _r2_score(labels: Sequence[float], predictions: Sequence[float]) -> float:
    label_mean = sum(labels) / float(len(labels))
    total_sum_squares = sum((label - label_mean) ** 2 for label in labels)
    if total_sum_squares == 0.0:
        return 0.0
    residual_sum_squares = sum(
        (label - prediction) ** 2
        for label, prediction in zip(labels, predictions)
    )
    return 1.0 - residual_sum_squares / total_sum_squares


def _explained_variance_score(labels: Sequence[float], errors: Sequence[float]) -> float:
    label_variance = _population_variance(labels)
    if label_variance == 0.0:
        return 0.0
    return 1.0 - _population_variance(errors) / label_variance


def _population_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / float(len(values))
    return sum((value - mean_value) ** 2 for value in values) / float(len(values))


def _matthews_correlation_coefficient(*, tp: int, fp: int, fn: int, tn: int) -> float:
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denominator


def _binary_pr_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    if sum(labels) == 0:
        return 0.0
    if has_sklearn_metrics():
        from sklearn.metrics import average_precision_score  # type: ignore

        try:
            return _normalize_metric_value(average_precision_score(labels, scores))
        except ValueError:
            pass
    return _average_precision_fallback(labels, scores)


def _binary_roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(set(labels)) < 2:
        return 0.0
    if has_sklearn_metrics():
        from sklearn.metrics import roc_auc_score  # type: ignore

        try:
            return _normalize_metric_value(roc_auc_score(labels, scores))
        except ValueError:
            pass
    return _roc_auc_fallback(labels, scores)


def _average_precision_fallback(labels: Sequence[int], scores: Sequence[float]) -> float:
    total_positives = sum(labels)
    if total_positives == 0:
        return 0.0

    ranked_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    precision_sum = 0.0
    true_positives = 0
    for rank, index in enumerate(ranked_indices, start=1):
        if labels[index] == 1:
            true_positives += 1
            precision_sum += true_positives / float(rank)
    return precision_sum / float(total_positives)


def _roc_auc_fallback(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive_scores = [score for label, score in zip(labels, scores) if label == 1]
    negative_scores = [score for label, score in zip(labels, scores) if label == 0]
    if not positive_scores or not negative_scores:
        return 0.0

    wins = 0.0
    for positive_score in positive_scores:
        for negative_score in negative_scores:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / float(len(positive_scores) * len(negative_scores))


def _pearson_correlation(labels: Sequence[float], predictions: Sequence[float]) -> float:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    if not labels:
        return 0.0

    label_mean = sum(labels) / float(len(labels))
    prediction_mean = sum(predictions) / float(len(predictions))
    covariance = sum(
        (label - label_mean) * (prediction - prediction_mean)
        for label, prediction in zip(labels, predictions)
    )
    label_variance = sum((label - label_mean) ** 2 for label in labels)
    prediction_variance = sum((prediction - prediction_mean) ** 2 for prediction in predictions)
    if label_variance == 0.0 or prediction_variance == 0.0:
        return 0.0
    return covariance / math.sqrt(label_variance * prediction_variance)


def _spearman_correlation(labels: Sequence[float], predictions: Sequence[float]) -> float:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    if not labels:
        return 0.0

    if has_scipy_stats():
        from scipy.stats import spearmanr  # type: ignore

        correlation = spearmanr(labels, predictions).correlation
        return _normalize_metric_value(correlation)

    label_ranks = _rankdata_average(labels)
    prediction_ranks = _rankdata_average(predictions)
    return _pearson_correlation(label_ranks, prediction_ranks)


def _rankdata_average(values: Sequence[float]) -> List[float]:
    indexed_values = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed_values):
        end = start
        current_value = indexed_values[start][1]
        while end + 1 < len(indexed_values) and indexed_values[end + 1][1] == current_value:
            end += 1
        average_rank = (start + end + 2) / 2.0
        for position in range(start, end + 1):
            original_index = indexed_values[position][0]
            ranks[original_index] = average_rank
        start = end + 1
    return ranks


def _normalize_metric_value(raw_value: Any) -> float:
    value = float(raw_value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value
