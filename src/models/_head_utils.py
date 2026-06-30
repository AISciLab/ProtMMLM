from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence


Matrix = List[List[float]]
Vector = List[float]


def ensure_matrix(values: Sequence[float] | Sequence[Sequence[float]], *, name: str) -> Matrix:
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    first = values[0]  # type: ignore[index]
    if isinstance(first, (list, tuple)):
        matrix = [[float(component) for component in row] for row in values]  # type: ignore[arg-type]
    else:
        matrix = [[float(component) for component in values]]  # type: ignore[arg-type]

    feature_dim = len(matrix[0])
    if feature_dim == 0:
        raise ValueError(f"{name} rows cannot be empty.")
    for row in matrix:
        if len(row) != feature_dim:
            raise ValueError(f"{name} must have consistent row lengths.")

    return matrix


def apply_activation(matrix: Matrix, activation: str) -> Matrix:
    if activation == "identity":
        return [list(row) for row in matrix]
    if activation == "tanh":
        return [[math.tanh(value) for value in row] for row in matrix]
    if activation == "relu":
        return [[value if value > 0.0 else 0.0 for value in row] for row in matrix]

    raise ValueError(f"Unsupported activation: {activation}")


class LinearLayer:
    def __init__(self, input_dim: int, output_dim: int, *, seed: int = 0) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}.")

        generator = random.Random(seed)
        scale = 1.0 / math.sqrt(float(input_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights: Matrix = [
            [generator.uniform(-scale, scale) for _ in range(input_dim)]
            for _ in range(output_dim)
        ]
        self.bias: Vector = [0.0 for _ in range(output_dim)]

    def forward(self, inputs: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        matrix = ensure_matrix(inputs, name="inputs")
        for row in matrix:
            if len(row) != self.input_dim:
                raise ValueError(
                    f"Expected input_dim={self.input_dim}, got row length {len(row)}."
                )

        return [
            [
                sum(row[index] * self.weights[out_index][index] for index in range(self.input_dim))
                + self.bias[out_index]
                for out_index in range(self.output_dim)
            ]
            for row in matrix
        ]

    def __call__(self, inputs: Sequence[float] | Sequence[Sequence[float]]) -> Matrix:
        return self.forward(inputs)


def clone_matrix(matrix: Iterable[Iterable[float]]) -> Matrix:
    return [[float(value) for value in row] for row in matrix]
