from __future__ import annotations

from typing import Any, Sequence

from src.models._head_utils import Matrix
from src.models.projection_heads import ProjectionHead


class DynReconstructionHead:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        activation: str = "tanh",
        seed: int = 100,
    ) -> None:
        self.projection = ProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            seed=seed,
        )

    def forward(
        self,
        fused_missing_representation: Sequence[float] | Sequence[Sequence[float]] | Any,
    ) -> Matrix | Any:
        return self.projection(fused_missing_representation)

    def __call__(
        self,
        fused_missing_representation: Sequence[float] | Sequence[Sequence[float]] | Any,
    ) -> Matrix | Any:
        return self.forward(fused_missing_representation)

    def parameters(self) -> Sequence[Any]:
        return self.projection.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.projection.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.projection.load_state_dict(state_dict)
