from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.datasets.pretrain_dataset import PretrainDataset, PretrainSample
from src.losses.alignment import info_nce_loss
from src.losses.consistency import consistency_loss
from src.losses.reconstruction import reconstruction_loss
from src.models.fusion_transformer import FusionTransformer
from src.models.projection_heads import ProjectionHead
from src.models.reconstruction_heads import DynReconstructionHead
from src.models.sequence_encoder_esmc import MODEL_SPECS, SeqEncoderESMC
from src.models.structure_dynamics_encoder import STTransformer
from src.training.checkpoint_selection import is_better, resolve_checkpoint_dir


@dataclass
class PretrainTrainerConfig:
    manifest_path: str
    batch_size_pretrain: int = 1
    sequence_model_name: str = "esmc_600m"
    sequence_pooling: str = "mean_pool"
    structure_pooling: str = "mean_pool"
    fusion_pooling: str = "cls"
    embedding_dim: int = 8
    projection_dim: int = 4
    lambda_align: float = 0.1
    lambda_cons: float = 1.0
    lambda_recon: float = 0.2
    dyn_whole_modality_dropout_prob: float = 1.0
    consistency_mode: str = "cosine"
    checkpoint_path: Optional[str] = None
    max_residues: int = 100
    max_frames: int = 160
    backend_mode: str = "real"
    st_num_layers: int = 4
    st_num_heads: int = 8
    st_dropout: float = 0.1
    fusion_num_layers: int = 2
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    optimizer: str = "AdamW"
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_flash_attn: bool = True
    sequence_encoder_trainable: bool = True
    device: str = "cpu"
    max_epochs_pretrain: int = 10
    validation_ratio_pretrain: float = 0.0
    validation_seed_pretrain: int = 42
    validation_interval_pretrain: int = 1
    checkpoint_interval_pretrain: int = 5
    show_progress_pretrain: bool = True
    progress_log_interval_pretrain: int = 50
    min_delta_pretrain: float = 0.0
    tensorboard_log_dir: Optional[str] = None


@dataclass(frozen=True)
class PretrainStepResult:
    total_loss: float
    alignment_loss: float
    consistency_loss: float
    reconstruction_loss: float
    batch_size: int
    global_step: int
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_loss": self.total_loss,
            "alignment_loss": self.alignment_loss,
            "consistency_loss": self.consistency_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "batch_size": self.batch_size,
            "global_step": self.global_step,
            "metrics": dict(self.metrics),
        }


@dataclass
class PretrainSelectionState:
    monitor_name: str = "loss"
    monitor_mode: str = "min"
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    last_epoch: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PretrainSelectionState":
        return cls(
            monitor_name=str(payload.get("monitor_name", "loss")),
            monitor_mode=str(payload.get("monitor_mode", "min")),
            best_metric=None if payload.get("best_metric") is None else float(payload["best_metric"]),
            best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
            last_epoch=None if payload.get("last_epoch") is None else int(payload["last_epoch"]),
        )


class PretrainTrainer:
    def __init__(
        self,
        config: PretrainTrainerConfig,
        *,
        sequence_encoder: Optional[SeqEncoderESMC] = None,
        structure_encoder: Optional[STTransformer] = None,
        fusion_transformer: Optional[FusionTransformer] = None,
        seq_projection_head: Optional[ProjectionHead] = None,
        dyn_projection_head: Optional[ProjectionHead] = None,
        recon_head: Optional[DynReconstructionHead] = None,
    ) -> None:
        self.config = config
        self.backend_mode = config.backend_mode
        self.global_step = 0
        self.current_epoch = 0
        self.last_step_result: Optional[PretrainStepResult] = None
        self._optimizer: Any | None = None
        self._summary_writer: Any | None = None
        self.selection_state = PretrainSelectionState()
        if self.backend_mode != "real":
            raise ValueError(
                f"Unsupported backend_mode={self.backend_mode!r}. Expected 'real'."
            )
        if self.config.batch_size_pretrain <= 0:
            raise ValueError(f"batch_size_pretrain must be positive, got {self.config.batch_size_pretrain}.")
        if not 0.0 <= self.config.dyn_whole_modality_dropout_prob <= 1.0:
            raise ValueError(
                "dyn_whole_modality_dropout_prob must be in [0, 1], "
                f"got {self.config.dyn_whole_modality_dropout_prob}."
            )
        if not 0.0 <= self.config.validation_ratio_pretrain < 1.0:
            raise ValueError(
                "validation_ratio_pretrain must be in [0, 1), "
                f"got {self.config.validation_ratio_pretrain}."
            )

        _ensure_real_runtime_dependencies()
        if self.config.tensorboard_log_dir and importlib.util.find_spec("tensorboard") is None:
            raise RuntimeError(
                "tensorboard logging was requested, but the 'tensorboard' package is not installed."
            )

        embedding_dim = config.embedding_dim
        projection_dim = config.projection_dim
        sequence_output_dim = MODEL_SPECS[config.sequence_model_name].d_model

        self.sequence_encoder = sequence_encoder or SeqEncoderESMC(
            model_name=config.sequence_model_name,
            pooling=config.sequence_pooling,
            backend_mode="real",
            use_flash_attn=config.use_flash_attn,
            device=config.device,
        )
        self.structure_encoder = structure_encoder or STTransformer(
            d_model=embedding_dim,
            num_layers=config.st_num_layers,
            num_heads=config.st_num_heads,
            dropout=config.st_dropout,
            pooling=config.structure_pooling,
            backend_mode="real",
            device=config.device,
        )
        self.fusion_transformer = fusion_transformer or FusionTransformer(
            d_model=embedding_dim,
            num_layers=config.fusion_num_layers,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
            pooling=config.fusion_pooling,
            backend_mode="real",
            device=config.device,
        )
        self.seq_projection_head = seq_projection_head or ProjectionHead(
            sequence_output_dim,
            projection_dim,
            hidden_dim=projection_dim,
            seed=10,
        )
        self.dyn_projection_head = dyn_projection_head or ProjectionHead(
            embedding_dim,
            projection_dim,
            hidden_dim=projection_dim,
            seed=20,
        )
        self.recon_head = recon_head or DynReconstructionHead(
            embedding_dim,
            embedding_dim,
            hidden_dim=embedding_dim,
            seed=30,
        )
        self._configure_sequence_encoder_trainability()

    @property
    def optimizer(self) -> Any | None:
        return self._optimizer

    def train_step(self, batch: Sequence[PretrainSample]) -> PretrainStepResult:
        if not batch:
            raise ValueError("train_step requires at least one sample.")
        return self._train_step_real(batch)

    def _train_step_real(self, batch: Sequence[PretrainSample]) -> PretrainStepResult:
        torch = importlib.import_module("torch")
        nn_utils = importlib.import_module("torch.nn.utils")

        self._set_sequence_encoder_mode(training=True)
        losses = self._forward_real(batch, torch, deterministic_dyn_dropout=False)

        if self._optimizer is None:
            self._optimizer = self._build_optimizer(torch)

        self._optimizer.zero_grad(set_to_none=True)
        losses["total_loss_tensor"].backward()

        parameters = list(self.trainable_parameters())
        if self.config.grad_clip > 0.0 and parameters:
            nn_utils.clip_grad_norm_(parameters, self.config.grad_clip)

        self._optimizer.step()

        self.global_step += 1
        metric_values = _loss_metric_values(losses)
        step_result = PretrainStepResult(
            total_loss=metric_values["total_loss"],
            alignment_loss=metric_values["alignment_loss"],
            consistency_loss=metric_values["consistency_loss"],
            reconstruction_loss=metric_values["reconstruction_loss"],
            batch_size=len(batch),
            global_step=self.global_step,
            metrics=metric_values,
        )
        self.last_step_result = step_result
        return step_result

    def _forward_real(
        self,
        batch: Sequence[PretrainSample],
        torch_module: Any,
        *,
        deterministic_dyn_dropout: bool,
    ) -> Dict[str, Any]:
        sequences = [sample.sequence for sample in batch]
        seq_output = self.sequence_encoder(sequences)
        seq_embeddings = _ensure_tensor_matrix(
            seq_output.pooled_embedding,
            name="seq_embeddings",
            torch_module=torch_module,
            device=self.config.device,
        )

        dyn_rows: List[Any] = []
        for sample in batch:
            dyn_output = self.structure_encoder.encode_paths(
                nature_path=sample.nature_path,
                md_path=sample.md_path,
                max_residues=self.config.max_residues,
                max_frames=self.config.max_frames,
            )
            dyn_rows.append(
                _first_tensor_row(
                    dyn_output.pooled_embedding,
                    name="dyn_pooled_embedding",
                    torch_module=torch_module,
                    device=self.config.device,
                )
            )
        dyn_embeddings = torch_module.stack(dyn_rows, dim=0)
        dyn_dropout_mask = self._sample_dyn_dropout_mask(
            batch,
            torch_module,
            deterministic=deterministic_dyn_dropout,
        )
        missing_has_dyn = [not is_dropped for is_dropped in dyn_dropout_mask]

        z_seq = self.seq_projection_head(seq_embeddings)
        z_dyn = self.dyn_projection_head(dyn_embeddings)

        full_fusion = self.fusion_transformer(
            seq_embeddings,
            dyn_embeddings,
            has_dyn=[True] * len(batch),
        )
        missing_fusion = self.fusion_transformer(
            seq_embeddings,
            dyn_embeddings,
            has_dyn=missing_has_dyn,
        )

        fused_full = _ensure_tensor_matrix(
            full_fusion.fused_pooled,
            name="fused_full",
            torch_module=torch_module,
            device=self.config.device,
        )
        fused_missing = _ensure_tensor_matrix(
            missing_fusion.fused_pooled,
            name="fused_missing",
            torch_module=torch_module,
            device=self.config.device,
        )
        predicted_dyn = self.recon_head(fused_missing)

        alignment = info_nce_loss(z_seq, z_dyn)
        consistency = consistency_loss(
            fused_missing,
            fused_full,
            mode=self.config.consistency_mode,
            valid_mask=dyn_dropout_mask,
        )
        reconstruction = reconstruction_loss(
            predicted_dyn,
            dyn_embeddings,
            valid_mask=dyn_dropout_mask,
        )
        raw_total_loss = alignment + consistency + reconstruction
        weighted_alignment = self.config.lambda_align * alignment
        weighted_consistency = self.config.lambda_cons * consistency
        weighted_reconstruction = self.config.lambda_recon * reconstruction

        total_loss = (
            weighted_alignment
            + weighted_consistency
            + weighted_reconstruction
        )
        seq_dyn_cosine = _mean_cosine_similarity(
            z_seq,
            z_dyn,
            torch_module,
        )
        fused_missing_full_cosine = _mean_cosine_similarity(
            fused_missing,
            fused_full,
            torch_module,
            valid_mask=dyn_dropout_mask,
        )
        reconstruction_relative_mse = _relative_mse_to_target_norm(
            predicted_dyn,
            dyn_embeddings,
            torch_module,
            valid_mask=dyn_dropout_mask,
        )
        return {
            "total_loss_tensor": total_loss,
            "raw_total_loss_tensor": raw_total_loss,
            "alignment_loss_tensor": alignment,
            "consistency_loss_tensor": consistency,
            "reconstruction_loss_tensor": reconstruction,
            "weighted_alignment_loss_tensor": weighted_alignment,
            "weighted_consistency_loss_tensor": weighted_consistency,
            "weighted_reconstruction_loss_tensor": weighted_reconstruction,
            "seq_dyn_cosine_similarity_tensor": seq_dyn_cosine,
            "fused_missing_full_cosine_similarity_tensor": fused_missing_full_cosine,
            "reconstruction_relative_mse_tensor": reconstruction_relative_mse,
        }

    def evaluate_dataset(self, dataset: PretrainDataset) -> Dict[str, Any]:
        if len(dataset) == 0:
            raise ValueError("evaluate_dataset requires at least one full pretrain sample.")
        torch = importlib.import_module("torch")
        self._set_sequence_encoder_mode(training=False)
        total_samples = 0
        totals: Dict[str, float] = {}
        with torch.no_grad():
            for batch in _iter_sample_batches(dataset, self.config.batch_size_pretrain):
                losses = self._forward_real(batch, torch, deterministic_dyn_dropout=True)
                batch_size = len(batch)
                total_samples += batch_size
                for key, value in _loss_metric_values(losses).items():
                    totals[key] = totals.get(key, 0.0) + value * batch_size

        if total_samples <= 0:
            raise ValueError("evaluate_dataset did not receive any batched samples.")
        return {
            "overall": {
                "num_samples": total_samples,
                "metrics": {
                    key: value / total_samples
                    for key, value in sorted(totals.items())
                },
            }
        }

    def fit(
        self,
        train_dataset: PretrainDataset,
        *,
        validation_dataset: PretrainDataset | None = None,
        checkpoint_root: str | Path | None = None,
        max_epochs: int | None = None,
        validation_interval: int | None = None,
    ) -> Dict[str, Any]:
        if len(train_dataset) == 0:
            raise ValueError("fit requires at least one training sample.")

        resolved_validation_dataset = validation_dataset or train_dataset
        resolved_max_epochs = self.config.max_epochs_pretrain if max_epochs is None else int(max_epochs)
        resolved_validation_interval = (
            self.config.validation_interval_pretrain
            if validation_interval is None
            else int(validation_interval)
        )
        if resolved_max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {resolved_max_epochs}.")
        if resolved_validation_interval <= 0:
            raise ValueError(
                f"validation_interval must be positive, got {resolved_validation_interval}."
            )
        if self.config.checkpoint_interval_pretrain <= 0:
            raise ValueError(
                "checkpoint_interval_pretrain must be positive, "
                f"got {self.config.checkpoint_interval_pretrain}."
            )
        if self.config.progress_log_interval_pretrain <= 0:
            raise ValueError(
                "progress_log_interval_pretrain must be positive, "
                f"got {self.config.progress_log_interval_pretrain}."
            )

        checkpoint_dir: Path | None = None
        if checkpoint_root is not None or self.config.checkpoint_path is not None:
            checkpoint_dir = resolve_checkpoint_dir(
                checkpoint_root if checkpoint_root is not None else str(self.config.checkpoint_path)
            )

        start_epoch = self.current_epoch + 1
        completed_epochs = 0
        latest_validation: Dict[str, Any] | None = None
        checkpoint_paths: Dict[str, str] = {}
        num_train_batches = _num_batches(len(train_dataset), self.config.batch_size_pretrain)

        try:
            for epoch in range(start_epoch, resolved_max_epochs + 1):
                epoch_step_results: list[PretrainStepResult] = []
                epoch_batches = _iter_sample_batches(train_dataset, self.config.batch_size_pretrain)
                batch_iterator = _maybe_progress_bar(
                    epoch_batches,
                    enabled=self.config.show_progress_pretrain,
                    description=f"pretrain epoch {epoch}/{resolved_max_epochs}",
                )
                if self.config.show_progress_pretrain and batch_iterator is epoch_batches:
                    print(
                        f"pretrain epoch={epoch}/{resolved_max_epochs} "
                        f"batches={num_train_batches}",
                        flush=True,
                    )
                for batch_index, batch in enumerate(batch_iterator, start=1):
                    step_result = self.train_step(batch)
                    epoch_step_results.append(step_result)
                    self._log_training_step(step_result)
                    _update_progress(
                        batch_iterator,
                        step_result,
                        epoch=epoch,
                        max_epochs=resolved_max_epochs,
                        batch_index=batch_index,
                        num_batches=num_train_batches,
                        enabled=self.config.show_progress_pretrain,
                        log_interval=self.config.progress_log_interval_pretrain,
                    )

                self.current_epoch = epoch
                self.selection_state.last_epoch = epoch
                completed_epochs += 1
                epoch_metrics = _summarize_step_results(epoch_step_results)
                self._log_training_epoch(epoch, epoch_metrics)

                should_validate = (epoch % resolved_validation_interval == 0) or (epoch == resolved_max_epochs)
                current_validation: Dict[str, Any] | None = None
                monitor_name = self.selection_state.monitor_name
                improved = False

                if should_validate:
                    current_validation = self.evaluate_dataset(resolved_validation_dataset)
                    latest_validation = current_validation
                    self._log_validation_epoch(epoch, current_validation)
                    monitor_value = float(current_validation["overall"]["metrics"][monitor_name])
                    improved = is_better(
                        monitor_value,
                        self.selection_state.best_metric,
                        mode=self.selection_state.monitor_mode,
                        min_delta=self.config.min_delta_pretrain,
                    )
                    if improved:
                        self.selection_state.best_metric = monitor_value
                        self.selection_state.best_epoch = epoch

                if checkpoint_dir is not None:
                    latest_path = self.save_checkpoint(
                        checkpoint_dir / "latest.pth",
                        epoch=epoch,
                        monitor_name=monitor_name,
                        val_metrics=current_validation,
                        checkpoint_kind="latest",
                    )
                    checkpoint_paths["latest"] = str(latest_path)
                    if epoch % self.config.checkpoint_interval_pretrain == 0:
                        periodic_path = self.save_checkpoint(
                            checkpoint_dir / f"epoch_{epoch:04d}.pth",
                            epoch=epoch,
                            monitor_name=monitor_name,
                            val_metrics=current_validation,
                            checkpoint_kind="periodic",
                        )
                        checkpoint_paths[f"epoch_{epoch:04d}"] = str(periodic_path)
                    if improved:
                        best_path = self.save_checkpoint(
                            checkpoint_dir / "best.pth",
                            epoch=epoch,
                            monitor_name=monitor_name,
                            val_metrics=current_validation,
                            checkpoint_kind="best",
                        )
                        checkpoint_paths["best"] = str(best_path)
        finally:
            self._close_summary_writer()

        best_metric = self.selection_state.best_metric
        return {
            "stage": "pretrain",
            "epochs_completed": completed_epochs,
            "start_epoch": start_epoch,
            "final_epoch": self.current_epoch,
            "batch_size_pretrain": self.config.batch_size_pretrain,
            "num_train_batches": num_train_batches,
            "validation_interval": resolved_validation_interval,
            "checkpoint_interval": self.config.checkpoint_interval_pretrain,
            "best_epoch": self.selection_state.best_epoch,
            "best_metric": None if best_metric is None else float(best_metric),
            "checkpoint_paths": checkpoint_paths,
            "last_total_loss": None if self.last_step_result is None else self.last_step_result.total_loss,
            "last_validation_loss": None
            if latest_validation is None
            else float(latest_validation["overall"]["metrics"]["loss"]),
        }

    def trainable_parameters(self) -> Sequence[Any]:
        parameters: list[Any] = []
        seen: set[int] = set()
        components = [
            self.sequence_encoder,
            self.structure_encoder,
            self.fusion_transformer,
            self.seq_projection_head,
            self.dyn_projection_head,
            self.recon_head,
        ]
        for component in components:
            if not hasattr(component, "parameters"):
                continue
            for parameter in component.parameters():
                parameter_id = id(parameter)
                if parameter_id in seen:
                    continue
                if not _parameter_requires_grad(parameter):
                    continue
                seen.add(parameter_id)
                parameters.append(parameter)
        return parameters

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        epoch: Optional[int] = None,
        monitor_name: Optional[str] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        checkpoint_kind: str = "manual",
    ) -> Path:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "stage": "pretrain",
            "checkpoint_kind": checkpoint_kind,
            "backend_mode": self.backend_mode,
            "epoch": self.current_epoch if epoch is None else epoch,
            "monitor_name": monitor_name,
            "val_metrics": val_metrics,
            "global_step": self.global_step,
            "config": asdict(self.config),
            "component_summary": self.component_summary(),
            "last_step_result": None if self.last_step_result is None else self.last_step_result.to_dict(),
            "selection_state": self.selection_state.to_dict(),
        }
        torch = importlib.import_module("torch")
        payload["model_state"] = self._model_state_dict()
        payload["optimizer_state"] = None if self._optimizer is None else self._optimizer.state_dict()
        torch.save(payload, path)
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        torch = importlib.import_module("torch")
        payload = torch.load(path, map_location=self.config.device)

        backend_mode = payload.get("backend_mode")
        if backend_mode != self.backend_mode:
            raise ValueError(
                f"Checkpoint backend_mode={backend_mode!r} does not match trainer backend_mode={self.backend_mode!r}."
            )

        self.global_step = int(payload.get("global_step", 0))
        epoch = payload.get("epoch")
        if epoch is not None:
            self.current_epoch = int(epoch)
        last_step_result = payload.get("last_step_result")
        if isinstance(last_step_result, dict):
            metrics = last_step_result.get("metrics")
            self.last_step_result = PretrainStepResult(
                total_loss=float(last_step_result["total_loss"]),
                alignment_loss=float(last_step_result["alignment_loss"]),
                consistency_loss=float(last_step_result["consistency_loss"]),
                reconstruction_loss=float(last_step_result["reconstruction_loss"]),
                batch_size=int(last_step_result["batch_size"]),
                global_step=int(last_step_result["global_step"]),
                metrics=dict(metrics) if isinstance(metrics, dict) else {},
            )
        else:
            self.last_step_result = None

        selection_state = payload.get("selection_state")
        if isinstance(selection_state, dict):
            self.selection_state = PretrainSelectionState.from_dict(selection_state)

        model_state = payload.get("model_state") or {}
        self._load_model_state(model_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            self._optimizer = self._build_optimizer(torch)
            self._optimizer.load_state_dict(optimizer_state)

        return payload

    def config_summary(self) -> Dict[str, Any]:
        summary = asdict(self.config)
        summary["backend_mode"] = self.backend_mode
        return summary

    def _configure_sequence_encoder_trainability(self) -> None:
        if self.config.sequence_encoder_trainable:
            self.sequence_encoder.unfreeze_backbone()
        else:
            self.sequence_encoder.freeze_backbone()

    def _set_sequence_encoder_mode(self, *, training: bool) -> None:
        if training and self.config.sequence_encoder_trainable:
            self.sequence_encoder.train(True)
            return
        self.sequence_encoder.eval()

    def component_summary(self) -> Dict[str, str]:
        summary = {
            "sequence_encoder": "real",
            "structure_encoder": "real",
            "fusion_transformer": "real",
            "seq_projection_head": "real",
            "dyn_projection_head": "real",
            "recon_head": "real",
            "optimizer": self.config.optimizer,
        }
        if self.config.tensorboard_log_dir:
            summary["tensorboard"] = self.config.tensorboard_log_dir
        return summary

    def _get_summary_writer(self) -> Any | None:
        if not self.config.tensorboard_log_dir:
            return None
        if self._summary_writer is None:
            tensorboard_module = importlib.import_module("torch.utils.tensorboard")
            log_dir = Path(self.config.tensorboard_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._summary_writer = tensorboard_module.SummaryWriter(log_dir=str(log_dir))
        return self._summary_writer

    def _close_summary_writer(self) -> None:
        if self._summary_writer is None:
            return
        self._summary_writer.flush()
        self._summary_writer.close()
        self._summary_writer = None

    def _log_training_step(self, step_result: PretrainStepResult) -> None:
        writer = self._get_summary_writer()
        if writer is None:
            return
        writer.add_scalar("pretrain/train_step/total_loss", step_result.total_loss, step_result.global_step)
        writer.add_scalar("pretrain/train_step/alignment_loss", step_result.alignment_loss, step_result.global_step)
        writer.add_scalar("pretrain/train_step/consistency_loss", step_result.consistency_loss, step_result.global_step)
        writer.add_scalar("pretrain/train_step/reconstruction_loss", step_result.reconstruction_loss, step_result.global_step)
        legacy_keys = {
            "total_loss",
            "alignment_loss",
            "consistency_loss",
            "reconstruction_loss",
        }
        for key, value in sorted(step_result.metrics.items()):
            if key in legacy_keys:
                continue
            writer.add_scalar(f"pretrain/train_step/{key}", float(value), step_result.global_step)

    def _log_training_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        writer = self._get_summary_writer()
        if writer is None:
            return
        for key, value in sorted(metrics.items()):
            writer.add_scalar(f"pretrain/train_epoch/{key}", value, epoch)

    def _log_validation_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        writer = self._get_summary_writer()
        if writer is None:
            return
        for key, value in sorted(metrics["overall"]["metrics"].items()):
            writer.add_scalar(f"pretrain/val/{key}", float(value), epoch)

    def _build_optimizer(self, torch_module: Any) -> Any:
        parameters = list(self.trainable_parameters())
        if not parameters:
            raise RuntimeError(
                "No trainable parameters were registered for the real pretrain path."
            )

        optimizer_name = self.config.optimizer.strip().lower()
        if optimizer_name == "adamw":
            return torch_module.optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if optimizer_name == "sgd":
            return torch_module.optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _model_state_dict(self) -> Dict[str, Any]:
        return {
            "sequence_encoder": self.sequence_encoder.state_dict(),
            "structure_encoder": self.structure_encoder.state_dict(),
            "fusion_transformer": self.fusion_transformer.state_dict(),
            "seq_projection_head": self.seq_projection_head.state_dict(),
            "dyn_projection_head": self.dyn_projection_head.state_dict(),
            "recon_head": self.recon_head.state_dict(),
        }

    def _load_model_state(self, model_state: Dict[str, Any]) -> None:
        self.sequence_encoder.load_state_dict(model_state.get("sequence_encoder", {}))
        self.structure_encoder.load_state_dict(model_state.get("structure_encoder", {}))
        self.fusion_transformer.load_state_dict(model_state.get("fusion_transformer", {}))
        self.seq_projection_head.load_state_dict(model_state.get("seq_projection_head", {}))
        self.dyn_projection_head.load_state_dict(model_state.get("dyn_projection_head", {}))
        self.recon_head.load_state_dict(model_state.get("recon_head", {}))

    def _sample_dyn_dropout_mask(
        self,
        batch: Sequence[PretrainSample],
        torch_module: Any,
        *,
        deterministic: bool,
    ) -> list[bool]:
        probability = float(self.config.dyn_whole_modality_dropout_prob)
        if probability <= 0.0:
            return [False] * len(batch)
        if probability >= 1.0:
            return [True] * len(batch)
        if deterministic:
            return [
                _stable_unit_interval(
                    f"{sample.protein_id}:{sample.sequence_hash}:{sample.nature_path}:{sample.md_path}"
                )
                < probability
                for sample in batch
            ]
        random_values = torch_module.rand((len(batch),))
        return [bool(value) for value in (random_values < probability).detach().cpu().tolist()]


def build_pretrain_trainer(config: PretrainTrainerConfig) -> PretrainTrainer:
    return PretrainTrainer(config)


def _ensure_real_runtime_dependencies() -> None:
    missing_dependencies = []
    if importlib.util.find_spec("torch") is None:
        missing_dependencies.append("torch")
    if missing_dependencies:
        raise RuntimeError(
            "Real pretrain trainer requires additional runtime dependencies: "
            f"{', '.join(missing_dependencies)}"
        )


def _ensure_tensor_matrix(
    values: Any,
    *,
    name: str,
    torch_module: Any,
    device: str,
) -> Any:
    if isinstance(values, torch_module.Tensor):
        tensor = values
    else:
        tensor = torch_module.as_tensor(values, dtype=torch_module.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, dim], got {tuple(tensor.shape)}.")
    return tensor.to(device=device, dtype=torch_module.float32)


def _first_tensor_row(
    values: Any,
    *,
    name: str,
    torch_module: Any,
    device: str,
) -> Any:
    matrix = _ensure_tensor_matrix(
        values,
        name=name,
        torch_module=torch_module,
        device=device,
    )
    if matrix.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one sample row, got {matrix.shape[0]}.")
    return matrix[0]


def _tensor_or_float_to_python(value: Any) -> float:
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
    return float(value)


def _loss_metric_values(losses: Dict[str, Any]) -> Dict[str, float]:
    total_loss = _tensor_or_float_to_python(losses["total_loss_tensor"])
    raw_total_loss = _tensor_or_float_to_python(losses["raw_total_loss_tensor"])
    alignment_loss = _tensor_or_float_to_python(losses["alignment_loss_tensor"])
    consistency_loss = _tensor_or_float_to_python(losses["consistency_loss_tensor"])
    reconstruction_loss = _tensor_or_float_to_python(losses["reconstruction_loss_tensor"])
    weighted_alignment = _tensor_or_float_to_python(losses["weighted_alignment_loss_tensor"])
    weighted_consistency = _tensor_or_float_to_python(losses["weighted_consistency_loss_tensor"])
    weighted_reconstruction = _tensor_or_float_to_python(losses["weighted_reconstruction_loss_tensor"])

    return {
        "loss": total_loss,
        "total_loss": total_loss,
        "alignment_loss": alignment_loss,
        "consistency_loss": consistency_loss,
        "reconstruction_loss": reconstruction_loss,
        "raw/total_loss": raw_total_loss,
        "raw/alignment_loss": alignment_loss,
        "raw/consistency_loss": consistency_loss,
        "raw/reconstruction_loss": reconstruction_loss,
        "weighted/total_loss": total_loss,
        "weighted/alignment_loss": weighted_alignment,
        "weighted/consistency_loss": weighted_consistency,
        "weighted/reconstruction_loss": weighted_reconstruction,
        "diagnostics/seq_dyn_cosine_similarity": _tensor_or_float_to_python(
            losses["seq_dyn_cosine_similarity_tensor"]
        ),
        "diagnostics/fused_missing_full_cosine_similarity": _tensor_or_float_to_python(
            losses["fused_missing_full_cosine_similarity_tensor"]
        ),
        "diagnostics/reconstruction_relative_mse": _tensor_or_float_to_python(
            losses["reconstruction_relative_mse_tensor"]
        ),
    }


def _mean_cosine_similarity(
    left: Any,
    right: Any,
    torch_module: Any,
    *,
    valid_mask: Sequence[bool] | None = None,
) -> Any:
    functional = importlib.import_module("torch.nn.functional")
    left_tensor = left.detach()
    right_tensor = right.detach()
    if valid_mask is not None:
        mask_tensor = torch_module.as_tensor(
            valid_mask,
            dtype=torch_module.bool,
            device=left_tensor.device,
        )
        left_tensor = left_tensor[mask_tensor]
        right_tensor = right_tensor[mask_tensor]
    if left_tensor.shape[0] == 0:
        return (left_tensor.sum() + right_tensor.sum()) * 0.0
    return functional.cosine_similarity(left_tensor, right_tensor, dim=-1).mean()


def _relative_mse_to_target_norm(
    predicted: Any,
    target: Any,
    torch_module: Any,
    *,
    valid_mask: Sequence[bool] | None = None,
) -> Any:
    predicted_tensor = predicted.detach()
    target_tensor = target.detach()
    if valid_mask is not None:
        mask_tensor = torch_module.as_tensor(
            valid_mask,
            dtype=torch_module.bool,
            device=predicted_tensor.device,
        )
        predicted_tensor = predicted_tensor[mask_tensor]
        target_tensor = target_tensor[mask_tensor]
    if predicted_tensor.shape[0] == 0:
        return (predicted_tensor.sum() + target_tensor.sum()) * 0.0

    squared_error = (predicted_tensor - target_tensor).pow(2).mean(dim=-1)
    target_power = target_tensor.pow(2).mean(dim=-1).clamp_min(1e-12)
    return (squared_error / target_power).mean()


def _parameter_requires_grad(parameter: Any) -> bool:
    return bool(getattr(parameter, "requires_grad", True))


def _maybe_progress_bar(
    batches: Sequence[Sequence[PretrainSample]],
    *,
    enabled: bool,
    description: str,
) -> Any:
    if not enabled or importlib.util.find_spec("tqdm") is None:
        return batches
    tqdm_module = importlib.import_module("tqdm.auto")
    return tqdm_module.tqdm(
        batches,
        total=len(batches),
        desc=description,
        dynamic_ncols=True,
        leave=True,
    )


def _update_progress(
    progress_iterator: Any,
    step_result: PretrainStepResult,
    *,
    epoch: int,
    max_epochs: int,
    batch_index: int,
    num_batches: int,
    enabled: bool,
    log_interval: int,
) -> None:
    if not enabled:
        return
    if hasattr(progress_iterator, "set_postfix"):
        progress_iterator.set_postfix(
            loss=f"{step_result.total_loss:.4f}",
            align=f"{step_result.alignment_loss:.4f}",
            cons=f"{step_result.consistency_loss:.4f}",
            recon=f"{step_result.reconstruction_loss:.4f}",
        )
        return
    if batch_index % log_interval != 0 and batch_index != num_batches:
        return
    print(
        f"pretrain epoch={epoch}/{max_epochs} "
        f"batch={batch_index}/{num_batches} "
        f"loss={step_result.total_loss:.6f} "
        f"align={step_result.alignment_loss:.6f} "
        f"cons={step_result.consistency_loss:.6f} "
        f"recon={step_result.reconstruction_loss:.6f}",
        flush=True,
    )


def _iter_sample_batches(
    dataset: PretrainDataset,
    batch_size: int,
) -> Sequence[Sequence[PretrainSample]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return [
        list(dataset.samples[start_index : start_index + batch_size])
        for start_index in range(0, len(dataset), batch_size)
    ]


def _num_batches(total_samples: int, batch_size: int) -> int:
    if total_samples <= 0:
        return 0
    return (total_samples + batch_size - 1) // batch_size


def _summarize_step_results(step_results: Sequence[PretrainStepResult]) -> Dict[str, float]:
    if not step_results:
        raise ValueError("step_results cannot be empty.")
    total_samples = sum(step_result.batch_size for step_result in step_results)
    if total_samples <= 0:
        raise ValueError("step_results must contain a positive total batch size.")
    metric_keys = sorted(
        {
            key
            for step_result in step_results
            for key in step_result.metrics
        }
    )
    if not metric_keys:
        return {
            "total_loss": sum(step.total_loss * step.batch_size for step in step_results) / total_samples,
            "alignment_loss": sum(step.alignment_loss * step.batch_size for step in step_results) / total_samples,
            "consistency_loss": sum(step.consistency_loss * step.batch_size for step in step_results) / total_samples,
            "reconstruction_loss": sum(step.reconstruction_loss * step.batch_size for step in step_results)
            / total_samples,
        }
    return {
        key: sum(
            step.metrics.get(key, 0.0) * step.batch_size
            for step in step_results
        )
        / total_samples
        for key in metric_keys
    }


def _stable_unit_interval(seed_text: str) -> float:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    numerator = int(digest[:16], 16)
    denominator = float((16**16) - 1)
    return numerator / denominator
