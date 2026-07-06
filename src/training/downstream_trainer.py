from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.datasets.downstream_adapters import load_downstream_samples

from src.datasets.downstream_dataset import (
    DownstreamDataset,
    DownstreamStage1Sample,
    task_kind_for_name,
)
from src.evaluation.evaluator import EvaluationRecord, ProtMMLMEvaluator
from src.losses.alignment import info_nce_loss
from src.losses.consistency import consistency_loss
from src.losses.reconstruction import reconstruction_loss
from src.losses.task import (
    binary_classification_loss,
    multilabel_classification_loss,
    regression_loss,
)
from src.models.fusion_transformer import FusionTransformer
from src.models.projection_heads import ProjectionHead
from src.models.reconstruction_heads import DynReconstructionHead
from src.models.sequence_encoder_esmc import MODEL_SPECS, SeqEncoderESMC
from src.models.structure_dynamics_encoder import STTransformer
from src.models.task_heads import (
    BinaryClassificationHead,
    BinaryMLPClassificationHead,
    MLPRegressionHead,
    MultiLabelClassificationHead,
    RegressionHead,
)
from src.training.checkpoint_selection import (
    EarlyStoppingState,
    DownstreamSelectionState,
    extract_metric,
    is_better,
    monitor_mode_for_metric,
    primary_monitor_name_for_task,
    resolve_checkpoint_dir,
    significantly_worse,
)


@dataclass
class DownstreamTrainerConfig:
    manifest_path: str
    task_name: str
    sample_limit: int | None = None
    batch_size: int = 1
    sequence_model_name: str = "esmc_600m"
    sequence_pooling: str = "mean_pool"
    structure_pooling: str = "mean_pool"
    fusion_pooling: str = "cls"
    embedding_dim: int = 8
    projection_dim: int = 4
    lambda_align: float = 0.1
    lambda_cons: float = 1.0
    lambda_recon: float = 0.2
    consistency_mode: str = "cosine"
    checkpoint_path: Optional[str] = None
    pretrain_checkpoint_path: Optional[str] = None
    backend_mode: str = "real"
    regression_loss_mode: str = "huber"
    regression_delta: float = 1.0
    multilabel_pos_weight_mode: str = "none"
    multilabel_max_pos_weight: float = 20.0
    max_sequence_length: int = 100
    protein_max_sequence_length: int = 100
    peptide_max_sequence_length: int = 100
    max_residues: int = 100
    protein_max_residues: int = 100
    peptide_max_residues: int = 100
    max_frames: int = 160
    max_epochs_downstream: int = 10
    min_epochs_downstream: int = 1
    patience_downstream: int = 3
    min_delta_downstream: float = 0.0
    validation_interval_downstream: int = 1
    seq_guard_tolerance: float = 0.0
    st_num_layers: int = 4
    st_num_heads: int = 8
    st_dropout: float = 0.1
    fusion_num_layers: int = 2
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    optimizer: str = "AdamW"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_flash_attn: bool = True
    sequence_encoder_trainable: bool = True
    structure_encoder_trainable: bool = True
    fusion_transformer_trainable: bool = True
    show_progress_downstream: bool = True
    progress_log_interval_downstream: int = 50
    gradient_accumulation_steps: int = 1
    sequence_only_baseline: bool = False
    force_missing_dynamics_for_all_samples: bool = False
    force_natural_structure_only_for_all_samples: bool = False
    mixed_task_mode: str = "coverage_aware"
    partial_pair_handling: str = "seq_fallback"
    aux_loss_reweight_mode: str = "batch_coverage"
    min_full_fraction_for_aux: float = 0.1
    task_head_type: str = "auto"
    train_only_task_head: bool = False
    task_head_hidden_dims: tuple[int, ...] = (512, 128)
    task_head_dropout: float = 0.1
    save_best_guarded_checkpoint: bool = True
    single_missing_task_fallback: str = "fused_missing"
    single_task_feature_mode: str = "legacy"
    monitor_subset: str | None = None
    monitor_name: str | None = None
    manifest_validation_policy: str = "deterministic_grouped_9_1"
    shuffle_train_each_epoch: bool = True
    train_shuffle_seed: int = 17
    single_full_sample_oversample_factor: int = 1
    device: str = "cpu"


@dataclass(frozen=True)
class DownstreamStepResult:
    task_name: str
    task_kind: str
    total_loss: float
    task_loss: float
    alignment_loss: float
    consistency_loss: float
    reconstruction_loss: float
    batch_size: int
    num_full_samples: int
    num_seq_only_samples: int
    global_step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_kind": self.task_kind,
            "total_loss": self.total_loss,
            "task_loss": self.task_loss,
            "alignment_loss": self.alignment_loss,
            "consistency_loss": self.consistency_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "batch_size": self.batch_size,
            "num_full_samples": self.num_full_samples,
            "num_seq_only_samples": self.num_seq_only_samples,
            "global_step": self.global_step,
        }


class DownstreamTrainer:
    def __init__(
        self,
        config: DownstreamTrainerConfig,
        *,
        sequence_encoder: Optional[SeqEncoderESMC] = None,
        structure_encoder: Optional[STTransformer] = None,
        fusion_transformer: Optional[FusionTransformer] = None,
        seq_projection_head: Optional[ProjectionHead] = None,
        dyn_projection_head: Optional[ProjectionHead] = None,
        recon_head: Optional[DynReconstructionHead] = None,
        seq_task_adapter: Optional[ProjectionHead] = None,
        single_task_residual_adapter: Optional[ProjectionHead] = None,
        task_head: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.task_name = _normalize_task_name(config.task_name)
        self.task_kind = task_kind_for_name(self.task_name)
        self.backend_mode = config.backend_mode
        self.global_step = 0
        self.current_epoch = 0
        self.last_step_result: Optional[DownstreamStepResult] = None
        self._optimizer: Any | None = None
        monitor_name = (
            str(config.monitor_name).strip()
            if config.monitor_name is not None and str(config.monitor_name).strip()
            else primary_monitor_name_for_task(self.task_kind)
        )
        self.selection_state = DownstreamSelectionState(
            early_stopping=EarlyStoppingState(
                monitor_name=monitor_name,
                monitor_mode=monitor_mode_for_metric(monitor_name),
            )
        )

        enabled_control_modes = [
            name
            for name, enabled in (
                ("sequence_only_baseline", self.config.sequence_only_baseline),
                ("force_missing_dynamics_for_all_samples", self.config.force_missing_dynamics_for_all_samples),
                ("force_natural_structure_only_for_all_samples", self.config.force_natural_structure_only_for_all_samples),
            )
            if enabled
        ]
        if len(enabled_control_modes) > 1:
            raise ValueError(
                "Control modes are mutually exclusive: "
                f"{', '.join(enabled_control_modes)}. "
                "Use sequence_only_baseline for the pure sequence baseline, "
                "force_missing_dynamics_for_all_samples for the missing-dynamics/mask-token control, "
                "or force_natural_structure_only_for_all_samples for the natural-structure-only control."
            )

        if self.backend_mode != "real":
            raise ValueError(
                f"Unsupported backend_mode={self.backend_mode!r}. Expected 'real'."
            )

        _ensure_real_runtime_dependencies()
        sequence_output_dim = MODEL_SPECS[config.sequence_model_name].d_model
        self.sequence_output_dim = sequence_output_dim

        self.sequence_encoder = sequence_encoder or SeqEncoderESMC(
            model_name=config.sequence_model_name,
            pooling=config.sequence_pooling,
            backend_mode="real",
            use_flash_attn=config.use_flash_attn,
            device=config.device,
            max_sequence_length=config.max_sequence_length,
        )
        self.structure_encoder = structure_encoder or STTransformer(
            d_model=config.embedding_dim,
            num_layers=config.st_num_layers,
            num_heads=config.st_num_heads,
            dropout=config.st_dropout,
            pooling=config.structure_pooling,
            backend_mode="real",
            device=config.device,
        )
        self.fusion_transformer = fusion_transformer or FusionTransformer(
            d_model=config.embedding_dim,
            num_layers=config.fusion_num_layers,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
            pooling=config.fusion_pooling,
            backend_mode="real",
            device=config.device,
        )
        self.seq_projection_head = seq_projection_head or ProjectionHead(
            sequence_output_dim,
            config.projection_dim,
            hidden_dim=config.projection_dim,
            seed=110,
        )
        self.dyn_projection_head = dyn_projection_head or ProjectionHead(
            config.embedding_dim,
            config.projection_dim,
            hidden_dim=config.projection_dim,
            seed=120,
        )
        self.recon_head = recon_head or DynReconstructionHead(
            config.embedding_dim,
            config.embedding_dim,
            hidden_dim=config.embedding_dim,
            seed=130,
        )
        self.seq_task_adapter = seq_task_adapter or ProjectionHead(
            sequence_output_dim,
            config.embedding_dim,
            hidden_dim=config.embedding_dim,
            activation="tanh",
            seed=140,
        )
        self.single_task_residual_adapter = single_task_residual_adapter or ProjectionHead(
            config.embedding_dim,
            sequence_output_dim,
            hidden_dim=sequence_output_dim,
            activation="tanh",
            seed=150,
        )
        self.task_head = task_head or self._build_task_head(config.embedding_dim)
        self._configure_module_trainability()

    @property
    def optimizer(self) -> Any | None:
        return self._optimizer

    def train_step(self, batch: Sequence[DownstreamStage1Sample]) -> DownstreamStepResult:
        if not batch:
            raise ValueError("train_step requires at least one sample.")
        return self._train_step_real(batch)

    def _train_step_real(self, batch: Sequence[DownstreamStage1Sample]) -> DownstreamStepResult:
        torch = importlib.import_module("torch")
        nn_utils = importlib.import_module("torch.nn.utils")

        self._set_module_modes(training=True)
        losses = self._forward_real(batch, torch)
        if self._optimizer is None:
            self._optimizer = self._build_optimizer(torch)
            self._optimizer.zero_grad(set_to_none=True)
        else:
            self._sync_optimizer_parameters()

        accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
        (losses["total_loss_tensor"] / float(accumulation_steps)).backward()

        should_step = ((self.global_step + 1) % accumulation_steps) == 0
        if should_step:
            parameters = list(self.trainable_parameters())
            if self.config.grad_clip > 0.0 and parameters:
                nn_utils.clip_grad_norm_(parameters, self.config.grad_clip)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        step_result = DownstreamStepResult(
            task_name=self.task_name,
            task_kind=self.task_kind,
            total_loss=float(losses["total_loss_tensor"].detach().cpu().item()),
            task_loss=float(losses["task_loss_tensor"].detach().cpu().item()),
            alignment_loss=float(losses["alignment_loss_tensor"].detach().cpu().item()),
            consistency_loss=float(losses["consistency_loss_tensor"].detach().cpu().item()),
            reconstruction_loss=float(losses["reconstruction_loss_tensor"].detach().cpu().item()),
            batch_size=len(batch),
            num_full_samples=int(losses["num_full_samples"]),
            num_seq_only_samples=int(losses["num_seq_only_samples"]),
            global_step=self.global_step,
        )
        self.last_step_result = step_result
        return step_result

    def _forward_real(self, batch: Sequence[DownstreamStage1Sample], torch_module: Any) -> Dict[str, Any]:
        if self.config.sequence_only_baseline:
            return self._forward_seq_only_real(batch, torch_module)
        if self._is_pair_batch(batch):
            return self._forward_pair_real(batch, torch_module)
        return self._forward_single_real(batch, torch_module)

    def _forward_seq_only_real(self, batch: Sequence[DownstreamStage1Sample], torch_module: Any) -> Dict[str, Any]:
        if self._is_pair_batch(batch):
            protein_sequences = [
                sample.sequence[: self.config.protein_max_sequence_length]
                for sample in batch
            ]
            peptide_sequences = [
                self._require_peptide_sequence(sample)[: self.config.peptide_max_sequence_length]
                for sample in batch
            ]
            protein_seq_output = self.sequence_encoder(protein_sequences)
            peptide_seq_output = self.sequence_encoder(peptide_sequences)
            protein_seq_embeddings = _ensure_tensor_matrix(
                protein_seq_output.pooled_embedding,
                name="protein_seq_embeddings",
                torch_module=torch_module,
                device=self.config.device,
            )
            peptide_seq_embeddings = _ensure_tensor_matrix(
                peptide_seq_output.pooled_embedding,
                name="peptide_seq_embeddings",
                torch_module=torch_module,
                device=self.config.device,
            )
            features = torch_module.cat([protein_seq_embeddings, peptide_seq_embeddings], dim=-1)
        else:
            seq_output = self.sequence_encoder([sample.sequence for sample in batch])
            features = _ensure_tensor_matrix(
                seq_output.pooled_embedding,
                name="seq_embeddings",
                torch_module=torch_module,
                device=self.config.device,
            )

        predictions = self.task_head(self._maybe_detach_task_features(features))
        task_loss = self._compute_task_loss(predictions, [sample.target for sample in batch])
        zero = task_loss * 0.0
        return {
            "total_loss_tensor": task_loss,
            "task_loss_tensor": task_loss,
            "alignment_loss_tensor": zero,
            "consistency_loss_tensor": zero,
            "reconstruction_loss_tensor": zero,
            "num_full_samples": 0,
            "num_seq_only_samples": len(batch),
        }

    def _forward_single_real(self, batch: Sequence[DownstreamStage1Sample], torch_module: Any) -> Dict[str, Any]:
        sequences = [sample.sequence for sample in batch]
        seq_output = self.sequence_encoder(sequences)
        seq_embeddings = _ensure_tensor_matrix(
            seq_output.pooled_embedding,
            name="seq_embeddings",
            torch_module=torch_module,
            device=self.config.device,
        )

        dyn_embeddings, task_structure_mask, aux_md_mask = self._encode_single_dyn_embeddings(
            batch,
            torch_module,
        )
        num_full_samples = sum(int(value) for value in aux_md_mask)
        num_seq_only_samples = len(batch) - num_full_samples

        full_view = self.fusion_transformer(
            seq_embeddings,
            dyn_embeddings,
            has_dyn=task_structure_mask,
        )
        missing_view = self.fusion_transformer(
            seq_embeddings,
            dyn_embeddings,
            has_dyn=[False] * len(batch),
        )

        fused_full = _ensure_tensor_matrix(
            full_view.fused_pooled,
            name="fused_full",
            torch_module=torch_module,
            device=self.config.device,
        )
        fused_missing = _ensure_tensor_matrix(
            missing_view.fused_pooled,
            name="fused_missing",
            torch_module=torch_module,
            device=self.config.device,
        )
        seq_task_features = self.seq_task_adapter(seq_embeddings)
        task_features = self._choose_single_task_features(
            seq_embeddings=seq_embeddings,
            seq_task_features=seq_task_features,
            fused_missing=fused_missing,
            fused_full=fused_full,
            full_mask=task_structure_mask,
            torch_module=torch_module,
        )
        predictions = self.task_head(self._maybe_detach_task_features(task_features))
        task_loss = self._compute_task_loss(predictions, [sample.target for sample in batch])

        if num_full_samples == 0:
            zero = task_loss * 0.0
            alignment = zero
            consistency = zero
            reconstruction = zero
            coverage_scale = 0.0
        else:
            projected_seq = self.seq_projection_head(seq_embeddings)
            projected_dyn = self.dyn_projection_head(dyn_embeddings)
            reconstructed_dyn = self.recon_head(fused_missing)
            alignment = info_nce_loss(
                projected_seq,
                projected_dyn,
                valid_mask=aux_md_mask,
            )
            consistency = consistency_loss(
                fused_missing,
                fused_full,
                mode=self.config.consistency_mode,
                valid_mask=aux_md_mask,
            )
            reconstruction = reconstruction_loss(
                reconstructed_dyn,
                dyn_embeddings,
                valid_mask=aux_md_mask,
            )
            coverage_scale = self._coverage_scale(num_full_samples, len(batch))

        total_loss = (
            task_loss
            + self.config.lambda_align * coverage_scale * alignment
            + self.config.lambda_cons * coverage_scale * consistency
            + self.config.lambda_recon * coverage_scale * reconstruction
        )
        return {
            "total_loss_tensor": total_loss,
            "task_loss_tensor": task_loss,
            "alignment_loss_tensor": alignment,
            "consistency_loss_tensor": consistency,
            "reconstruction_loss_tensor": reconstruction,
            "num_full_samples": num_full_samples,
            "num_seq_only_samples": num_seq_only_samples,
        }

    def _forward_pair_real(self, batch: Sequence[DownstreamStage1Sample], torch_module: Any) -> Dict[str, Any]:
        protein_sequences = [
            sample.sequence[: self.config.protein_max_sequence_length]
            for sample in batch
        ]
        peptide_sequences = [
            self._require_peptide_sequence(sample)[: self.config.peptide_max_sequence_length]
            for sample in batch
        ]
        batch_size = len(batch)
        protein_seq_output = self.sequence_encoder(protein_sequences)
        protein_seq_embeddings = _ensure_tensor_matrix(
            protein_seq_output.pooled_embedding,
            name="protein_seq_embeddings",
            torch_module=torch_module,
            device=self.config.device,
        )
        peptide_seq_output = self.sequence_encoder(peptide_sequences)
        peptide_seq_embeddings = _ensure_tensor_matrix(
            peptide_seq_output.pooled_embedding,
            name="peptide_seq_embeddings",
            torch_module=torch_module,
            device=self.config.device,
        )

        protein_dyn_embeddings, protein_full_mask, protein_aux_md_mask = self._encode_side_dyn_embeddings(
            batch,
            torch_module,
            side="protein",
            max_residues=self.config.protein_max_residues,
        )
        peptide_dyn_embeddings, peptide_full_mask, peptide_aux_md_mask = self._encode_side_dyn_embeddings(
            batch,
            torch_module,
            side="peptide",
            max_residues=self.config.peptide_max_residues,
        )

        protein_full_view = self.fusion_transformer(
            protein_seq_embeddings,
            protein_dyn_embeddings,
            has_dyn=protein_full_mask,
        )
        protein_missing_view = self.fusion_transformer(
            protein_seq_embeddings,
            protein_dyn_embeddings,
            has_dyn=[False] * batch_size,
        )
        peptide_full_view = self.fusion_transformer(
            peptide_seq_embeddings,
            peptide_dyn_embeddings,
            has_dyn=peptide_full_mask,
        )
        peptide_missing_view = self.fusion_transformer(
            peptide_seq_embeddings,
            peptide_dyn_embeddings,
            has_dyn=[False] * batch_size,
        )

        protein_fused_full = _ensure_tensor_matrix(
            protein_full_view.fused_pooled,
            name="protein_fused_full",
            torch_module=torch_module,
            device=self.config.device,
        )
        protein_fused_missing = _ensure_tensor_matrix(
            protein_missing_view.fused_pooled,
            name="protein_fused_missing",
            torch_module=torch_module,
            device=self.config.device,
        )
        peptide_fused_full = _ensure_tensor_matrix(
            peptide_full_view.fused_pooled,
            name="peptide_fused_full",
            torch_module=torch_module,
            device=self.config.device,
        )
        peptide_fused_missing = _ensure_tensor_matrix(
            peptide_missing_view.fused_pooled,
            name="peptide_fused_missing",
            torch_module=torch_module,
            device=self.config.device,
        )

        task_features = self._choose_pair_task_features(
            protein_fused_missing=protein_fused_missing,
            peptide_fused_missing=peptide_fused_missing,
            protein_fused_full=protein_fused_full,
            peptide_fused_full=peptide_fused_full,
            protein_full_mask=protein_full_mask,
            peptide_full_mask=peptide_full_mask,
            torch_module=torch_module,
        )
        predictions = self.task_head(self._maybe_detach_task_features(task_features))
        task_loss = self._compute_task_loss(predictions, [sample.target for sample in batch])

        pair_full_mask = [bool(left or right) for left, right in zip(protein_aux_md_mask, peptide_aux_md_mask)]
        num_full_samples = sum(int(value) for value in pair_full_mask)
        num_seq_only_samples = len(batch) - num_full_samples
        side_full_mask = protein_aux_md_mask + peptide_aux_md_mask
        valid_side_count = sum(int(value) for value in side_full_mask)
        if not any(side_full_mask):
            zero = task_loss * 0.0
            alignment = zero
            consistency = zero
            reconstruction = zero
            coverage_scale = 0.0
        else:
            side_seq_embeddings = torch_module.cat([protein_seq_embeddings, peptide_seq_embeddings], dim=0)
            side_dyn_embeddings = torch_module.cat([protein_dyn_embeddings, peptide_dyn_embeddings], dim=0)
            side_fused_full = torch_module.cat([protein_fused_full, peptide_fused_full], dim=0)
            side_fused_missing = torch_module.cat([protein_fused_missing, peptide_fused_missing], dim=0)
            projected_seq = self.seq_projection_head(side_seq_embeddings)
            projected_dyn = self.dyn_projection_head(side_dyn_embeddings)
            reconstructed_dyn = self.recon_head(side_fused_missing)
            alignment = info_nce_loss(
                projected_seq,
                projected_dyn,
                valid_mask=side_full_mask,
            )
            consistency = consistency_loss(
                side_fused_missing,
                side_fused_full,
                mode=self.config.consistency_mode,
                valid_mask=side_full_mask,
            )
            reconstruction = reconstruction_loss(
                reconstructed_dyn,
                side_dyn_embeddings,
                valid_mask=side_full_mask,
            )
            coverage_scale = self._coverage_scale(valid_side_count, len(side_full_mask))

        total_loss = (
            task_loss
            + self.config.lambda_align * coverage_scale * alignment
            + self.config.lambda_cons * coverage_scale * consistency
            + self.config.lambda_recon * coverage_scale * reconstruction
        )
        return {
            "total_loss_tensor": total_loss,
            "task_loss_tensor": task_loss,
            "alignment_loss_tensor": alignment,
            "consistency_loss_tensor": consistency,
            "reconstruction_loss_tensor": reconstruction,
            "num_full_samples": num_full_samples,
            "num_seq_only_samples": num_seq_only_samples,
        }

    def evaluate_dataset(self, dataset: DownstreamDataset) -> Dict[str, Any]:
        prediction_bundle = self._prediction_records_and_losses(dataset)
        evaluator = ProtMMLMEvaluator(task_name=self.task_name)
        metric_subset = self._primary_report_subset()
        control_subset_names = None
        if self.config.sequence_only_baseline:
            control_subset_names = ("seq_only",)
        report = evaluator.evaluate(
            prediction_bundle["records"],
            subset_names=control_subset_names,
        )
        report[metric_subset]["metrics"]["loss"] = float(prediction_bundle["total_loss"])
        report[metric_subset]["metrics"]["task_loss"] = float(prediction_bundle["task_loss"])
        report[metric_subset]["metrics"]["alignment_loss"] = float(prediction_bundle["alignment_loss"])
        report[metric_subset]["metrics"]["consistency_loss"] = float(prediction_bundle["consistency_loss"])
        report[metric_subset]["metrics"]["reconstruction_loss"] = float(prediction_bundle["reconstruction_loss"])
        return report

    def predict_dataset(self, dataset: DownstreamDataset) -> list[EvaluationRecord]:
        return list(self._prediction_records_and_losses(dataset)["records"])

    def _primary_report_subset(self) -> str:
        configured_subset = self.config.monitor_subset
        if configured_subset is not None and str(configured_subset).strip():
            return str(configured_subset).strip().lower()
        if self.config.sequence_only_baseline:
            return "seq_only"
        return "overall"

    def _prediction_records_and_losses(self, dataset: DownstreamDataset) -> Dict[str, Any]:
        if dataset.task_name != self.task_name:
            raise ValueError(
                f"Dataset task_name={dataset.task_name!r} does not match trainer task_name={self.task_name!r}."
            )
        if len(dataset) == 0:
            raise ValueError("Prediction requires at least one downstream sample.")

        self._set_module_modes(training=False)
        samples = list(dataset)
        batch_size = int(self.config.batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        records: List[EvaluationRecord] = []
        weighted_losses = {
            "task_loss": 0.0,
            "alignment_loss": 0.0,
            "consistency_loss": 0.0,
            "reconstruction_loss": 0.0,
            "total_loss": 0.0,
        }
        total_samples = 0
        for batch in _iter_sample_batches(samples, batch_size):
            prediction_bundle = self._predict_and_losses_real(batch)
            full_predictions = _predictions_for_metrics(
                prediction_bundle["full_predictions"],
                task_kind=self.task_kind,
            )
            raw_missing_predictions = prediction_bundle.get("missing_predictions")
            missing_predictions = (
                _predictions_for_metrics(raw_missing_predictions, task_kind=self.task_kind)
                if raw_missing_predictions is not None
                else [None] * len(batch)
            )
            for loss_name in weighted_losses:
                weighted_losses[loss_name] += float(prediction_bundle[loss_name]) * len(batch)
            total_samples += len(batch)

            for sample, full_prediction, missing_prediction in zip(batch, full_predictions, missing_predictions):
                modality_subset = _evaluation_modality_subset(
                    sample,
                    force_missing_dynamics_for_all_samples=self.config.force_missing_dynamics_for_all_samples,
                    force_natural_structure_only_for_all_samples=self.config.force_natural_structure_only_for_all_samples,
                )
                if (not self.config.sequence_only_baseline) and modality_subset != "seq_only":
                    records.append(
                        EvaluationRecord(
                            sample_id=sample.sample_id,
                            label=sample.target,
                            prediction=full_prediction,
                            task_name=self.task_name,
                            has_dyn=(modality_subset == "full"),
                            modality_subset=modality_subset,
                            seq_only_prediction=missing_prediction,
                            full_prediction=full_prediction,
                        )
                    )
                else:
                    prediction = missing_prediction if missing_prediction is not None else full_prediction
                    records.append(
                        EvaluationRecord(
                            sample_id=sample.sample_id,
                            label=sample.target,
                            prediction=prediction,
                            task_name=self.task_name,
                            has_dyn=False,
                            modality_subset="seq_only",
                        )
                    )

        return {
            "records": records,
            "task_loss": weighted_losses["task_loss"] / float(total_samples),
            "alignment_loss": weighted_losses["alignment_loss"] / float(total_samples),
            "consistency_loss": weighted_losses["consistency_loss"] / float(total_samples),
            "reconstruction_loss": weighted_losses["reconstruction_loss"] / float(total_samples),
            "total_loss": weighted_losses["total_loss"] / float(total_samples),
        }

    def fit(
        self,
        train_dataset: DownstreamDataset,
        *,
        validation_dataset: DownstreamDataset | None = None,
        test_dataset: DownstreamDataset | None = None,
        checkpoint_root: str | Path | None = None,
        max_epochs: int | None = None,
        validation_interval: int | None = None,
    ) -> Dict[str, Any]:
        if train_dataset.task_name != self.task_name:
            raise ValueError(
                f"Dataset task_name={train_dataset.task_name!r} does not match trainer task_name={self.task_name!r}."
            )
        if len(train_dataset) == 0:
            raise ValueError("fit requires at least one downstream sample.")
        if test_dataset is not None and test_dataset.task_name != self.task_name:
            raise ValueError(
                f"Test dataset task_name={test_dataset.task_name!r} does not match trainer task_name={self.task_name!r}."
            )

        resolved_validation_dataset = validation_dataset or train_dataset
        resolved_max_epochs = self.config.max_epochs_downstream if max_epochs is None else int(max_epochs)
        resolved_validation_interval = (
            self.config.validation_interval_downstream
            if validation_interval is None
            else int(validation_interval)
        )
        if resolved_max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {resolved_max_epochs}.")
        if resolved_validation_interval <= 0:
            raise ValueError(
                f"validation_interval must be positive, got {resolved_validation_interval}."
            )

        batch_size = int(self.config.batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if self.config.progress_log_interval_downstream <= 0:
            raise ValueError(
                "progress_log_interval_downstream must be positive, "
                f"got {self.config.progress_log_interval_downstream}."
            )
        if int(self.config.gradient_accumulation_steps) <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.config.gradient_accumulation_steps}."
            )
        start_epoch = self.current_epoch + 1
        completed_epochs = 0
        completed_train_steps = 0
        latest_validation: Dict[str, Any] | None = None
        latest_test: Dict[str, Any] | None = None
        latest_selection: Dict[str, Any] | None = None
        checkpoint_paths: Dict[str, str] = {}

        for epoch in range(start_epoch, resolved_max_epochs + 1):
            train_samples = self._prepare_training_samples(train_dataset, epoch=epoch)
            num_train_batches = _num_batches(len(train_samples), batch_size)
            epoch_batches = list(_iter_sample_batches(train_samples, batch_size))
            batch_iterator = _maybe_progress_bar(
                epoch_batches,
                enabled=self.config.show_progress_downstream,
                description=f"downstream {self.task_name} epoch {epoch}/{resolved_max_epochs}",
            )
            if self.config.show_progress_downstream and batch_iterator is epoch_batches:
                print(
                    f"downstream task={self.task_name} "
                    f"epoch={epoch}/{resolved_max_epochs} "
                    f"batches={num_train_batches}",
                    flush=True,
                )
            for batch_index, train_batch in enumerate(batch_iterator, start=1):
                step_result = self.train_step(train_batch)
                completed_train_steps += 1
                _update_progress(
                    batch_iterator,
                    step_result,
                    task_name=self.task_name,
                    epoch=epoch,
                    max_epochs=resolved_max_epochs,
                    batch_index=batch_index,
                    num_batches=num_train_batches,
                    enabled=self.config.show_progress_downstream,
                    log_interval=self.config.progress_log_interval_downstream,
                )
            accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
            if self._optimizer is not None and (completed_train_steps % accumulation_steps) != 0:
                parameters = list(self.trainable_parameters())
                if self.config.grad_clip > 0.0 and parameters:
                    nn_utils = importlib.import_module("torch.nn.utils")
                    nn_utils.clip_grad_norm_(parameters, self.config.grad_clip)
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)
            self.current_epoch = epoch
            completed_epochs += 1

            should_validate = (epoch % resolved_validation_interval == 0) or (epoch == resolved_max_epochs)
            if not should_validate:
                continue

            latest_validation = self.evaluate_dataset(resolved_validation_dataset)
            _print_epoch_report(
                prefix="epoch_val",
                epoch=epoch,
                report=latest_validation,
            )
            if test_dataset is not None:
                latest_test = self.evaluate_dataset(test_dataset)
                _print_epoch_report(
                    prefix="epoch_test",
                    epoch=epoch,
                    report=latest_test,
                )
            latest_selection = self.update_validation_state(
                epoch=epoch,
                val_metrics=latest_validation,
                checkpoint_root=checkpoint_root,
            )
            checkpoint_paths.update(latest_selection.get("checkpoint_paths", {}))
            if latest_selection["should_stop"]:
                break

        early_stopping = self.selection_state.early_stopping
        return {
            "stage": "downstream",
            "epochs_completed": completed_epochs,
            "train_steps_completed": completed_train_steps,
            "batch_size": batch_size,
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "start_epoch": start_epoch,
            "final_epoch": self.current_epoch,
            "validation_interval": resolved_validation_interval,
            "best_epoch": early_stopping.best_epoch,
            "best_metric": early_stopping.best_metric,
            "is_seq_only_baseline": bool(self.config.sequence_only_baseline),
            "monitor_subset": self._primary_report_subset(),
            "monitor_name": early_stopping.monitor_name,
            "should_stop": early_stopping.should_stop,
            "stop_reason": early_stopping.stop_reason,
            "last_total_loss": None if self.last_step_result is None else self.last_step_result.total_loss,
            "last_validation_metric": None
            if latest_validation is None
            else extract_metric(
                latest_validation,
                subset=self._primary_report_subset(),
                monitor_name=early_stopping.monitor_name,
            ),
            "last_test_metric": None
            if latest_test is None
            else extract_metric(
                latest_test,
                subset=self._primary_report_subset(),
                monitor_name=early_stopping.monitor_name,
            ),
            "checkpoint_paths": checkpoint_paths,
        }

    def trainable_parameters(self) -> Sequence[Any]:
        parameters: list[Any] = []
        seen: set[int] = set()
        if self.config.train_only_task_head:
            components = [self.task_head]
        elif self.config.sequence_only_baseline:
            components = [
                self.sequence_encoder,
                self.task_head,
            ]
        else:
            components = [
                self.sequence_encoder,
                self.structure_encoder,
                self.fusion_transformer,
                self.seq_projection_head,
                self.dyn_projection_head,
                self.recon_head,
                self.seq_task_adapter,
                self.single_task_residual_adapter,
                self.task_head,
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
            "stage": "downstream",
            "checkpoint_kind": checkpoint_kind,
            "backend_mode": self.backend_mode,
            "task_name": self.task_name,
            "task_kind": self.task_kind,
            "epoch": self.current_epoch if epoch is None else epoch,
            "monitor_name": monitor_name,
            "val_metrics": val_metrics,
            "global_step": self.global_step,
            "config": asdict(self.config),
            "config_summary": self.config_summary(),
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

        if payload.get("stage") != "downstream":
            raise ValueError(f"Checkpoint stage={payload.get('stage')!r} is not compatible with downstream trainer.")
        if payload.get("backend_mode") != self.backend_mode:
            raise ValueError(
                f"Checkpoint backend_mode={payload.get('backend_mode')!r} does not match trainer backend_mode={self.backend_mode!r}."
            )
        if _normalize_task_name(payload.get("task_name")) != self.task_name:
            raise ValueError(
                f"Checkpoint task_name={payload.get('task_name')!r} does not match trainer task_name={self.task_name!r}."
            )

        self.global_step = int(payload.get("global_step", 0))
        epoch = payload.get("epoch")
        if epoch is not None:
            self.current_epoch = int(epoch)
        selection_state = payload.get("selection_state")
        if isinstance(selection_state, dict):
            self.selection_state = DownstreamSelectionState.from_dict(selection_state)
        last_step_result = payload.get("last_step_result")
        if isinstance(last_step_result, dict):
            self.last_step_result = DownstreamStepResult(
                task_name=str(last_step_result["task_name"]),
                task_kind=str(last_step_result["task_kind"]),
                total_loss=float(last_step_result["total_loss"]),
                task_loss=float(last_step_result["task_loss"]),
                alignment_loss=float(last_step_result["alignment_loss"]),
                consistency_loss=float(last_step_result["consistency_loss"]),
                reconstruction_loss=float(last_step_result["reconstruction_loss"]),
                batch_size=int(last_step_result["batch_size"]),
                num_full_samples=int(last_step_result["num_full_samples"]),
                num_seq_only_samples=int(last_step_result["num_seq_only_samples"]),
                global_step=int(last_step_result["global_step"]),
            )
        else:
            self.last_step_result = None

        self._load_model_state(payload.get("model_state") or {})
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            self._optimizer = self._build_optimizer(torch)
            self._optimizer.load_state_dict(optimizer_state)
        return payload

    def load_pretrain_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        path = Path(checkpoint_path)
        torch = importlib.import_module("torch")
        payload = torch.load(path, map_location=self.config.device)

        if payload.get("stage") != "pretrain":
            raise ValueError(
                f"Checkpoint stage={payload.get('stage')!r} is not compatible with pretrain initialization."
            )
        backend_mode = payload.get("backend_mode")
        if backend_mode != self.backend_mode:
            raise ValueError(
                f"Checkpoint backend_mode={backend_mode!r} does not match trainer backend_mode={self.backend_mode!r}."
            )

        model_state = payload.get("model_state") or {}
        if not isinstance(model_state, dict):
            raise ValueError("Pretrain checkpoint model_state must be a mapping.")
        payload["load_summary"] = self._load_pretrain_model_state(model_state)
        return payload

    def update_validation_state(
        self,
        *,
        epoch: int,
        val_metrics: Dict[str, Any],
        checkpoint_root: str | Path | None = None,
    ) -> Dict[str, Any]:
        if epoch <= 0:
            raise ValueError(f"epoch must be positive, got {epoch}.")

        early_stopping = self.selection_state.early_stopping
        monitor_name = early_stopping.monitor_name
        primary_subset = self._primary_report_subset()
        overall_metric = extract_metric(val_metrics, subset=primary_subset, monitor_name=monitor_name)
        if overall_metric is None:
            raise ValueError(f"Validation metrics do not contain {primary_subset} {monitor_name!r}.")

        seq_only_metric = extract_metric(val_metrics, subset="seq_only", monitor_name=monitor_name)
        improved_overall = is_better(
            overall_metric,
            early_stopping.best_metric,
            mode=early_stopping.monitor_mode,
            min_delta=self.config.min_delta_downstream,
        )
        if improved_overall:
            early_stopping.best_metric = overall_metric
            early_stopping.best_epoch = epoch
            early_stopping.bad_epochs = 0
            early_stopping.should_stop = False
            early_stopping.stop_reason = None
        elif epoch >= self.config.min_epochs_downstream:
            early_stopping.bad_epochs += 1

        if epoch >= self.config.max_epochs_downstream:
            early_stopping.should_stop = True
            early_stopping.stop_reason = "max_epochs"
        elif epoch >= self.config.min_epochs_downstream and early_stopping.bad_epochs >= self.config.patience_downstream:
            early_stopping.should_stop = True
            early_stopping.stop_reason = "patience"

        guarded_worse = significantly_worse(
            seq_only_metric,
            self.selection_state.best_guarded_seq_only_metric,
            mode=early_stopping.monitor_mode,
            tolerance=self.config.seq_guard_tolerance,
        )
        update_guarded = improved_overall and not guarded_worse
        if update_guarded:
            self.selection_state.best_guarded_metric = overall_metric
            self.selection_state.best_guarded_epoch = epoch
            self.selection_state.best_guarded_seq_only_metric = seq_only_metric

        should_update_single_best = improved_overall and (
            (not self.config.save_best_guarded_checkpoint) or update_guarded
        )

        self.selection_state.last_epoch = epoch
        checkpoint_paths: Dict[str, str] = {}
        if (checkpoint_root is not None or self.config.checkpoint_path is not None) and should_update_single_best:
            checkpoint_dir = resolve_checkpoint_dir(
                checkpoint_root if checkpoint_root is not None else str(self.config.checkpoint_path)
            )
            best_path = self.save_checkpoint(
                checkpoint_dir / "downstream_best_overall.ckpt",
                epoch=epoch,
                monitor_name=monitor_name,
                val_metrics=val_metrics,
                checkpoint_kind="best_overall",
            )
            checkpoint_paths["best_overall"] = str(best_path)

        return {
            "stage": "downstream",
            "epoch": epoch,
            "monitor_name": monitor_name,
            "overall_metric": overall_metric,
            "seq_only_metric": seq_only_metric,
            "improved_overall": improved_overall,
            "updated_guarded": update_guarded,
            "should_update_single_best": should_update_single_best,
            "should_stop": early_stopping.should_stop,
            "stop_reason": early_stopping.stop_reason,
            "checkpoint_paths": checkpoint_paths,
        }

    def config_summary(self) -> Dict[str, Any]:
        summary = asdict(self.config)
        summary["task_kind"] = self.task_kind
        summary["backend_mode"] = self.backend_mode
        summary["modality_source_condition"] = self._modality_source_condition()
        summary["initialization_mode"] = (
            "initialize_from_pretrain" if self.config.pretrain_checkpoint_path else "random_init"
        )
        summary["aux_loss_weights"] = {
            "lambda_align": self.config.lambda_align,
            "lambda_cons": self.config.lambda_cons,
            "lambda_recon": self.config.lambda_recon,
        }
        summary["train_only_task_head"] = bool(self.config.train_only_task_head)
        summary["task_head_type"] = self.config.task_head_type
        summary["task_head_hidden_dims"] = tuple(self.config.task_head_hidden_dims)
        summary["task_head_dropout"] = float(self.config.task_head_dropout)
        summary["multilabel_pos_weight_mode"] = self.config.multilabel_pos_weight_mode
        summary["multilabel_max_pos_weight"] = float(self.config.multilabel_max_pos_weight)
        return summary

    def component_summary(self) -> Dict[str, str]:
        if self.config.sequence_only_baseline:
            return {
                "sequence_encoder": "real",
                "structure_encoder": "disabled_seq_only",
                "fusion_transformer": "disabled_seq_only",
                "seq_projection_head": "disabled_seq_only",
                "dyn_projection_head": "disabled_seq_only",
                "recon_head": "disabled_seq_only",
                "seq_task_adapter": "disabled_seq_only",
                "task_head": self._task_head_summary(),
                "optimizer": self.config.optimizer,
                "train_only_task_head": str(bool(self.config.train_only_task_head)),
                "task_head_type": self.config.task_head_type,
                "task_head_hidden_dims": str(tuple(self.config.task_head_hidden_dims)),
                "task_head_dropout": str(float(self.config.task_head_dropout)),
                "sequence_encoder_trainable": str(bool(self.config.sequence_encoder_trainable)),
                "modality_source_condition": self._modality_source_condition(),
                "pretrain_checkpoint_loaded": str(bool(self.config.pretrain_checkpoint_path)),
                "lambda_align": str(self.config.lambda_align),
                "lambda_cons": str(self.config.lambda_cons),
                "lambda_recon": str(self.config.lambda_recon),
            }
        summary = {
            "sequence_encoder": "real",
            "structure_encoder": "real",
            "fusion_transformer": "real",
            "seq_projection_head": "real",
            "dyn_projection_head": "real",
            "recon_head": "real",
            "seq_task_adapter": str(self.config.single_missing_task_fallback),
            "single_task_residual_adapter": self.config.single_task_feature_mode,
            "task_head": self._task_head_summary(),
            "optimizer": self.config.optimizer,
            "train_only_task_head": str(bool(self.config.train_only_task_head)),
            "task_head_type": self.config.task_head_type,
            "task_head_hidden_dims": str(tuple(self.config.task_head_hidden_dims)),
            "task_head_dropout": str(float(self.config.task_head_dropout)),
            "multilabel_pos_weight_mode": self.config.multilabel_pos_weight_mode,
            "multilabel_max_pos_weight": str(float(self.config.multilabel_max_pos_weight)),
            "sequence_encoder_trainable": str(bool(self.config.sequence_encoder_trainable)),
            "structure_encoder_trainable": str(bool(self.config.structure_encoder_trainable)),
            "fusion_transformer_trainable": str(bool(self.config.fusion_transformer_trainable)),
            "modality_source_condition": self._modality_source_condition(),
            "pretrain_checkpoint_loaded": str(bool(self.config.pretrain_checkpoint_path)),
            "lambda_align": str(self.config.lambda_align),
            "lambda_cons": str(self.config.lambda_cons),
            "lambda_recon": str(self.config.lambda_recon),
        }
        if self.config.force_missing_dynamics_for_all_samples:
            summary["modality_mode"] = "force_missing_dynamics_for_all_samples"
        if self.config.force_natural_structure_only_for_all_samples:
            summary["modality_mode"] = "force_natural_structure_only_for_all_samples"
        return summary

    def _task_head_summary(self) -> str:
        if self.task_kind == "binary":
            return "binary_mlp" if isinstance(self.task_head, BinaryMLPClassificationHead) else "binary_linear"
        if self.task_kind == "regression":
            return "regression_mlp" if isinstance(self.task_head, MLPRegressionHead) else "regression_linear"
        if self.task_kind == "multilabel":
            return "multilabel_linear"
        return "unknown"

    def _modality_source_condition(self) -> str:
        if self.config.sequence_only_baseline:
            return "seq_only_baseline"
        if self.config.force_missing_dynamics_for_all_samples:
            return "force_missing_dynamics"
        if self.config.force_natural_structure_only_for_all_samples:
            return "force_natural_structure_only"
        return "full"

    def _is_pair_batch(self, batch: Sequence[DownstreamStage1Sample]) -> bool:
        peptide_flags = [bool(sample.peptide_sequence) for sample in batch]
        if all(peptide_flags):
            return True
        if not any(peptide_flags):
            if self._task_expects_pair_samples():
                missing_peptides = [sample.sample_id for sample in batch]
                preview = ", ".join(missing_peptides[:5])
                raise ValueError(
                    f"{self.task_name} requires pair-level samples with peptide_sequence. "
                    f"First missing sample_id(s): {preview}"
                )
            return False
        missing_peptides = [sample.sample_id for sample, has_peptide in zip(batch, peptide_flags) if not has_peptide]
        preview = ", ".join(missing_peptides[:5])
        raise ValueError(
            "Mixed pair and single samples are not supported in the same batch. "
            f"First missing peptide sample_id(s): {preview}"
        )

    def _task_expects_pair_samples(self) -> bool:
        return self.task_name == "ppikb"

    def _require_peptide_sequence(self, sample: DownstreamStage1Sample) -> str:
        if not sample.peptide_sequence:
            raise ValueError(f"{self.task_name} sample {sample.sample_id!r} is missing peptide_sequence.")
        return sample.peptide_sequence

    def _encode_single_dyn_embeddings(
        self,
        batch: Sequence[DownstreamStage1Sample],
        torch_module: Any,
    ) -> tuple[Any, list[bool], list[bool]]:
        dyn_rows: List[Any] = []
        task_structure_mask: list[bool] = []
        aux_md_mask: list[bool] = []
        structure_backend = getattr(self.structure_encoder, "_backend", None)
        for sample in batch:
            task_has_structure, aux_has_md, nature_path, md_path = self._resolve_control_structure_source(
                has_dyn=bool(sample.has_dyn),
                nature_path=sample.nature_path,
                md_path=sample.md_path,
            )
            task_structure_mask.append(task_has_structure)
            aux_md_mask.append(aux_has_md)
            if task_has_structure:
                if not nature_path:
                    raise ValueError(
                        f"Structured sample {sample.sample_id!r} is missing nature_path."
                    )
                if aux_has_md and not md_path:
                    raise ValueError(
                        f"Full sample {sample.sample_id!r} is missing md_path."
                    )
                if hasattr(structure_backend, "note_paths"):
                    structure_backend.note_paths(
                        nature_path=nature_path,
                        md_path=md_path,
                    )
                dyn_output = self.structure_encoder.encode_paths(
                    nature_path=nature_path,
                    md_path=md_path,
                    max_residues=self.config.max_residues,
                    max_frames=self.config.max_frames,
                )
                dyn_rows.append(
                    _first_tensor_row(
                        dyn_output.pooled_embedding,
                        name="dyn_embedding",
                        torch_module=torch_module,
                        device=self.config.device,
                    )
                )
            else:
                dyn_rows.append(
                    torch_module.tensor(
                        _deterministic_vector(
                            f"downstream-dyn-placeholder:{sample.sample_id}:{sample.sequence_hash}",
                            self.config.embedding_dim,
                        ),
                        dtype=torch_module.float32,
                        device=self.config.device,
                    )
                )
        return torch_module.stack(dyn_rows, dim=0), task_structure_mask, aux_md_mask

    def _encode_side_dyn_embeddings(
        self,
        batch: Sequence[DownstreamStage1Sample],
        torch_module: Any,
        *,
        side: str,
        max_residues: int,
    ) -> tuple[Any, list[bool], list[bool]]:
        dyn_rows: List[Any] = []
        task_structure_mask: list[bool] = []
        aux_md_mask: list[bool] = []
        structure_backend = getattr(self.structure_encoder, "_backend", None)
        for sample in batch:
            has_dyn, nature_path, md_path, cache_path, hash_value = self._side_dyn_metadata(sample, side=side)
            task_has_structure, aux_has_md, resolved_nature_path, resolved_md_path = self._resolve_control_structure_source(
                has_dyn=has_dyn,
                nature_path=nature_path,
                md_path=md_path,
            )
            task_structure_mask.append(task_has_structure)
            aux_md_mask.append(aux_has_md)
            if task_has_structure:
                if not resolved_nature_path:
                    raise ValueError(
                        f"Structured {side} side for sample {sample.sample_id!r} is missing nature_path."
                    )
                if aux_has_md and not resolved_md_path:
                    raise ValueError(
                        f"Full {side} side for sample {sample.sample_id!r} is missing md_path."
                    )
                if hasattr(structure_backend, "note_paths"):
                    structure_backend.note_paths(
                        nature_path=resolved_nature_path,
                        md_path=resolved_md_path,
                    )
                dyn_output = self.structure_encoder.encode_paths(
                    nature_path=resolved_nature_path,
                    md_path=resolved_md_path,
                    max_residues=max_residues,
                    max_frames=self.config.max_frames,
                )
                dyn_rows.append(
                    _first_tensor_row(
                        dyn_output.pooled_embedding,
                        name=f"{side}_dyn_embedding",
                        torch_module=torch_module,
                        device=self.config.device,
                    )
                )
            else:
                dyn_rows.append(
                    torch_module.tensor(
                        _deterministic_vector(
                            f"downstream-dyn-placeholder:{side}:{sample.sample_id}:{hash_value}",
                            self.config.embedding_dim,
                        ),
                        dtype=torch_module.float32,
                        device=self.config.device,
                    )
                )
        return torch_module.stack(dyn_rows, dim=0), task_structure_mask, aux_md_mask

    def _resolve_control_structure_source(
        self,
        *,
        has_dyn: bool,
        nature_path: str | None,
        md_path: str | None,
    ) -> tuple[bool, bool, str | None, str | None]:
        if self.config.force_missing_dynamics_for_all_samples:
            return False, False, None, None
        if self.config.force_natural_structure_only_for_all_samples:
            return bool(nature_path), False, nature_path, None
        if has_dyn:
            return True, True, nature_path, md_path
        return False, False, None, None

    def _side_dyn_metadata(
        self,
        sample: DownstreamStage1Sample,
        *,
        side: str,
    ) -> tuple[bool, str | None, str | None, str | None, str]:
        if side == "protein":
            return (
                bool(sample.has_dyn),
                sample.nature_path,
                sample.md_path,
                None,
                sample.sequence_hash,
            )
        if side == "peptide":
            return (
                bool(sample.peptide_has_dyn),
                sample.peptide_nature_path,
                sample.peptide_md_path,
                None,
                sample.peptide_sequence_hash or sample.pair_key or sample.sample_id,
            )
        raise ValueError(f"Unsupported pair side: {side}")

    def _configure_module_trainability(self) -> None:
        if self.config.sequence_encoder_trainable:
            self.sequence_encoder.unfreeze_backbone()
        else:
            self.sequence_encoder.freeze_backbone()
        if self.config.structure_encoder_trainable:
            self.structure_encoder.unfreeze_backbone()
        else:
            self.structure_encoder.freeze_backbone()
        if self.config.fusion_transformer_trainable:
            self.fusion_transformer.unfreeze_backbone()
        else:
            self.fusion_transformer.freeze_backbone()
        if self.config.train_only_task_head:
            for component in (
                self.sequence_encoder,
                self.structure_encoder,
                self.fusion_transformer,
                self.seq_projection_head,
                self.dyn_projection_head,
                self.recon_head,
                self.seq_task_adapter,
                self.single_task_residual_adapter,
            ):
                self._set_component_requires_grad(component, requires_grad=False)
            self._set_component_requires_grad(self.task_head, requires_grad=True)

    def _set_module_modes(self, *, training: bool) -> None:
        head_only = bool(self.config.train_only_task_head)
        self._set_sequence_encoder_mode(training=training and not head_only)
        self._set_component_mode(
            self.structure_encoder,
            training=training and self.config.structure_encoder_trainable and not head_only,
        )
        self._set_component_mode(
            self.fusion_transformer,
            training=training and self.config.fusion_transformer_trainable and not head_only,
        )
        self._set_component_mode(self.seq_projection_head, training=training and not head_only)
        self._set_component_mode(self.dyn_projection_head, training=training and not head_only)
        self._set_component_mode(self.recon_head, training=training and not head_only)
        self._set_component_mode(self.seq_task_adapter, training=training and not head_only)
        self._set_component_mode(self.single_task_residual_adapter, training=training and not head_only)
        self._set_task_head_mode(training=training)

    def _set_sequence_encoder_mode(self, *, training: bool) -> None:
        if training and self.config.sequence_encoder_trainable:
            self.sequence_encoder.train(True)
            return
        self.sequence_encoder.eval()

    def _set_task_head_mode(self, *, training: bool) -> None:
        if training and hasattr(self.task_head, "train"):
            self.task_head.train(True)
        elif hasattr(self.task_head, "eval"):
            self.task_head.eval()

    @staticmethod
    def _set_component_mode(component: Any, *, training: bool) -> None:
        if training and hasattr(component, "train"):
            component.train(True)
        elif hasattr(component, "eval"):
            component.eval()

    @staticmethod
    def _set_component_requires_grad(component: Any, *, requires_grad: bool) -> None:
        if not hasattr(component, "parameters"):
            return
        for parameter in component.parameters():
            if hasattr(parameter, "requires_grad"):
                parameter.requires_grad = requires_grad

    def _build_task_head(self, embedding_dim: int) -> Any:
        single_feature_mode = self.config.single_task_feature_mode.strip().lower()
        is_pair_task = self._task_expects_pair_samples()
        single_mixed_input_dim = (
            self.sequence_output_dim
            if single_feature_mode == "raw_seq_residual" and (not self.config.sequence_only_baseline) and not is_pair_task
            else embedding_dim
        )
        task_head_type = self._normalized_task_head_type()
        if self.task_kind == "binary":
            if self.config.sequence_only_baseline:
                input_dim = self.sequence_output_dim * 2 if is_pair_task else self.sequence_output_dim
            else:
                input_dim = embedding_dim * 2 if is_pair_task else single_mixed_input_dim
            if task_head_type == "mlp":
                return BinaryMLPClassificationHead(
                    input_dim,
                    hidden_dims=self.config.task_head_hidden_dims,
                    dropout=self.config.task_head_dropout,
                    seed=230,
                )
            return BinaryClassificationHead(input_dim, seed=220)
        if self.task_kind == "regression":
            if self.config.sequence_only_baseline:
                input_dim = self.sequence_output_dim * 2 if is_pair_task else self.sequence_output_dim
                if self.task_name == "ppikb":
                    return MLPRegressionHead(input_dim, hidden_dims=(512, 128), dropout=0.1, seed=520)
            else:
                input_dim = embedding_dim * 2 if is_pair_task else single_mixed_input_dim
                if self._use_mlp_task_head_for_mixed_regression():
                    return MLPRegressionHead(input_dim, hidden_dims=(512, 128), dropout=0.1, seed=620)
            return RegressionHead(input_dim, seed=420)
        if self.task_kind == "multilabel":
            num_labels = _infer_multilabel_dim(self.config.manifest_path, self.task_name)
            if self.config.sequence_only_baseline:
                input_dim = self.sequence_output_dim * 2 if is_pair_task else self.sequence_output_dim
            else:
                input_dim = embedding_dim * 2 if is_pair_task else single_mixed_input_dim
            return MultiLabelClassificationHead(input_dim, num_labels=num_labels, seed=320)
        raise ValueError(f"Unsupported task_kind: {self.task_kind}")

    def _maybe_detach_task_features(self, features: Any) -> Any:
        if self.config.train_only_task_head and hasattr(features, "detach"):
            return features.detach()
        return features

    def _compute_task_loss(self, predictions: Any, targets: Sequence[Any]) -> Any:
        if self.task_kind == "binary":
            return binary_classification_loss(predictions, [float(target) for target in targets])
        if self.task_kind == "regression":
            return regression_loss(
                predictions,
                [float(target) for target in targets],
                mode=self.config.regression_loss_mode,
                delta=self.config.regression_delta,
            )
        if self.task_kind == "multilabel":
            return multilabel_classification_loss(
                predictions,
                targets,
                pos_weight_mode=self.config.multilabel_pos_weight_mode,
                max_pos_weight=self.config.multilabel_max_pos_weight,
            )
        raise ValueError(f"Unsupported task_kind: {self.task_kind}")

    def _build_optimizer(self, torch_module: Any) -> Any:
        parameters = list(self.trainable_parameters())
        if not parameters:
            raise RuntimeError("No trainable parameters were registered for the real downstream path.")
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

    def _normalized_task_head_type(self) -> str:
        task_head_type = self.config.task_head_type.strip().lower()
        if task_head_type not in {"auto", "linear", "mlp"}:
            raise ValueError(f"Unsupported task_head_type={self.config.task_head_type!r}. Expected auto, linear, or mlp.")
        return task_head_type

    def _use_mlp_task_head_for_mixed_regression(self) -> bool:
        if self.task_kind != "regression" or self.config.sequence_only_baseline:
            return False
        task_head_type = self._normalized_task_head_type()
        if task_head_type == "mlp":
            return True
        if task_head_type == "linear":
            return False
        return self.task_name == "ppikb"

    def _coverage_scale(self, valid_count: int, total_count: int) -> float:
        if total_count <= 0:
            return 0.0
        fraction = float(valid_count) / float(total_count)
        if self.config.aux_loss_reweight_mode.strip().lower() == "off":
            return 1.0
        if fraction < float(self.config.min_full_fraction_for_aux):
            return 0.0
        return fraction

    def _choose_single_task_features(
        self,
        *,
        seq_embeddings: Any,
        seq_task_features: Any,
        fused_missing: Any,
        fused_full: Any,
        full_mask: list[bool],
        torch_module: Any,
    ) -> Any:
        mode = self.config.mixed_task_mode.strip().lower()
        if mode == "always_fused":
            return fused_full
        fallback_mode = self.config.single_missing_task_fallback.strip().lower()
        feature_mode = self.config.single_task_feature_mode.strip().lower()
        if feature_mode == "raw_seq_residual":
            missing_delta = self.single_task_residual_adapter(fused_missing - seq_task_features)
            full_delta = self.single_task_residual_adapter(fused_full - seq_task_features)
            full_task_features = seq_embeddings + full_delta
            missing_task_features = seq_embeddings + missing_delta
            if fallback_mode == "seq_adapter" and not self.config.force_missing_dynamics_for_all_samples:
                missing_task_features = seq_embeddings
            return torch_where_rows(full_mask, full_task_features, missing_task_features)
        if fallback_mode == "seq_adapter":
            full_task_features = seq_task_features + (fused_full - fused_missing)
            return torch_where_rows(full_mask, full_task_features, seq_task_features)
        return torch_where_rows(full_mask, fused_full, fused_missing)

    def _choose_pair_task_features(
        self,
        *,
        protein_fused_missing: Any,
        peptide_fused_missing: Any,
        protein_fused_full: Any,
        peptide_fused_full: Any,
        protein_full_mask: list[bool],
        peptide_full_mask: list[bool],
        torch_module: Any,
    ) -> Any:
        pair_missing = torch_module.cat([protein_fused_missing, peptide_fused_missing], dim=-1)
        pair_fused = torch_module.cat([protein_fused_full, peptide_fused_full], dim=-1)
        mode = self.config.mixed_task_mode.strip().lower()
        if mode == "always_fused":
            return pair_fused

        handling = self.config.partial_pair_handling.strip().lower()
        if handling == "fused":
            pair_mask = [bool(left or right) for left, right in zip(protein_full_mask, peptide_full_mask)]
            return torch_where_rows(pair_mask, pair_fused, pair_missing)

        protein_selected = torch_where_rows(protein_full_mask, protein_fused_full, protein_fused_missing)
        peptide_selected = torch_where_rows(peptide_full_mask, peptide_fused_full, peptide_fused_missing)
        return torch_module.cat([protein_selected, peptide_selected], dim=-1)

    def _prepare_training_samples(
        self,
        train_dataset: DownstreamDataset,
        *,
        epoch: int,
    ) -> list[DownstreamStage1Sample]:
        samples = list(train_dataset)
        if (not self.config.sequence_only_baseline) and not self._task_expects_pair_samples():
            oversample_factor = max(1, int(self.config.single_full_sample_oversample_factor))
            if oversample_factor > 1:
                full_samples = [sample for sample in samples if bool(sample.has_dyn)]
                if full_samples:
                    samples.extend(full_samples * (oversample_factor - 1))
        if self.config.shuffle_train_each_epoch:
            import random

            rng = random.Random(int(self.config.train_shuffle_seed) + int(epoch))
            rng.shuffle(samples)
        return samples

    def _sync_optimizer_parameters(self) -> None:
        if self._optimizer is None:
            return
        existing_parameter_ids = {
            id(parameter)
            for group in self._optimizer.param_groups
            for parameter in group.get("params", [])
        }
        new_parameters = [
            parameter
            for parameter in self.trainable_parameters()
            if id(parameter) not in existing_parameter_ids
        ]
        if not new_parameters:
            return
        self._optimizer.add_param_group(
            {
                "params": new_parameters,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            }
        )

    def _model_state_dict(self) -> Dict[str, Any]:
        if self.config.sequence_only_baseline:
            return {
                "sequence_encoder": self.sequence_encoder.state_dict(),
                "structure_encoder": {},
                "fusion_transformer": {},
                "seq_projection_head": {},
                "dyn_projection_head": {},
                "recon_head": {},
                "seq_task_adapter": {},
                "single_task_residual_adapter": {},
                "task_head": self.task_head.state_dict() if hasattr(self.task_head, "state_dict") else {},
            }
        return {
            "sequence_encoder": self.sequence_encoder.state_dict(),
            "structure_encoder": self.structure_encoder.state_dict(),
            "fusion_transformer": self.fusion_transformer.state_dict(),
            "seq_projection_head": self.seq_projection_head.state_dict(),
            "dyn_projection_head": self.dyn_projection_head.state_dict(),
            "recon_head": self.recon_head.state_dict(),
            "seq_task_adapter": self.seq_task_adapter.state_dict(),
            "single_task_residual_adapter": self.single_task_residual_adapter.state_dict(),
            "task_head": self.task_head.state_dict() if hasattr(self.task_head, "state_dict") else {},
        }

    def _load_model_state(self, model_state: Dict[str, Any]) -> None:
        self.sequence_encoder.load_state_dict(model_state.get("sequence_encoder", {}))
        if self.config.sequence_only_baseline:
            if hasattr(self.task_head, "load_state_dict"):
                task_head_state = model_state.get("task_head", {})
                self._validate_task_head_state(task_head_state)
                self.task_head.load_state_dict(task_head_state)
            return
        self.structure_encoder.load_state_dict(model_state.get("structure_encoder", {}))
        self.fusion_transformer.load_state_dict(model_state.get("fusion_transformer", {}))
        self.seq_projection_head.load_state_dict(model_state.get("seq_projection_head", {}))
        self.dyn_projection_head.load_state_dict(model_state.get("dyn_projection_head", {}))
        self.recon_head.load_state_dict(model_state.get("recon_head", {}))
        self.seq_task_adapter.load_state_dict(model_state.get("seq_task_adapter", {}))
        self.single_task_residual_adapter.load_state_dict(model_state.get("single_task_residual_adapter", {}))
        if hasattr(self.task_head, "load_state_dict"):
            task_head_state = model_state.get("task_head", {})
            self._validate_task_head_state(task_head_state)
            self.task_head.load_state_dict(task_head_state)

    def _load_pretrain_model_state(self, model_state: Dict[str, Any]) -> Dict[str, str]:
        load_summary: Dict[str, str] = {}
        sequence_state = model_state.get("sequence_encoder")
        if not sequence_state:
            raise ValueError("Pretrain checkpoint is missing required component 'sequence_encoder'.")
        self.sequence_encoder.load_state_dict(sequence_state)
        load_summary["sequence_encoder"] = "loaded"

        if self.config.sequence_only_baseline:
            load_summary["structure_encoder"] = "skipped_seq_only"
            load_summary["fusion_transformer"] = "skipped_seq_only"
            load_summary["seq_projection_head"] = "skipped_seq_only"
            load_summary["dyn_projection_head"] = "skipped_seq_only"
            load_summary["recon_head"] = "skipped_seq_only"
            load_summary["seq_task_adapter"] = "skipped_seq_only"
            load_summary["single_task_residual_adapter"] = "skipped_seq_only"
            load_summary["task_head"] = "not_loaded_from_pretrain"
            return load_summary

        for component_name, component in (
            ("structure_encoder", self.structure_encoder),
            ("fusion_transformer", self.fusion_transformer),
            ("seq_projection_head", self.seq_projection_head),
            ("dyn_projection_head", self.dyn_projection_head),
            ("recon_head", self.recon_head),
        ):
            component_state = model_state.get(component_name)
            if not component_state:
                raise ValueError(
                    f"Pretrain checkpoint is missing required multimodal component {component_name!r}."
                )
            component.load_state_dict(component_state)
            load_summary[component_name] = "loaded"

        load_summary["seq_task_adapter"] = "not_loaded_from_pretrain"
        load_summary["single_task_residual_adapter"] = "not_loaded_from_pretrain"
        load_summary["task_head"] = "not_loaded_from_pretrain"
        return load_summary

    def _validate_task_head_state(self, task_head_state: Any) -> None:
        if not isinstance(task_head_state, dict):
            return
        expected_input_dim = getattr(self.task_head, "input_dim", None)
        if expected_input_dim is None:
            return
        torch_layer_state = task_head_state.get("torch_layer")
        if isinstance(torch_layer_state, dict):
            if "weight" not in torch_layer_state:
                raise ValueError(
                    "Checkpoint task_head state is not a valid Linear layer state."
                )
            weight = torch_layer_state["weight"]
            observed_input_dim = int(weight.shape[1])
            if observed_input_dim != int(expected_input_dim):
                raise ValueError(
                    f"Checkpoint task_head input dim {observed_input_dim} is incompatible with "
                    f"current {self.task_name} input dim {expected_input_dim}. "
                    "Pair-level target+peptide training requires a fresh compatible downstream head."
                )
            if hasattr(self.task_head, "_torch_mlp"):
                raise ValueError(
                    "Checkpoint task_head uses a linear head, but current configuration expects an MLP head. Retrain downstream or load a compatible checkpoint."
                )
            return

        torch_mlp_state = task_head_state.get("torch_mlp")
        if isinstance(torch_mlp_state, dict):
            first_weight = torch_mlp_state.get("0.weight")
            if first_weight is None:
                raise ValueError("Checkpoint task_head MLP state is missing first layer weight.")
            observed_input_dim = int(first_weight.shape[1])
            if observed_input_dim != int(expected_input_dim):
                raise ValueError(
                    f"Checkpoint task_head input dim {observed_input_dim} is incompatible with current {self.task_name} input dim {expected_input_dim}."
                )
            if hasattr(self.task_head, "_torch_layer"):
                raise ValueError(
                    "Checkpoint task_head uses an MLP head, but current configuration expects a linear head. Retrain downstream or load a compatible checkpoint."
                )
            return

    def _predict_and_losses_real(self, batch: Sequence[DownstreamStage1Sample]) -> Dict[str, Any]:
        if self.config.sequence_only_baseline:
            return self._predict_and_losses_seq_only_real(batch)
        if self._is_pair_batch(batch):
            return self._predict_and_losses_pair_real(batch)
        return self._predict_and_losses_single_real(batch)

    def _predict_and_losses_seq_only_real(self, batch: Sequence[DownstreamStage1Sample]) -> Dict[str, Any]:
        torch = importlib.import_module("torch")
        with torch.no_grad():
            if self._is_pair_batch(batch):
                protein_sequences = [
                    sample.sequence[: self.config.protein_max_sequence_length]
                    for sample in batch
                ]
                peptide_sequences = [
                    self._require_peptide_sequence(sample)[: self.config.peptide_max_sequence_length]
                    for sample in batch
                ]
                protein_seq_output = self.sequence_encoder(protein_sequences)
                peptide_seq_output = self.sequence_encoder(peptide_sequences)
                protein_seq_embeddings = _ensure_tensor_matrix(
                    protein_seq_output.pooled_embedding,
                    name="protein_seq_embeddings",
                    torch_module=torch,
                    device=self.config.device,
                )
                peptide_seq_embeddings = _ensure_tensor_matrix(
                    peptide_seq_output.pooled_embedding,
                    name="peptide_seq_embeddings",
                    torch_module=torch,
                    device=self.config.device,
                )
                features = torch.cat([protein_seq_embeddings, peptide_seq_embeddings], dim=-1)
            else:
                seq_output = self.sequence_encoder([sample.sequence for sample in batch])
                features = _ensure_tensor_matrix(
                    seq_output.pooled_embedding,
                    name="seq_embeddings",
                    torch_module=torch,
                    device=self.config.device,
                )

            predictions = self.task_head(features)
            task_loss = self._compute_task_loss(predictions, [sample.target for sample in batch])
            zero = task_loss * 0.0
        return {
            "full_predictions": predictions,
            "missing_predictions": None,
            "task_loss": float(task_loss.detach().cpu().item()),
            "alignment_loss": float(zero.detach().cpu().item()),
            "consistency_loss": float(zero.detach().cpu().item()),
            "reconstruction_loss": float(zero.detach().cpu().item()),
            "total_loss": float(task_loss.detach().cpu().item()),
        }

    def _predict_and_losses_single_real(self, batch: Sequence[DownstreamStage1Sample]) -> Dict[str, Any]:
        torch = importlib.import_module("torch")
        with torch.no_grad():
            sequences = [sample.sequence for sample in batch]
            seq_output = self.sequence_encoder(sequences)
            seq_embeddings = _ensure_tensor_matrix(
                seq_output.pooled_embedding,
                name="seq_embeddings",
                torch_module=torch,
                device=self.config.device,
            )

            dyn_embeddings, task_structure_mask, aux_md_mask = self._encode_single_dyn_embeddings(
                batch,
                torch,
            )
            num_full_samples = sum(int(value) for value in aux_md_mask)

            full_view = self.fusion_transformer(
                seq_embeddings,
                dyn_embeddings,
                has_dyn=task_structure_mask,
            )
            missing_view = self.fusion_transformer(
                seq_embeddings,
                dyn_embeddings,
                has_dyn=[False] * len(batch),
            )
            fused_full = _ensure_tensor_matrix(
                full_view.fused_pooled,
                name="fused_full",
                torch_module=torch,
                device=self.config.device,
            )
            fused_missing = _ensure_tensor_matrix(
                missing_view.fused_pooled,
                name="fused_missing",
                torch_module=torch,
                device=self.config.device,
            )
            seq_task_features = self.seq_task_adapter(seq_embeddings)
            task_features = self._choose_single_task_features(
                seq_embeddings=seq_embeddings,
                seq_task_features=seq_task_features,
                fused_missing=fused_missing,
                fused_full=fused_full,
                full_mask=task_structure_mask,
                torch_module=torch,
            )
            full_predictions = self.task_head(task_features)
            fallback_mode = self.config.single_missing_task_fallback.strip().lower()
            feature_mode = self.config.single_task_feature_mode.strip().lower()
            if feature_mode == "raw_seq_residual":
                missing_features = seq_embeddings + self.single_task_residual_adapter(fused_missing - seq_task_features)
                if fallback_mode == "seq_adapter" and not self.config.force_missing_dynamics_for_all_samples:
                    missing_features = seq_embeddings
            else:
                missing_features = seq_task_features if fallback_mode == "seq_adapter" else fused_missing
            missing_predictions = self.task_head(missing_features)
            task_loss = self._compute_task_loss(full_predictions, [sample.target for sample in batch])

            if num_full_samples == 0:
                zero = task_loss * 0.0
                alignment = zero
                consistency = zero
                reconstruction = zero
                coverage_scale = 0.0
            else:
                projected_seq = self.seq_projection_head(seq_embeddings)
                projected_dyn = self.dyn_projection_head(dyn_embeddings)
                reconstructed_dyn = self.recon_head(fused_missing)
                alignment = info_nce_loss(projected_seq, projected_dyn, valid_mask=aux_md_mask)
                consistency = consistency_loss(
                    fused_missing,
                    fused_full,
                    mode=self.config.consistency_mode,
                    valid_mask=aux_md_mask,
                )
                reconstruction = reconstruction_loss(
                    reconstructed_dyn,
                    dyn_embeddings,
                    valid_mask=aux_md_mask,
                )
                coverage_scale = self._coverage_scale(num_full_samples, len(batch))

            total_loss = (
                task_loss
                + self.config.lambda_align * coverage_scale * alignment
                + self.config.lambda_cons * coverage_scale * consistency
                + self.config.lambda_recon * coverage_scale * reconstruction
            )
        return {
            "full_predictions": full_predictions,
            "missing_predictions": missing_predictions,
            "task_loss": float(task_loss.detach().cpu().item()),
            "alignment_loss": float(alignment.detach().cpu().item()),
            "consistency_loss": float(consistency.detach().cpu().item()),
            "reconstruction_loss": float(reconstruction.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }

    def _predict_and_losses_pair_real(self, batch: Sequence[DownstreamStage1Sample]) -> Dict[str, Any]:
        torch = importlib.import_module("torch")
        with torch.no_grad():
            protein_sequences = [
                sample.sequence[: self.config.protein_max_sequence_length]
                for sample in batch
            ]
            peptide_sequences = [
                self._require_peptide_sequence(sample)[: self.config.peptide_max_sequence_length]
                for sample in batch
            ]

            batch_size = len(batch)
            protein_seq_output = self.sequence_encoder(protein_sequences)
            protein_seq_embeddings = _ensure_tensor_matrix(
                protein_seq_output.pooled_embedding,
                name="protein_seq_embeddings",
                torch_module=torch,
                device=self.config.device,
            )
            peptide_seq_output = self.sequence_encoder(peptide_sequences)
            peptide_seq_embeddings = _ensure_tensor_matrix(
                peptide_seq_output.pooled_embedding,
                name="peptide_seq_embeddings",
                torch_module=torch,
                device=self.config.device,
            )

            protein_dyn_embeddings, protein_full_mask, protein_aux_md_mask = self._encode_side_dyn_embeddings(
                batch,
                torch,
                side="protein",
                max_residues=self.config.protein_max_residues,
            )
            peptide_dyn_embeddings, peptide_full_mask, peptide_aux_md_mask = self._encode_side_dyn_embeddings(
                batch,
                torch,
                side="peptide",
                max_residues=self.config.peptide_max_residues,
            )

            protein_full_view = self.fusion_transformer(
                protein_seq_embeddings,
                protein_dyn_embeddings,
                has_dyn=protein_full_mask,
            )
            protein_missing_view = self.fusion_transformer(
                protein_seq_embeddings,
                protein_dyn_embeddings,
                has_dyn=[False] * batch_size,
            )
            peptide_full_view = self.fusion_transformer(
                peptide_seq_embeddings,
                peptide_dyn_embeddings,
                has_dyn=peptide_full_mask,
            )
            peptide_missing_view = self.fusion_transformer(
                peptide_seq_embeddings,
                peptide_dyn_embeddings,
                has_dyn=[False] * batch_size,
            )

            protein_fused_full = _ensure_tensor_matrix(
                protein_full_view.fused_pooled,
                name="protein_fused_full",
                torch_module=torch,
                device=self.config.device,
            )
            protein_fused_missing = _ensure_tensor_matrix(
                protein_missing_view.fused_pooled,
                name="protein_fused_missing",
                torch_module=torch,
                device=self.config.device,
            )
            peptide_fused_full = _ensure_tensor_matrix(
                peptide_full_view.fused_pooled,
                name="peptide_fused_full",
                torch_module=torch,
                device=self.config.device,
            )
            peptide_fused_missing = _ensure_tensor_matrix(
                peptide_missing_view.fused_pooled,
                name="peptide_fused_missing",
                torch_module=torch,
                device=self.config.device,
            )

            pair_fused_missing = torch.cat([protein_fused_missing, peptide_fused_missing], dim=-1)
            task_features = self._choose_pair_task_features(
                protein_fused_missing=protein_fused_missing,
                peptide_fused_missing=peptide_fused_missing,
                protein_fused_full=protein_fused_full,
                peptide_fused_full=peptide_fused_full,
                protein_full_mask=protein_full_mask,
                peptide_full_mask=peptide_full_mask,
                torch_module=torch,
            )
            full_predictions = self.task_head(task_features)
            missing_predictions = self.task_head(pair_fused_missing)
            task_loss = self._compute_task_loss(full_predictions, [sample.target for sample in batch])

            side_full_mask = protein_aux_md_mask + peptide_aux_md_mask
            valid_side_count = sum(int(value) for value in side_full_mask)
            if not any(side_full_mask):
                zero = task_loss * 0.0
                alignment = zero
                consistency = zero
                reconstruction = zero
                coverage_scale = 0.0
            else:
                side_seq_embeddings = torch.cat([protein_seq_embeddings, peptide_seq_embeddings], dim=0)
                side_dyn_embeddings = torch.cat([protein_dyn_embeddings, peptide_dyn_embeddings], dim=0)
                side_fused_full = torch.cat([protein_fused_full, peptide_fused_full], dim=0)
                side_fused_missing = torch.cat([protein_fused_missing, peptide_fused_missing], dim=0)
                projected_seq = self.seq_projection_head(side_seq_embeddings)
                projected_dyn = self.dyn_projection_head(side_dyn_embeddings)
                reconstructed_dyn = self.recon_head(side_fused_missing)
                alignment = info_nce_loss(projected_seq, projected_dyn, valid_mask=side_full_mask)
                consistency = consistency_loss(
                    side_fused_missing,
                    side_fused_full,
                    mode=self.config.consistency_mode,
                    valid_mask=side_full_mask,
                )
                reconstruction = reconstruction_loss(
                    reconstructed_dyn,
                    side_dyn_embeddings,
                    valid_mask=side_full_mask,
                )
                coverage_scale = self._coverage_scale(valid_side_count, len(side_full_mask))

            total_loss = (
                task_loss
                + self.config.lambda_align * coverage_scale * alignment
                + self.config.lambda_cons * coverage_scale * consistency
                + self.config.lambda_recon * coverage_scale * reconstruction
            )
        return {
            "full_predictions": full_predictions,
            "missing_predictions": missing_predictions,
            "task_loss": float(task_loss.detach().cpu().item()),
            "alignment_loss": float(alignment.detach().cpu().item()),
            "consistency_loss": float(consistency.detach().cpu().item()),
            "reconstruction_loss": float(reconstruction.detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
        }


def build_downstream_trainer(
    config: DownstreamTrainerConfig,
) -> DownstreamTrainer:
    return DownstreamTrainer(config)


def _ensure_real_runtime_dependencies() -> None:
    missing_dependencies = []
    if importlib.util.find_spec("torch") is None:
        missing_dependencies.append("torch")
    if missing_dependencies:
        raise RuntimeError(
            "Real downstream trainer requires additional runtime dependencies: "
            f"{', '.join(missing_dependencies)}"
        )


def _infer_multilabel_dim(manifest_path: str, task_name: str) -> int:
    path = Path(manifest_path)
    if path.is_file():
        dataset = DownstreamDataset.from_manifest(path, task_name=task_name, sample_limit=1)
        first_target = dataset[0].target
        if not isinstance(first_target, list):
            raise ValueError("Expected multilabel target to be a list of floats.")
        return len(first_target)
    if path.is_dir():
        samples = load_downstream_samples(task_name, path, sample_limit=1)
        if not samples:
            raise ValueError(f"No samples were loaded from {path} for task {task_name!r}.")
        first_target = DownstreamDataset.from_samples_with_pretrain_index.__globals__["parse_target"](
            samples[0].label,
            task_kind=task_kind_for_name(task_name),
        )
        if not isinstance(first_target, list):
            raise ValueError("Expected multilabel target to be a list of floats.")
        return len(first_target)
    raise FileNotFoundError(f"Downstream source does not exist: {path}")


def _iter_sample_batches(
    samples: Sequence[DownstreamStage1Sample],
    batch_size: int,
) -> Iterator[Sequence[DownstreamStage1Sample]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    for start_index in range(0, len(samples), batch_size):
        yield samples[start_index : start_index + batch_size]


def _num_batches(total_samples: int, batch_size: int) -> int:
    if total_samples <= 0:
        return 0
    return (total_samples + batch_size - 1) // batch_size


def _maybe_progress_bar(
    batches: Sequence[Sequence[DownstreamStage1Sample]],
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
    step_result: DownstreamStepResult,
    *,
    task_name: str,
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
            task=f"{step_result.task_loss:.4f}",
            align=f"{step_result.alignment_loss:.4f}",
            cons=f"{step_result.consistency_loss:.4f}",
            recon=f"{step_result.reconstruction_loss:.4f}",
        )
        return
    if batch_index % log_interval != 0 and batch_index != num_batches:
        return
    print(
        f"downstream task={task_name} "
        f"epoch={epoch}/{max_epochs} "
        f"batch={batch_index}/{num_batches} "
        f"loss={step_result.total_loss:.6f} "
        f"task={step_result.task_loss:.6f} "
        f"align={step_result.alignment_loss:.6f} "
        f"cons={step_result.consistency_loss:.6f} "
        f"recon={step_result.reconstruction_loss:.6f}",
        flush=True,
    )


def _print_epoch_report(*, prefix: str, epoch: int, report: Dict[str, Any]) -> None:
    split_name = prefix.removeprefix("epoch_")
    print(
        f"{prefix}.epoch={epoch} task={report['task_name']} kind={report['task_kind']}",
        flush=True,
    )
    rows = _epoch_report_rows(report)
    for line in _render_epoch_report_table(
        epoch=epoch,
        split_name=split_name,
        task_kind=report["task_kind"],
        rows=rows,
    ):
        print(line, flush=True)


def _ordered_report_subsets(report: Dict[str, Any]) -> list[str]:
    return [
        subset_name
        for subset_name in ("overall", "seq_only", "nature_only", "partial", "full")
        if subset_name in report
    ]


def _epoch_report_rows(report: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for subset_name in _ordered_report_subsets(report):
        subset = report[subset_name]
        row = {
            "subset": subset_name,
            "num_samples": subset.get("num_samples", 0),
        }
        row.update(subset.get("metrics") or {})
        rows.append(row)
    return rows


def _render_epoch_report_table(
    *,
    epoch: int,
    split_name: str,
    task_kind: str,
    rows: Sequence[Dict[str, Any]],
) -> list[str]:
    metric_columns = _metric_columns_for_task_kind(task_kind)
    columns = ["epoch", "split", "subset", "num_samples", *metric_columns]
    table_rows: list[Dict[str, Any]] = []
    for row in rows:
        table_rows.append(
            {
                "epoch": epoch,
                "split": split_name,
                "subset": row.get("subset", ""),
                "num_samples": row.get("num_samples", 0),
                **{column: row.get(column, "") for column in metric_columns},
            }
        )
    return _format_table(columns, table_rows)


def _metric_columns_for_task_kind(task_kind: str) -> list[str]:
    if task_kind == "binary":
        return ["accuracy", "precision", "recall", "f1_score", "aupr", "roc_auc"]
    if task_kind == "multilabel":
        return ["Precision", "Coverage", "Accuracy", "Abs True", "Abs False"]
    if task_kind == "regression":
        return ["MSE", "MAE", "RMSE", "R2", "Pearson", "Spearman"]
    return []


def _format_table(columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    rendered_rows = [
        {column: _format_cell_value(row.get(column, "")) for column in columns}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(rendered_row[column]) for rendered_row in rendered_rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    lines = [header, separator]
    for rendered_row in rendered_rows:
        lines.append(
            " | ".join(rendered_row[column].ljust(widths[column]) for column in columns)
        )
    return lines


def _format_cell_value(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _evaluation_modality_subset(
    sample: DownstreamStage1Sample,
    *,
    force_missing_dynamics_for_all_samples: bool = False,
    force_natural_structure_only_for_all_samples: bool = False,
) -> str:
    if force_natural_structure_only_for_all_samples:
        if sample.is_pair_sample:
            return "nature_only" if (sample.nature_path or sample.peptide_nature_path) else "seq_only"
        return "nature_only" if sample.nature_path else "seq_only"
    if force_missing_dynamics_for_all_samples:
        return "seq_only"
    if sample.is_pair_sample:
        if sample.has_dyn and sample.peptide_has_dyn:
            return "full"
        if sample.has_dyn or sample.peptide_has_dyn:
            return "partial"
        return "seq_only"
    return "full" if sample.has_dyn else "seq_only"


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


def torch_where_rows(mask: list[bool], true_rows: Any, false_rows: Any) -> Any:
    if len(mask) == 0:
        return true_rows
    torch_module = importlib.import_module("torch")
    mask_tensor = torch_module.tensor(mask, dtype=torch_module.bool, device=true_rows.device).unsqueeze(-1)
    return torch_module.where(mask_tensor, true_rows, false_rows)


def _parameter_requires_grad(parameter: Any) -> bool:
    return bool(getattr(parameter, "requires_grad", True))


def _predictions_for_metrics(
    predictions: Any,
    *,
    task_kind: str,
) -> list[Any]:
    if _uses_torch_tensor(predictions):
        torch = importlib.import_module("torch")
        tensor = predictions.detach().to(dtype=torch.float32, device="cpu")
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if task_kind == "binary":
            return torch.sigmoid(tensor.squeeze(-1)).tolist()
        if task_kind == "multilabel":
            return torch.sigmoid(tensor).tolist()
        if task_kind == "regression":
            return tensor.squeeze(-1).tolist()
        raise ValueError(f"Unsupported task_kind: {task_kind}")

    matrix = _ensure_matrix(predictions, name="predictions")
    if task_kind == "binary":
        return [_sigmoid(row[0]) for row in matrix]
    if task_kind == "multilabel":
        return [[_sigmoid(value) for value in row] for row in matrix]
    if task_kind == "regression":
        return [row[0] for row in matrix]
    raise ValueError(f"Unsupported task_kind: {task_kind}")


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


def _deterministic_vector(key: str, dim: int) -> List[float]:
    values: List[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{key}:{counter}".encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            chunk = digest[offset : offset + 4]
            if len(chunk) < 4:
                continue
            integer_value = int.from_bytes(chunk, byteorder="big", signed=False)
            values.append((integer_value / float(2**32 - 1)) * 2.0 - 1.0)
            if len(values) >= dim:
                break
        counter += 1
    return values


def _normalize_task_name(task_name: str | None) -> str:
    if task_name is None:
        raise ValueError("task_name is required.")
    normalized = str(task_name).strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("task_name cannot be empty.")
    return normalized


def _uses_torch_tensor(value: object) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    torch = importlib.import_module("torch")
    return isinstance(value, torch.Tensor)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))
