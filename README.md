# ProtMMLM: Structure- and Dynamics-aware Multimodal Pre-training for Protein Prediction

ProtMMLM is a multimodal protein learning framework for protein representation learning and downstream prediction. It jointly models protein sequence information, native structural information, and molecular-dynamics-derived inputs to learn structure- and dynamics-aware representations.

This repository provides a reproducible workflow for preparing manifests, pretraining ProtMMLM, fine-tuning downstream tasks from pretrained weights, and running probing analyses.

## 1. Environment Setup

Create and activate a Python environment:

```bash
conda create -n protmmlm python=3.10 -y
conda activate protmmlm
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -U huggingface_hub
conda install -c bioconda -c conda-forge mmseqs2 -y
```

Use a CUDA-enabled PyTorch build for GPU training and inference. All scripts expose command-line help, for example:

```bash
PYTHONPATH=. python scripts/run_pretrain.py --help
```

## 2. Repository Layout

Keep the repository organized as follows:

```text
├── checkpoints/
│   ├── esmc-600m/
│   └── pretrain_checkpoint.pth
├── configs/
│   ├── pretrain/pretrain.yaml
│   └── downstream/
│       ├── toxteller.yaml
│       ├── prmftp.yaml
│       ├── ppikb.yaml
│       └── conotoxin.yaml
├── datasets/
│   ├── pretrain/
│   └── downstream/
├── esm/                         # Vendored ESM runtime source; model weights are excluded.
├── examples/                    # One-fold downstream examples for quick verification.
├── scripts/                     # Data preparation, training, fine-tuning, and analysis entry points.
├── src/                         # Datasets, models, losses, trainers, and evaluators.
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

Large datasets, model weights, generated caches, and training outputs are intentionally excluded from git.

## 3. Data and Checkpoints

### 3.1 Download datasets

Dataset repository: https://huggingface.co/datasets/AISciLab/ProtMMLM-datasets

Download the released dataset into `datasets/`:

```bash
huggingface-cli download AISciLab/ProtMMLM-datasets \
  --repo-type dataset \
  --local-dir datasets
```

Expected layout after download:

```text
datasets/pretrain/
├── all_sequences.fasta
├── nature.tar.gz
└── md.tar.gz

datasets/downstream/
├── Toxteller/
├── PrMFTP/
├── PPIKB/
├── Conotoxin/
├── Thermodynamic/
└── all_interactions_all.csv
```

Extract the pretraining structure archives before preparing the pretraining manifest:

```bash
tar -xf datasets/pretrain/nature.tar.gz -C datasets/pretrain
tar -xf datasets/pretrain/md.tar.gz -C datasets/pretrain
```

After extraction:

```text
datasets/pretrain/
├── all_sequences.fasta
├── nature/
└── md/
```

If the downloaded dataset contains an extra top-level directory, move or symlink its `pretrain/` and `downstream/` folders into `datasets/pretrain/` and `datasets/downstream/`.

### 3.2 Download and place checkpoints

The ESM runtime source is vendored under `esm/`, but ESMC-600M weights are not committed. Place the ESMC-600M checkpoint at:

```text
checkpoints/esmc-600m/data/weights/esmc_600m_2024_12_v0.pth
```

The ProtMMLM pretrained checkpoint can be downloaded from [this link](https://drive.google.com/file/d/1o-upySIGIF1kjTYXOeOCqjLTq5o_GxPP/view?usp=sharing). Place it at:

```text
checkpoints/pretrain_checkpoint.pth
```

For downstream fine-tuning and analysis, ProtMMLM can initialize from any compatible pretrained checkpoint. This may be:

- `outputs/pretrain/best.pth` from your own pretraining run.
- `checkpoints/pretrain_checkpoint.pth` downloaded from the link above.
- Another compatible checkpoint retrained by others.

The default downstream configs use `outputs/pretrain/best.pth`. You can override it at runtime:

```bash
PYTHONPATH=. python scripts/run_downstream_finetuning.py \
  --config configs/downstream/ppikb.yaml \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth
```

## 4. Quick Start

If you only want to verify the code with the released pretrained checkpoint, use the prepared example folds in `examples/`:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task ppikb \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth
```

Run all example folds:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task all \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth
```

For a short smoke test:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task ppikb \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth \
  --max-epochs 1 \
  --batch-size 2 \
  --sample-limit 32
```

## 5. Pretraining

Prepare the pretraining manifest:

```bash
PYTHONPATH=. python scripts/prepare_pretrain_manifest.py \
  --input-fasta datasets/pretrain/all_sequences.fasta \
  --nature-dir datasets/pretrain/nature \
  --md-dir datasets/pretrain/md \
  --output datasets/pretrain/pretrain_manifest.csv
```

Run multimodal pretraining:

```bash
PYTHONPATH=. python scripts/run_pretrain.py \
  --config configs/pretrain/pretrain.yaml
```

Default outputs:

```text
outputs/pretrain/latest.pth
outputs/pretrain/best.pth
outputs/tensorboard/pretrain/
```

Resume from the latest checkpoint:

```bash
PYTHONPATH=. python scripts/run_pretrain.py \
  --config configs/pretrain/pretrain.yaml \
  --checkpoint outputs/pretrain/latest.pth
```

## 6. Downstream Fine-tuning

Downstream behavior is controlled by YAML files in `configs/downstream/`. The pretrained checkpoint can be set in YAML with `pretrain_checkpoint_path` or at runtime with `--pretrain-checkpoint`.

Prepare downstream manifests from raw downloaded data. For example, Toxteller:

```bash
PYTHONPATH=. python scripts/prepare_downstream_manifests.py \
  --task toxteller \
  --input-path datasets/downstream/Toxteller \
  --pretrain-manifest datasets/pretrain/pretrain_manifest.csv \
  --output-dir datasets/downstream/processed/toxteller
```

Supported tasks:

```text
toxteller
prmftp
ppikb
conotoxin
```

Run fine-tuning:

```bash
PYTHONPATH=. python scripts/run_downstream_finetuning.py \
  --config configs/downstream/toxteller.yaml \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth

PYTHONPATH=. python scripts/run_downstream_finetuning.py \
  --config configs/downstream/prmftp.yaml \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth

PYTHONPATH=. python scripts/run_downstream_finetuning.py \
  --config configs/downstream/ppikb.yaml \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth

PYTHONPATH=. python scripts/run_downstream_finetuning.py \
  --config configs/downstream/conotoxin.yaml \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth
```

## 7. Direct Verification with Example Folds

The `examples/` directory contains one prepared fold for each downstream task:

```text
examples/toxteller/{train.csv,validation.csv,test.csv,toxteller_manifest.csv}
examples/prmftp/{train.csv,validation.csv,test.csv,prmftp_manifest.csv}
examples/ppikb/{train.csv,validation.csv,test.csv,ppikb_manifest.csv}
examples/conotoxin/{train.csv,validation.csv,test.csv,conotoxin_manifest.csv}
```

Use these folds when you already have pretrained ProtMMLM weights and want to skip raw-data preprocessing.

Run one downstream example fold:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task ppikb \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth
```

Run all example folds:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task all \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth
```

Short training verification:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task ppikb \
  --pretrain-checkpoint checkpoints/pretrain_checkpoint.pth \
  --max-epochs 1 \
  --batch-size 2 \
  --sample-limit 32
```

Example outputs are written under:

```text
outputs/downstream/<task>/examples_<task>_from_pretrained/
```

See [examples/README.md](examples/README.md) for more details.

## 8. Optional Sequence-clustered Splits

For publication-quality evaluation, generate explicit MMseqs2-based split files:

```bash
PYTHONPATH=. python scripts/prepare_mmseqs_cluster_splits.py \
  --input-csv datasets/downstream/processed/ppikb/ppikb_manifest.csv \
  --id-column sequence_hash \
  --sequence-column sequence \
  --label-column label \
  --output-dir datasets/downstream/processed/ppikb_mmseqs40_splits \
  --min-seq-id 0.4 \
  --coverage 0.8 \
  --cov-mode 0 \
  --train-fraction 0.8 \
  --validation-fraction 0.1 \
  --seed 42
```

The script writes FASTA files, MMseqs cluster assignments, split assignments, and a split summary JSON.

## 9. Analysis

The analysis scripts probe frozen ProtMMLM frame embeddings. They accept dataset paths and checkpoint paths from the command line, so the same commands work with either a checkpoint generated by this repository or a compatible external checkpoint.

Common arguments:

```text
--config configs/pretrain/pretrain.yaml
--manifest-path datasets/pretrain/pretrain_manifest.csv
--nature-dir datasets/pretrain/nature
--md-dir datasets/pretrain/md
--checkpoint checkpoints/pretrain_checkpoint.pth
--device cuda
--max-residues 100
--max-frames 160
--embedding-cache results/cache.npz
--extract-only
--reuse-embeddings
--pca-components 100,300
--probe mlp
--probe svm
```

### 9.1 Global Structural-property Probing

Probe radius of gyration, end-to-end distance, and asphericity from frozen ProtMMLM frame embeddings:

```bash
PYTHONPATH=. python scripts/analyze_frame_global_property_prediction.py \
  --config configs/pretrain/pretrain.yaml \
  --manifest-path datasets/pretrain/pretrain_manifest.csv \
  --nature-dir datasets/pretrain/nature \
  --md-dir datasets/pretrain/md \
  --checkpoint checkpoints/pretrain_checkpoint.pth \
  --output-dir results/frame_global_property_prediction/mlp/pca \
  --probe mlp \
  --pca-components 100,300 \
  --device cuda
```

### 9.2 Total-energy Probing

Extract a filtered total-score embedding cache:

```bash
PYTHONPATH=. python scripts/analyze_frame_global_energy_prediction.py \
  --config configs/pretrain/pretrain.yaml \
  --manifest-path datasets/pretrain/pretrain_manifest.csv \
  --nature-dir datasets/pretrain/nature \
  --md-dir datasets/pretrain/md \
  --checkpoint checkpoints/pretrain_checkpoint.pth \
  --energy-csv datasets/downstream/all_interactions_all.csv \
  --output-dir results/frame_global_energy_embeddings_cache_q01_q99 \
  --embedding-cache results/frame_global_energy_embeddings_cache_q01_q99/all_frame_fusion_energy_total_score_embeddings.npz \
  --target-quantile-low 0.01 \
  --target-quantile-high 0.99 \
  --extract-only \
  --device cuda
```

Reuse that cache for probe training:

```bash
PYTHONPATH=. python scripts/analyze_frame_global_energy_prediction.py \
  --config configs/pretrain/pretrain.yaml \
  --manifest-path datasets/pretrain/pretrain_manifest.csv \
  --checkpoint checkpoints/pretrain_checkpoint.pth \
  --energy-csv datasets/downstream/all_interactions_all.csv \
  --output-dir results/frame_global_energy_prediction/seed51/mlp/pca-500 \
  --embedding-cache results/frame_global_energy_embeddings_cache_q01_q99/all_frame_fusion_energy_total_score_embeddings.npz \
  --reuse-embeddings \
  --probe mlp \
  --pca-components 500 \
  --device cuda
```

### 9.3 Energy-term Probing

Extract the 9-term energy embedding cache:

```bash
PYTHONPATH=. python scripts/analyze_frame_global_energy_terms_prediction.py \
  --config configs/pretrain/pretrain.yaml \
  --manifest-path datasets/pretrain/pretrain_manifest.csv \
  --nature-dir datasets/pretrain/nature \
  --md-dir datasets/pretrain/md \
  --checkpoint checkpoints/pretrain_checkpoint.pth \
  --energy-csv datasets/downstream/all_interactions_all.csv \
  --output-dir results/frame_global_energy_terms_prediction_extract \
  --embedding-cache results/frame_global_energy_terms_prediction_extract/energy_terms_embeddings.npz \
  --extract-only \
  --device cuda
```

Reuse that cache for probe training:

```bash
PYTHONPATH=. python scripts/analyze_frame_global_energy_terms_prediction.py \
  --config configs/pretrain/pretrain.yaml \
  --manifest-path datasets/pretrain/pretrain_manifest.csv \
  --checkpoint checkpoints/pretrain_checkpoint.pth \
  --energy-csv datasets/downstream/all_interactions_all.csv \
  --output-dir results/frame_global_energy_terms_prediction/seed51/mlp/pca-500 \
  --embedding-cache results/frame_global_energy_terms_prediction_extract/energy_terms_embeddings.npz \
  --reuse-embeddings \
  --probe mlp \
  --pca-components 500 \
  --device cuda
```
