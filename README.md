# ProtMMLM: Structure- and Dynamics-aware Multimodal Pre-training for Protein Prediction

ProtMMLM is a multimodal protein learning framework for protein representation learning and downstream prediction. It jointly uses protein sequence information, native-structure information, and molecular-dynamics-derived inputs to learn structure- and dynamics-aware representations.

## 1. Environment Setup

Create a Python environment and install the required dependencies，Please ensure you use a newer Python version (e.g., Python 3.12 or above) to properly obtain sequence embeddings:

Additional runtime dependencies may be required for specific workflows:

- ESMC runtime and weights for sequence encoding;
- MMseqs2 for sequence-clustered splits and Conotoxin negative-set filtering;
- CUDA-enabled PyTorch for GPU training and inference;
- external pretraining checkpoints for downstream initialization and probing.

Run all commands from the repository root:

```bash
export PYTHONPATH=.
```

You can view the full list of command-line arguments for any script with the `--help` flag. For example:

```bash
PYTHONPATH=. python scripts/run_pretrain.py --help
```

## 2. Repository Layout

```text
.
├── configs/
│   ├── pretrain/
│   │   └── pretrain.yaml
│   └── downstream/
│       ├── toxteller.yaml
│       ├── prmftp.yaml
│       ├── ppikb.yaml
│       └── conotoxin.yaml
├── data/
│   └── examples/
│       ├── pretrain_manifest.example.csv
│       └── downstream_manifest.example.csv
├── scripts/
│   ├── prepare_pretrain_manifest.py
│   ├── prepare_downstream_manifests.py
│   ├── prepare_structure_token_cache.py
│   ├── filter_conotoxin_negative_set.py
│   ├── run_pretrain.py
│   ├── run_downstream_finetuning.py
│   ├── probe_global_structural_properties.py
│   ├── probe_total_energy.py
│   ├── probe_energy_terms.py
│   └── analyze_rmsd_embedding_consistency.py
├── src/
│   ├── analysis/
│   ├── datasets/
│   ├── evaluation/
│   ├── losses/
│   ├── models/
│   ├── training/
│   └── utils/
├── requirements.txt
└── README.md
```

## 3. Data and Checkpoints

Only lightweight example manifests are included in this repository:

```text
data/examples/pretrain_manifest.example.csv
data/examples/downstream_manifest.example.csv
```

## 4. Example Workflow

**First**, prepare a pretraining manifest that links sequence, native-structure, and MD-derived inputs:

```bash
PYTHONPATH=. python scripts/prepare_pretrain_manifest.py \
  --input-fasta data/pretrain/all_sequences.fasta \
  --nature-dir data/pretrain/nature \
  --md-dir data/pretrain/MD \
  --output data/pretrain/pretrain_manifest.csv
```

**Second**, run multimodal pretraining with the formal pretraining configuration:

```bash
PYTHONPATH=. python scripts/run_pretrain.py \
  --config configs/pretrain/pretrain.yaml
```

**Third**, prepare downstream manifests. The following example prepares the ToxTeller task:

```bash
PYTHONPATH=. python scripts/prepare_downstream_manifests.py \
  --task toxteller \
  --input-path data/downstream/ToxTeller \
  --pretrain-manifest data/pretrain/pretrain_manifest.csv \
  --output-dir data/downstream/processed/toxteller
```

Supported downstream tasks are:

```text
toxteller
prmftp
ppikb
conotoxin
```

For PPIKB, the default raw-data root is `data/downstream/processed/regression`, with `run_id` selecting a run-specific split directory such as `run_1`. For Conotoxin, the negative-set filtering helper writes by default to `data/downstream/conotoxin/filtered_id70`, matching `configs/downstream/conotoxin.yaml`.

**Fourth**, fine-tune a downstream model:

```bash
PYTHONPATH=. python scripts/run_downstream_finetuning.py \
  --config configs/downstream/toxteller.yaml
```

## 5. Train

### 5.1 Multimodal pretraining

The pretraining workflow is implemented in `scripts/run_pretrain.py` and controlled by `configs/pretrain/pretrain.yaml`.

```bash
PYTHONPATH=. python scripts/run_pretrain.py \
  --config configs/pretrain/pretrain.yaml
```

### 5.2 Downstream fine-tuning

Downstream behavior is controlled by the YAML files in `configs/downstream/`.

For full fine-tuning:

```yaml
sequence_encoder_trainable: true
structure_encoder_trainable: true
fusion_transformer_trainable: true
train_only_task_head: false
```

For task-head-only MLP training:

```yaml
train_only_task_head: true
task_head_type: mlp
task_head_hidden_dims: [512, 128]
task_head_dropout: 0.1
```

Example command:

```bash
PYTHONPATH=. python scripts/run_downstream_finetuning.py \
  --config configs/downstream/toxteller.yaml
```

## 6. Analysis

### 6.1 Global structural-property probing

Probe Rg, Re, and asphericity from learned embeddings:

```bash
PYTHONPATH=. python scripts/probe_global_structural_properties.py \
  --config configs/pretrain/pretrain.yaml \
  --checkpoint path/to/pretrain_checkpoint.pth \
  --output-dir results/global_structural_properties
```

### 6.2 Total-energy probing

```bash
PYTHONPATH=. python scripts/probe_total_energy.py \
  --config configs/pretrain/pretrain.yaml \
  --checkpoint path/to/pretrain_checkpoint.pth \
  --energy-csv path/to/frame_energy.csv \
  --output-dir results/total_energy_probe
```

### 6.3 Energy-term probing

```bash
PYTHONPATH=. python scripts/probe_energy_terms.py \
  --config configs/pretrain/pretrain.yaml \
  --checkpoint path/to/pretrain_checkpoint.pth \
  --energy-csv path/to/frame_energy_terms.csv \
  --output-dir results/energy_terms_probe
```

### 6.4 RMSD-embedding consistency analysis

Analyze whether learned embeddings preserve trajectory-level structural geometry:

```bash
PYTHONPATH=. python scripts/analyze_rmsd_embedding_consistency.py \
  --config configs/pretrain/pretrain.yaml \
  --checkpoint path/to/pretrain_checkpoint.pth \
  --output-dir results/rmsd_embedding_consistency
```
