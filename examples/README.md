# Downstream Example Runs

This directory contains one prepared fold for each ProtMMLM downstream task:

```text
examples/toxteller/
examples/prmftp/
examples/ppikb/
examples/conotoxin/
```

Each task directory includes `train.csv`, `validation.csv`, `test.csv`, and a task manifest. These examples are intended for a quick training and evaluation check when you already have the ESMC backbone weights and one compatible ProtMMLM pretrained checkpoint. The ProtMMLM checkpoint may be produced by your own pretraining run, downloaded as a released checkpoint, or retrained by another group.

```text
checkpoints/esmc-600m/data/weights/esmc_600m_2024_12_v0.pth
outputs/pretrain/best.pth  # default example path
```

If your checkpoint is elsewhere, pass it with `--pretrain-checkpoint`.

## Run One Downstream Task from Pretrained ProtMMLM

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task ppikb \
  --pretrain-checkpoint /path/to/custom_protmmlm_pretrained.pth
```

## Run All Four Example Folds

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task all \
  --pretrain-checkpoint /path/to/custom_protmmlm_pretrained.pth
```

## Short Verification Run

For a fast sanity check, reduce epochs and optionally sample count:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task ppikb \
  --pretrain-checkpoint /path/to/custom_protmmlm_pretrained.pth \
  --max-epochs 1 \
  --batch-size 2 \
  --sample-limit 32
```

Use `--device cpu` only for dependency and data-flow checks. Full downstream verification with the ESMC backbone is expected to run on GPU:

```bash
PYTHONPATH=. python examples/run_downstream_example.py \
  --task toxteller \
  --pretrain-checkpoint /path/to/custom_protmmlm_pretrained.pth \
  --max-epochs 1 \
  --device cuda
```

## Outputs

Example runs write under:

```text
outputs/downstream/<task>/examples_<task>_from_pretrained/
```

Important files include:

```text
final_metrics.json
test_results.json
splits/train.csv
splits/validation.csv
splits/test.csv
```

If you want a fresh run from the pretrained checkpoint, remove or rename the previous example output folder for that task before rerunning.
