# Specific Task 2h: Linear-Attention Vision Transformer for End-to-End Mass Regression and Classification

This repository contains a complete workflow for:

1. **Self-supervised pretraining** on `Dataset_Specific_Unlabelled.h5` using a linear-attention ViT masked autoencoder (MAE), and
2. **Supervised finetuning** on `Dataset_Specific_labelled_full_only_for_2i.h5` for:
   - Binary classification (`Y`)
   - Mass regression (`m`)
   - pT regression (`pT`)

The core requirement (80/20 split with careful validation usage) is implemented in both notebook and script workflows.

---

## Dataset Usage

- **Pretraining dataset**: `data/Dataset_Specific_Unlabelled.h5`
- **Finetuning dataset**: `data/Dataset_Specific_labelled_full_only_for_2i.h5`

The labelled split protocol used for finetuning is:

- **20% test** (held out for final evaluation)
- **80% development**
  - inside this 80%: **90% train / 10% validation**

So overall proportions are:

- **72% train**
- **8% validation**
- **20% test**

This is explicitly implemented via `make_loaders(...)` and described in the notebook.

---

## Project Flow

### 1) Notebook workflow (`model.ipynb`)

The notebook provides a full, readable solution:

- data inspection and sanity checks,
- linear-attention ViT + MAE pretraining blocks,
- checkpoint loading from `checkpoints/<folder>/hyperparams.json` + `pretrained_encoder_best.pt`,
- finetuning experiments for classification/mass/pT,
- metrics and plots:
  - classification: val loss/accuracy, confusion matrix, ROC-AUC,
  - regression: val MSE and test MSE,
- reloading saved finetuned weights from `weights/` and re-evaluating on test split.

### 2) Pretraining sweep (`pretrain.py`)

`pretrain.py` runs many MAE pretraining variants (seed fixed to 67), changing model/hyperparameters such as:

- `mask_ratio`, `batch_size`, `patch`,
- `dim`, `depth`, `heads`,
- `lr`, `weight_decay`.

Outputs are saved under `checkpoints/<run_name>/`, including:

- `hyperparams.json`
- `train.log`
- `epoch_metrics.csv`
- `loss_curve.png`
- `mae_last.pt`, `mae_best.pt`, `pretrained_encoder_best.pt`
- `done.json`

Global pretraining summaries are available in:

- `checkpoints/summary.csv`
- `checkpoints/summary.json`

### 3) Finetuning all checkpoints (`finetune_all_checkpoints.py`)

This script discovers every valid checkpoint folder and runs **pretrained vs scratch** finetuning for all three tasks.

Outputs are saved under `results/<checkpoint_name>/`:

- per-task histories: `*_pretrained_history.csv`, `*_scratch_history.csv`
- per-task val curves: `*_val_curves.png`
- classification diagnostics:
  - `classification_confusion_matrices.png`
  - `classification_roc.png`
- task metrics: `*_metrics.json`
- finetuned weights:
  - `*_pretrained_finetune.pt`
  - `*_scratch_finetune.pt`
- checkpoint-level summary: `summary.json`

Global finetuning summaries are available in:

- `results/all_checkpoints_summary.csv`
- `results/all_checkpoints_summary.json`

---

## Key Results (from `results/all_checkpoints_summary.csv`)

Below are best observed test metrics across the checkpoint sweep.

### Classification (`Y`)

- **Best pretrained accuracy**: `0.8905`  
  (`linear_wide_mask60_bs16_p5_dim320_d4_h10_lr8e-05_wd0.0001_ep100_seed67`)
- **Best pretrained AUC**: `0.9493`  
  (`linear_deep_mask60_bs24_p5_dim256_d6_h8_lr8e-05_wd0.0001_ep100_seed67`)
- **Best scratch accuracy**: `0.8275`  
  (`linear_patch25_mask60_bs64_p25_dim256_d6_h8_lr0.0001_wd0.0001_ep100_seed67`)
- **Best scratch AUC**: `0.9006`  
  (`linear_patch25_mask60_bs64_p25_dim256_d6_h8_lr0.0001_wd0.0001_ep100_seed67`)

### Mass Regression (`m`)

- **Best pretrained MSE (normalized)**: `0.3243`
- **Best pretrained MSE (denormalized)**: `853.24`
- **Best scratch MSE (normalized)**: `0.4449`
- **Best scratch MSE (denormalized)**: `1170.66`

(denormalized values are computed as normalized MSE × `y_std^2` for the target)

### pT Regression (`pT`)

- **Best pretrained MSE (normalized)**: `0.7928`
- **Best pretrained MSE (denormalized)**: `9290.32`
- **Best scratch MSE (normalized)**: `1.0199`
- **Best scratch MSE (denormalized)**: `11952.13`

Overall trend: **pretrained consistently outperforms scratch** across classification, mass regression, and pT regression.

---

## How to Run

### A) Pretraining sweep

```bash
python pretrain.py
```

Useful options:

```bash
python pretrain.py --max-runs 3
python pretrain.py --overwrite
python pretrain.py --num-workers 4
```

### B) Finetune all checkpoints and collect results

```bash
python finetune_all_checkpoints.py
```

Useful options:

```bash
python finetune_all_checkpoints.py --checkpoint-filter linear_base
python finetune_all_checkpoints.py --patience 1000000
python finetune_all_checkpoints.py --batch-size 32 --num-workers 4
```

### C) Interactive notebook exploration

Open and run:

- `model.ipynb`

The notebook includes checkpoint selection by folder name and final weight reloading/evaluation cells.

---

## Files of Interest

- `model.ipynb` — end-to-end notebook solution with diagnostics and final evaluation.
- `pretrain.py` — MAE pretraining sweep generator.
- `finetune_all_checkpoints.py` — automated finetuning benchmark across all pretrained checkpoints.
- `checkpoints/` — pretrained artifacts and sweep summaries.
- `results/` — finetuning artifacts and benchmark summaries.
- `weights/` — finetuned model weights saved by notebook runs.

---

## Notes on Overfitting Control

- Test set is held out from training.
- Validation split is taken only from the training/development partition.
- Finetuning code supports early stopping (`patience`) and uses validation metrics for model selection.
- Scripts default to deterministic seed handling (`seed=67`) for reproducibility.
