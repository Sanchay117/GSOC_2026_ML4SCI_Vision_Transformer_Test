import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=8, embed_dim=192, patch=5):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)
        bsz, dim, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(bsz, height * width, dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=6, eps=1e-6):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.head_dim = dim // heads
        self.eps = eps

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        y = self.norm(x)
        bsz, n_tokens, dim = y.shape

        qkv = self.to_qkv(y).reshape(bsz, n_tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        kv = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        k_sum = k.sum(dim=2)

        z = 1.0 / (torch.einsum("b h n d, b h d -> b h n", q, k_sum) + self.eps)
        out = torch.einsum("b h n d, b h d e, b h n -> b h n e", q, kv, z)

        out = out.transpose(1, 2).contiguous().reshape(bsz, n_tokens, dim)
        out = self.proj(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, heads=6):
        super().__init__()
        self.attn = Attention(dim, heads=heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(self.n2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, dim=192, depth=4, heads=6, patch=5, in_ch=8):
        super().__init__()
        self.patch = PatchEmbed(in_ch=in_ch, embed_dim=dim, patch=patch)
        grid = 125 // patch
        num_tokens = grid * grid
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.blocks = nn.Sequential(*[Block(dim, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch(x)
        x = x + self.pos
        x = self.blocks(x)
        x = self.norm(x)
        return x


class FinetuneModel(nn.Module):
    def __init__(self, encoder, out_dim=1):
        super().__init__()
        self.encoder = encoder
        if isinstance(self.encoder.norm.normalized_shape, (tuple, list)):
            feat_dim = int(self.encoder.norm.normalized_shape[0])
        else:
            feat_dim = int(self.encoder.norm.normalized_shape)
        self.head = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        z = self.encoder(x).mean(dim=1)
        return self.head(z)


class JetDataset(Dataset):
    def __init__(self, h5_path, task="cls"):
        self.h5_path = h5_path
        self.file = None
        self.task = task
        self.y_mean = 0.0
        self.y_std = 1.0

        with h5py.File(h5_path, "r") as f:
            self.length = f["jet"].shape[0]
            self.available_keys = set(f.keys())

            if self.task == "mass":
                if "m" not in self.available_keys:
                    raise KeyError("Dataset missing key 'm'")
                targets = f["m"][:]
                self.y_mean = float(targets.mean())
                self.y_std = float(targets.std() + 1e-6)
            elif self.task == "pt":
                if "pT" not in self.available_keys:
                    raise KeyError("Dataset missing key 'pT'")
                targets = f["pT"][:]
                self.y_mean = float(targets.mean())
                self.y_std = float(targets.std() + 1e-6)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        x = self.file["jet"][idx]
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        x = (x - x.mean()) / (x.std() + 1e-6)

        if self.task == "cls":
            y = self.file["Y"][idx]
        elif self.task == "mass":
            raw_y = self.file["m"][idx]
            y = (raw_y - self.y_mean) / self.y_std
        elif self.task == "pt":
            raw_y = self.file["pT"][idx]
            y = (raw_y - self.y_mean) / self.y_std
        else:
            raise ValueError(f"Unknown task: {self.task}")

        y = torch.tensor(y, dtype=torch.float32).squeeze()
        return x, y


def make_loaders(h5_path, task, batch_size=32, val_frac_from_train=0.1, split_seed=42, num_workers=0):
    dataset = JetDataset(h5_path, task)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_full, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    val_size = max(1, int(val_frac_from_train * len(train_full)))
    train_size_final = len(train_full) - val_size

    train_ds, val_ds = random_split(
        train_full,
        [train_size_final, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset


def evaluate_regression(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float().view(-1)
            pred = model(x).squeeze(-1)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_classification(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float().view(-1)
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

            total_loss += loss.item()
            logits_all.append(logits.detach().cpu())
            labels_all.append(y.detach().cpu())

    logits_all = torch.cat(logits_all).numpy()
    labels_all = torch.cat(labels_all).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_all))
    preds = (probs >= 0.5).astype(np.int32)
    acc = float((preds == labels_all.astype(np.int32)).mean())

    return {
        "loss": total_loss / max(1, len(loader)),
        "labels": labels_all.astype(np.int32),
        "probs": probs,
        "preds": preds,
        "acc": acc,
    }


def encoder_kwargs_from_hparams(hp):
    return {
        "dim": int(hp.get("dim", 192)),
        "depth": int(hp.get("depth", 4)),
        "heads": int(hp.get("heads", 6)),
        "patch": int(hp.get("patch", 5)),
        "in_ch": int(hp.get("in_ch", 8)),
    }


def build_model(device, pretrained, hp, ckpt_path=None):
    encoder = Encoder(**encoder_kwargs_from_hparams(hp)).to(device)
    if pretrained:
        if ckpt_path is None:
            raise ValueError("ckpt_path is required when pretrained=True")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        encoder.load_state_dict(state)
    return FinetuneModel(encoder).to(device)


def history_to_csv(path, history):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train", "val", "train_acc", "val_acc"])
        writer.writeheader()
        for idx in range(len(history["train"])):
            writer.writerow(
                {
                    "epoch": idx + 1,
                    "train": history["train"][idx],
                    "val": history["val"][idx],
                    "train_acc": history["train_acc"][idx],
                    "val_acc": history["val_acc"][idx],
                }
            )


def save_val_curve(path, task, pre_hist, scratch_hist):
    if task == "cls":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(pre_hist["val"], label="Pretrained")
        axes[0].plot(scratch_hist["val"], label="Scratch")
        axes[0].set_title("Classification Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(pre_hist["val_acc"], label="Pretrained")
        axes[1].plot(scratch_hist["val_acc"], label="Scratch")
        axes[1].set_title("Classification Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return

    plt.figure(figsize=(6, 4))
    plt.plot(pre_hist["val"], label="Pretrained")
    plt.plot(scratch_hist["val"], label="Scratch")
    plt.title(f"{task} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_cls_diagnostics(run_dir, cls_pre_test, cls_scratch_test):
    cls_results = {}
    for name, out in [("Pretrained", cls_pre_test), ("Scratch", cls_scratch_test)]:
        tn, fp, fn, tp = confusion_matrix(out["labels"], out["preds"]).ravel()
        auc = float(roc_auc_score(out["labels"], out["probs"]))
        acc = float((out["preds"] == out["labels"]).mean())
        cls_results[name] = {
            "loss": float(out["loss"]),
            "accuracy": acc,
            "auc": auc,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = None
    for ax, name in zip(axes, ["Pretrained", "Scratch"]):
        cm = np.array(
            [
                [cls_results[name]["tn"], cls_results[name]["fp"]],
                [cls_results[name]["fn"], cls_results[name]["tp"]],
            ]
        )
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig(run_dir / "classification_confusion_matrices.png", dpi=150)
    plt.close()

    fpr_pre, tpr_pre, _ = roc_curve(cls_pre_test["labels"], cls_pre_test["probs"])
    fpr_scr, tpr_scr, _ = roc_curve(cls_scratch_test["labels"], cls_scratch_test["probs"])

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_pre, tpr_pre, label=f"Pretrained (AUC={cls_results['Pretrained']['auc']:.3f})")
    plt.plot(fpr_scr, tpr_scr, label=f"Scratch (AUC={cls_results['Scratch']['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Classification ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "classification_roc.png", dpi=150)
    plt.close()

    return cls_results


def train_and_eval(
    device,
    h5_path,
    task,
    hp,
    ckpt_path,
    pretrained,
    lr,
    head_lr,
    epochs,
    patience,
    batch_size,
    num_workers,
):
    train_loader, val_loader, test_loader, dataset = make_loaders(
        h5_path=h5_path,
        task=task,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(device=device, pretrained=pretrained, hp=hp, ckpt_path=ckpt_path)

    is_cls = task == "cls"
    criterion = nn.BCEWithLogitsLoss() if is_cls else nn.MSELoss()

    if pretrained:
        optimizer = Adam(
            [
                {"params": model.encoder.parameters(), "lr": lr},
                {"params": model.head.parameters(), "lr": head_lr},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train": [], "val": [], "train_acc": [], "val_acc": []}
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in tqdm(train_loader, leave=False, desc=f"{task} ep {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device).float().view(-1)

            out = model(x).squeeze(-1)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if is_cls:
                probs = torch.sigmoid(out)
                preds = (probs >= 0.5).float()
                train_correct += (preds == y).sum().item()
                train_total += y.numel()

        train_loss /= max(1, len(train_loader))

        if is_cls:
            val_metrics = evaluate_classification(model, val_loader, device)
            val_loss = val_metrics["loss"]
            train_acc = train_correct / max(1, train_total)
            val_acc = val_metrics["acc"]
        else:
            val_loss = evaluate_regression(model, val_loader, device)
            train_acc = np.nan
            val_acc = np.nan

        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["train_acc"].append(float(train_acc) if not np.isnan(train_acc) else np.nan)
        history["val_acc"].append(float(val_acc) if not np.isnan(val_acc) else np.nan)

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if is_cls:
        test_metrics = evaluate_classification(model, test_loader, device)
    else:
        test_metrics = {"loss": evaluate_regression(model, test_loader, device)}

    if task in {"mass", "pt"}:
        test_metrics["loss_denorm"] = float(test_metrics["loss"] * (dataset.y_std ** 2))

    return history, test_metrics, model


def run_checkpoint(args, ckpt_dir):
    hp_path = ckpt_dir / "hyperparams.json"
    ckpt_path = ckpt_dir / "pretrained_encoder_best.pt"

    with hp_path.open("r", encoding="utf-8") as f:
        hp = json.load(f)

    run_out = args.results_dir / ckpt_dir.name
    run_out.mkdir(parents=True, exist_ok=True)

    tasks = {
        "cls": {
            "epochs": args.cls_epochs,
            "pre_lr": args.cls_pre_lr,
            "pre_head_lr": args.cls_pre_head_lr,
            "scratch_lr": args.cls_scratch_lr,
            "scratch_head_lr": args.cls_scratch_head_lr,
        },
        "mass": {
            "epochs": args.mass_epochs,
            "pre_lr": args.mass_pre_lr,
            "pre_head_lr": args.mass_pre_head_lr,
            "scratch_lr": args.mass_scratch_lr,
            "scratch_head_lr": args.mass_scratch_head_lr,
        },
        "pt": {
            "epochs": args.pt_epochs,
            "pre_lr": args.pt_pre_lr,
            "pre_head_lr": args.pt_pre_head_lr,
            "scratch_lr": args.pt_scratch_lr,
            "scratch_head_lr": args.pt_scratch_head_lr,
        },
    }

    all_task_metrics = {}
    summary_rows = []

    for task, cfg in tasks.items():
        print(f"\n[{ckpt_dir.name}] Task={task} | Pretrained")
        pre_hist, pre_test, pre_model = train_and_eval(
            device=args.device,
            h5_path=args.labelled_h5,
            task=task,
            hp=hp,
            ckpt_path=ckpt_path,
            pretrained=True,
            lr=cfg["pre_lr"],
            head_lr=cfg["pre_head_lr"],
            epochs=cfg["epochs"],
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        print(f"[{ckpt_dir.name}] Task={task} | Scratch")
        scratch_hist, scratch_test, scratch_model = train_and_eval(
            device=args.device,
            h5_path=args.labelled_h5,
            task=task,
            hp=hp,
            ckpt_path=ckpt_path,
            pretrained=False,
            lr=cfg["scratch_lr"],
            head_lr=cfg["scratch_head_lr"],
            epochs=cfg["epochs"],
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        history_to_csv(run_out / f"{task}_pretrained_history.csv", pre_hist)
        history_to_csv(run_out / f"{task}_scratch_history.csv", scratch_hist)
        save_val_curve(run_out / f"{task}_val_curves.png", task, pre_hist, scratch_hist)

        torch.save(pre_model.state_dict(), run_out / f"{task}_pretrained_finetune.pt")
        torch.save(scratch_model.state_dict(), run_out / f"{task}_scratch_finetune.pt")

        task_metrics = {
            "pretrained": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in pre_test.items() if k not in {"labels", "preds", "probs"}},
            "scratch": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in scratch_test.items() if k not in {"labels", "preds", "probs"}},
        }

        if task == "cls":
            cls_diag = save_cls_diagnostics(run_out, pre_test, scratch_test)
            task_metrics["pretrained"].update(cls_diag["Pretrained"])
            task_metrics["scratch"].update(cls_diag["Scratch"])

        with (run_out / f"{task}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(task_metrics, f, indent=2)

        all_task_metrics[task] = task_metrics

        row = {
            "checkpoint": ckpt_dir.name,
            "task": task,
            "pre_loss": float(task_metrics["pretrained"].get("loss", np.nan)),
            "scratch_loss": float(task_metrics["scratch"].get("loss", np.nan)),
        }
        if task == "cls":
            row["pre_acc"] = float(task_metrics["pretrained"].get("accuracy", np.nan))
            row["scratch_acc"] = float(task_metrics["scratch"].get("accuracy", np.nan))
            row["pre_auc"] = float(task_metrics["pretrained"].get("auc", np.nan))
            row["scratch_auc"] = float(task_metrics["scratch"].get("auc", np.nan))
        else:
            row["pre_loss_denorm"] = float(task_metrics["pretrained"].get("loss_denorm", np.nan))
            row["scratch_loss_denorm"] = float(task_metrics["scratch"].get("loss_denorm", np.nan))
        summary_rows.append(row)

    with (run_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint": ckpt_dir.name, "tasks": all_task_metrics, "hyperparams": hp}, f, indent=2)

    return summary_rows


def discover_checkpoints(checkpoints_dir: Path, pattern: Optional[str] = None):
    folders = []
    for d in sorted(checkpoints_dir.iterdir()):
        if not d.is_dir():
            continue
        if pattern and pattern not in d.name:
            continue
        if (d / "hyperparams.json").exists() and (d / "pretrained_encoder_best.pt").exists():
            folders.append(d)
    return folders


def write_global_summary(path: Path, rows):
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments for all checkpoints")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--labelled-h5", type=str, default="data/Dataset_Specific_labelled_full_only_for_2i.h5")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-filter", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--patience", type=int, default=1000000)

    parser.add_argument("--cls-epochs", type=int, default=10)
    parser.add_argument("--mass-epochs", type=int, default=25)
    parser.add_argument("--pt-epochs", type=int, default=50)

    parser.add_argument("--cls-pre-lr", type=float, default=1e-4)
    parser.add_argument("--cls-pre-head-lr", type=float, default=1e-3)
    parser.add_argument("--cls-scratch-lr", type=float, default=1e-3)
    parser.add_argument("--cls-scratch-head-lr", type=float, default=1e-3)

    parser.add_argument("--mass-pre-lr", type=float, default=1e-5)
    parser.add_argument("--mass-pre-head-lr", type=float, default=1e-3)
    parser.add_argument("--mass-scratch-lr", type=float, default=1e-3)
    parser.add_argument("--mass-scratch-head-lr", type=float, default=1e-3)

    parser.add_argument("--pt-pre-lr", type=float, default=1e-4)
    parser.add_argument("--pt-pre-head-lr", type=float, default=1e-3)
    parser.add_argument("--pt-scratch-lr", type=float, default=1e-4)
    parser.add_argument("--pt-scratch-head-lr", type=float, default=1e-3)

    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    set_seed(args.seed)

    ckpt_dirs = discover_checkpoints(args.checkpoints_dir, args.checkpoint_filter)
    if not ckpt_dirs:
        raise FileNotFoundError(f"No valid checkpoint folders found in {args.checkpoints_dir}")

    all_rows = []
    for ckpt_dir in ckpt_dirs:
        print(f"\n===== Running checkpoint: {ckpt_dir.name} =====")
        rows = run_checkpoint(args, ckpt_dir)
        all_rows.extend(rows)

    write_global_summary(args.results_dir / "all_checkpoints_summary.csv", all_rows)
    with (args.results_dir / "all_checkpoints_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print("\nCompleted all checkpoint fine-tuning experiments.")
    print(f"Results written to: {args.results_dir}")


if __name__ == "__main__":
    main()
