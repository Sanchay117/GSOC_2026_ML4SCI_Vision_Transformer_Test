import argparse
import csv
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JetUnlabelledDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.file = None
        with h5py.File(h5_path, "r") as f:
            self.length = f["jet"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        x = self.file["jet"][idx]
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=8, embed_dim=256, patch=5):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch,
            stride=patch,
        )

    def forward(self, x):
        x = self.proj(x)
        bsz, dim, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(bsz, h * w, dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, eps=1e-6):
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
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.attn = Attention(dim, heads=heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(self.n2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, dim=256, depth=4, heads=8, patch=5, img_size=125):
        super().__init__()
        if img_size % patch != 0:
            raise ValueError("img_size must be divisible by patch")

        n_patches = (img_size // patch) ** 2
        self.patch = PatchEmbed(in_ch=8, embed_dim=dim, patch=patch)
        self.pos = nn.Parameter(torch.randn(1, n_patches, dim))
        self.blocks = nn.Sequential(*[Block(dim, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch(x)
        x = x + self.pos
        x = self.blocks(x)
        x = self.norm(x)
        return x


class MAE(nn.Module):
    def __init__(self, dim=256, depth=4, heads=8, patch=5, img_size=125, mask_ratio=0.60):
        super().__init__()
        self.encoder = Encoder(dim=dim, depth=depth, heads=heads, patch=patch, img_size=img_size)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 8 * patch * patch),
        )
        self.mask_ratio = mask_ratio
        self.patch_size = patch

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        bsz, channels, height, width = imgs.shape
        p = self.patch_size
        h, w = height // p, width // p
        x = imgs.reshape(bsz, channels, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(bsz, h * w, channels * p * p)
        return x

    def forward(self, x):
        bsz = x.size(0)
        target = self.patchify(x)

        tokens = self.encoder.patch(x)
        bsz, n_tokens, _ = tokens.shape

        mask_count = max(1, int(n_tokens * self.mask_ratio))
        noise = torch.rand(bsz, n_tokens, device=x.device)
        ids = torch.argsort(noise, dim=1)
        mask_idx = ids[:, :mask_count]

        masked_tokens = tokens.clone()
        batch_range = torch.arange(bsz, device=x.device)[:, None]
        masked_tokens[batch_range, mask_idx] = self.mask_token

        masked_tokens = masked_tokens + self.encoder.pos
        encoded = self.encoder.blocks(masked_tokens)
        encoded = self.encoder.norm(encoded)
        reconstruction = self.decoder(encoded)

        pred_masked = reconstruction[batch_range, mask_idx]
        target_masked = target[batch_range, mask_idx]
        loss = ((pred_masked - target_masked) ** 2).mean()
        return loss


def format_run_name(cfg):
    return (
        f"{cfg['name']}"
        f"_mask{int(cfg['mask_ratio'] * 100)}"
        f"_bs{cfg['batch_size']}"
        f"_p{cfg['patch']}"
        f"_dim{cfg['dim']}"
        f"_d{cfg['depth']}"
        f"_h{cfg['heads']}"
        f"_lr{cfg['lr']}"
        f"_wd{cfg['weight_decay']}"
        f"_ep{cfg['epochs']}"
        f"_seed{cfg['seed']}"
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def save_loss_plot(run_dir: Path, train_history, val_history) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MAE Pretraining Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close()


def create_loaders(h5_path: str, batch_size: int, val_frac: float, seed: int, num_workers: int, device: torch.device):
    dataset = JetUnlabelledDataset(h5_path)

    val_size = max(1, int(val_frac * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_one_run(cfg: dict, base_output_dir: Path, h5_path: str, device: torch.device, num_workers: int, overwrite: bool = False):
    run_name = format_run_name(cfg)
    run_dir = base_output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and (run_dir / "done.json").exists():
        print(f"[SKIP] {run_name} already finished")
        return {
            "run_name": run_name,
            "status": "skipped",
            "run_dir": str(run_dir),
        }

    write_json(run_dir / "hyperparams.json", cfg)
    log_path = run_dir / "train.log"

    def log(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"Starting run: {run_name}")
    log(f"Using device: {device}")

    set_seed(cfg["seed"])
    train_loader, val_loader = create_loaders(
        h5_path=h5_path,
        batch_size=cfg["batch_size"],
        val_frac=cfg["val_frac"],
        seed=cfg["seed"],
        num_workers=num_workers,
        device=device,
    )

    model = MAE(
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        patch=cfg["patch"],
        mask_ratio=cfg["mask_ratio"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = math.inf
    best_state = None
    train_history = []
    val_history = []
    csv_path = run_dir / "epoch_metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best_val_so_far"])
        writer.writeheader()

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"{run_name} | ep {epoch}/{cfg['epochs']}", leave=False)

            for batch in loop:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=True)

                loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                if cfg["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

                train_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= max(1, len(train_loader))

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    batch = batch.to(device, non_blocking=True)
                    loss = model(batch)
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))

            train_history.append(train_loss)
            val_history.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_so_far": best_val,
                }
            )
            log(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best_val={best_val:.6f}")

    torch.save(model.state_dict(), run_dir / "mae_last.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "mae_best.pt")
    torch.save(model.encoder.state_dict(), run_dir / "pretrained_encoder_best.pt")

    save_loss_plot(run_dir, train_history, val_history)

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "mask_ratio": cfg["mask_ratio"],
        "batch_size": cfg["batch_size"],
        "patch": cfg["patch"],
        "dim": cfg["dim"],
        "depth": cfg["depth"],
        "heads": cfg["heads"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "seed": cfg["seed"],
        "best_val_loss": float(best_val),
        "final_train_loss": float(train_history[-1]),
        "final_val_loss": float(val_history[-1]),
        "epochs": cfg["epochs"],
        "status": "completed",
    }
    write_json(run_dir / "done.json", summary)
    log(f"Finished run: {run_name}")
    return summary


DEFAULT_EXPERIMENTS = [
    {
        "name": "linear_base_lowmask",
        "mask_ratio": 0.40,
        "batch_size": 32,
        "dim": 192,
        "depth": 4,
        "heads": 6,
        "patch": 5,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_base",
        "mask_ratio": 0.60,
        "batch_size": 32,
        "dim": 256,
        "depth": 4,
        "heads": 8,
        "patch": 5,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_highmask",
        "mask_ratio": 0.75,
        "batch_size": 32,
        "dim": 256,
        "depth": 4,
        "heads": 8,
        "patch": 5,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_midmask_bigbatch",
        "mask_ratio": 0.50,
        "batch_size": 48,
        "dim": 256,
        "depth": 4,
        "heads": 8,
        "patch": 5,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_midmask_smallbatch",
        "mask_ratio": 0.50,
        "batch_size": 16,
        "dim": 256,
        "depth": 4,
        "heads": 8,
        "patch": 5,
        "lr": 8e-5,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_deep",
        "mask_ratio": 0.60,
        "batch_size": 24,
        "dim": 256,
        "depth": 6,
        "heads": 8,
        "patch": 5,
        "lr": 8e-5,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_deeper",
        "mask_ratio": 0.60,
        "batch_size": 16,
        "dim": 256,
        "depth": 8,
        "heads": 8,
        "patch": 5,
        "lr": 6e-5,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_wide",
        "mask_ratio": 0.60,
        "batch_size": 16,
        "dim": 320,
        "depth": 4,
        "heads": 10,
        "patch": 5,
        "lr": 8e-5,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_compact",
        "mask_ratio": 0.60,
        "batch_size": 64,
        "dim": 128,
        "depth": 4,
        "heads": 4,
        "patch": 5,
        "lr": 1.5e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_large",
        "mask_ratio": 0.60,
        "batch_size": 12,
        "dim": 384,
        "depth": 6,
        "heads": 12,
        "patch": 5,
        "lr": 6e-5,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_patch25",
        "mask_ratio": 0.60,
        "batch_size": 64,
        "dim": 256,
        "depth": 6,
        "heads": 8,
        "patch": 25,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_lowwd",
        "mask_ratio": 0.60,
        "batch_size": 32,
        "dim": 256,
        "depth": 4,
        "heads": 8,
        "patch": 5,
        "lr": 1e-4,
        "weight_decay": 5e-5,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
    {
        "name": "linear_highwd",
        "mask_ratio": 0.60,
        "batch_size": 32,
        "dim": 256,
        "depth": 4,
        "heads": 8,
        "patch": 5,
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "epochs": 100,
        "val_frac": 0.05,
        "grad_clip": 1.0,
        "seed": 67,
    },
]


def summarize_sweep(configs):
    keys = [
        "mask_ratio",
        "batch_size",
        "patch",
        "dim",
        "depth",
        "heads",
        "lr",
        "weight_decay",
        "epochs",
        "seed",
    ]
    summary = {k: sorted({cfg[k] for cfg in configs}) for k in keys}
    print("Sweep summary:")
    for key, values in summary.items():
        print(f"  - {key}: {values}")


def parse_args():
    parser = argparse.ArgumentParser(description="Linear-attention ViT pretraining sweep")
    parser.add_argument(
        "--h5-path",
        type=str,
        default="data/Dataset_Specific_Unlabelled.h5",
        help="Path to unlabelled HDF5 dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory for experiment outputs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Run only first N configs from default sweep",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun even when done.json exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = DEFAULT_EXPERIMENTS if args.max_runs is None else DEFAULT_EXPERIMENTS[: args.max_runs]
    summarize_sweep(configs)
    summaries = []

    for cfg in configs:
        summary = train_one_run(
            cfg=cfg,
            base_output_dir=output_dir,
            h5_path=args.h5_path,
            device=device,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    summary_path = output_dir / "summary.json"
    write_json(summary_path, {"runs": summaries})

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_name",
            "status",
            "mask_ratio",
            "batch_size",
            "patch",
            "dim",
            "depth",
            "heads",
            "lr",
            "weight_decay",
            "seed",
            "best_val_loss",
            "final_train_loss",
            "final_val_loss",
            "run_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    "run_name": row.get("run_name"),
                    "status": row.get("status"),
                    "mask_ratio": row.get("mask_ratio"),
                    "batch_size": row.get("batch_size"),
                    "patch": row.get("patch"),
                    "dim": row.get("dim"),
                    "depth": row.get("depth"),
                    "heads": row.get("heads"),
                    "lr": row.get("lr"),
                    "weight_decay": row.get("weight_decay"),
                    "seed": row.get("seed"),
                    "best_val_loss": row.get("best_val_loss"),
                    "final_train_loss": row.get("final_train_loss"),
                    "final_val_loss": row.get("final_val_loss"),
                    "run_dir": row.get("run_dir"),
                }
            )

    print(f"Finished {len(summaries)} runs. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()