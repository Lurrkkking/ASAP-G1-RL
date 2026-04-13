import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class TransitionDataset(Dataset):
    """Load transition data from exported NPZ.

    Supported NPZ layouts:
    1) Padded episode layout:
       s:      [E, T, S]
       a:      [E, T, A]
       s_next: [E, T, S]
       mask:   [E, T] (preferred) OR episode_lengths: [E]
    2) Flat layout:
       s:      [N, S]
       a:      [N, A]
       s_next: [N, S]
    """

    def __init__(self, npz_path: Path, target_mode: str = "delta"):
        if target_mode not in ("delta", "next"):
            raise ValueError("target_mode must be 'delta' or 'next'")

        with np.load(npz_path, allow_pickle=False) as d:
            required = ["s", "a", "s_next"]
            for k in required:
                if k not in d:
                    raise KeyError(f"Missing key in npz: {k}")

            s = d["s"].astype(np.float32)
            a = d["a"].astype(np.float32)
            s_next = d["s_next"].astype(np.float32)

            # Convert padded episodes to flat valid transitions.
            if s.ndim == 3:
                if "mask" in d:
                    mask = d["mask"].astype(bool)
                elif "episode_lengths" in d:
                    lengths = d["episode_lengths"].astype(np.int64)
                    mask = np.zeros(s.shape[:2], dtype=bool)
                    for i, l in enumerate(lengths):
                        mask[i, : int(l)] = True
                else:
                    raise KeyError("Padded NPZ needs either 'mask' or 'episode_lengths'")

                x = np.concatenate([s, a], axis=-1)
                y = (s_next - s) if target_mode == "delta" else s_next
                self.x = torch.from_numpy(x[mask])
                self.y = torch.from_numpy(y[mask])

            elif s.ndim == 2:
                x = np.concatenate([s, a], axis=-1)
                y = (s_next - s) if target_mode == "delta" else s_next
                self.x = torch.from_numpy(x)
                self.y = torch.from_numpy(y)
            else:
                raise ValueError(f"Unsupported s shape: {s.shape}")

        if self.x.shape[0] == 0:
            raise ValueError("No valid transitions found in dataset")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ResidualDynamicsMLP(nn.Module):
    """Simple MLP: input = [s_t, a_t], output = delta_s or s_next."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            bs = xb.shape[0]
            loss_sum += loss.item() * bs
            n += bs
    return loss_sum / max(n, 1)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    ds = TransitionDataset(Path(args.npz), target_mode=args.target_mode)
    total = len(ds)
    val_size = max(1, int(total * args.val_ratio))
    train_size = total - val_size
    if train_size < 1:
        raise ValueError("Dataset too small after val split")

    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    in_dim = ds.x.shape[1]
    out_dim = ds.y.shape[1]

    model = ResidualDynamicsMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output artifacts:
    # 1) best_residual_dynamics.pt: model checkpoint (normally do NOT edit manually)
    # 2) train_metrics.json: numeric logs for monitoring/comparison (normally do NOT edit manually)
    ckpt_path = out_dir / "best_residual_dynamics.pt"
    metrics_path = out_dir / "train_metrics.json"

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            bs = xb.shape[0]
            train_sum += loss.item() * bs
            n += bs

        train_loss = train_sum / max(n, 1)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[epoch {epoch:04d}] train_loss={train_loss:.6e} val_loss={val_loss:.6e}")

        # Keep best-by-validation checkpoint.
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "hidden_dim": args.hidden_dim,
                    "depth": args.depth,
                    "target_mode": args.target_mode,
                    "npz": str(args.npz),
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "npz": str(args.npz),
                "num_samples": total,
                "train_size": train_size,
                "val_size": val_size,
                "target_mode": args.target_mode,
                "best_val_loss": best_val,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[DONE] best checkpoint: {ckpt_path}")
    print(f"[DONE] metrics: {metrics_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Train residual dynamics model from exported NPZ transitions")

    # Must set:
    # --npz: your collected dataset path
    p.add_argument("--npz", type=str, required=True, help="Path to residual_dataset.npz")

    # Usually adjust these by experiment:
    p.add_argument("--out-dir", type=str, default="/root/autodl-tmp/ASAP/genesis_simulation/residual_dataset/train_out")
    p.add_argument("--target-mode", type=str, default="delta", choices=["delta", "next"], help="delta: predict s_next-s, next: predict s_next")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Runtime settings:
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
