import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class FixedStatePatchDataset(Dataset):
    def __init__(self, npz_path: Path):
        with np.load(npz_path, allow_pickle=False) as d:
            for key in ("s", "a_base"):
                if key not in d:
                    raise KeyError(f"Missing key in npz: {key}")
            self.s = torch.from_numpy(d["s"].astype(np.float32))
            self.a_base = torch.from_numpy(d["a_base"].astype(np.float32))
        if self.s.shape[0] == 0:
            raise ValueError("Empty fixed-state patch dataset")

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return self.s[idx], self.a_base[idx]


class ResidualDynamicsMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, depth: int = 3):
        super().__init__()
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


class DeltaActionPatchMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = state_dim + action_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s, a_base):
        x = torch.cat([s, a_base], dim=-1)
        return self.net(x)


class PatchTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

        ds = FixedStatePatchDataset(Path(args.dataset_npz))
        total = len(ds)
        val_size = max(1, int(total * args.val_ratio))
        train_size = total - val_size
        gen = torch.Generator().manual_seed(args.seed)
        self.train_ds, self.val_ds = random_split(ds, [train_size, val_size], generator=gen)

        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        sample_s, sample_a = ds[0]
        self.state_dim = int(sample_s.shape[0])
        self.action_dim = int(sample_a.shape[0])

        self.patch_model = DeltaActionPatchMLP(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        ).to(self.device)

        gap_ckpt = torch.load(args.gap_model_ckpt, map_location=self.device)
        self.gap_model = ResidualDynamicsMLP(
            in_dim=int(gap_ckpt["in_dim"]),
            out_dim=int(gap_ckpt["out_dim"]),
            hidden_dim=int(gap_ckpt["hidden_dim"]),
            depth=int(gap_ckpt["depth"]),
        ).to(self.device)
        self.gap_model.load_state_dict(gap_ckpt["model_state_dict"])
        self.gap_model.eval()
        for p in self.gap_model.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(self.patch_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.out_dir / "best_delta_action_patch.pt"
        self.last_ckpt_path = self.out_dir / "last_delta_action_patch.pt"
        self.metrics_path = self.out_dir / "train_metrics.json"

    def _checkpoint_payload(self, epoch: int, best_val_loss: float):
        return {
            "model_state_dict": self.patch_model.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.args.hidden_dim,
            "depth": self.args.depth,
            "max_delta_scale": self.args.max_delta_scale,
            "dataset_npz": self.args.dataset_npz,
            "gap_model_ckpt": self.args.gap_model_ckpt,
            "best_val_loss": best_val_loss,
            "epoch": epoch,
        }

    def _loss_terms(self, s, a_base):
        raw_delta = self.patch_model(s, a_base)
        delta = torch.tanh(raw_delta) * self.args.max_delta_scale
        a_patch = a_base + delta
        gap_pred = self.gap_model(torch.cat([s, a_patch], dim=-1))

        gap_abs = torch.abs(gap_pred)
        if gap_abs.shape[1] < 53:
            raise ValueError(f"Expected gap model output >=53 dims, got {gap_abs.shape[1]}")

        root_loss = gap_abs[:, :7].mean()
        dof_pos_loss = gap_abs[:, 7:30].mean()
        dof_vel_loss = gap_abs[:, 30:53].mean()
        gap_loss = (
            self.args.root_weight * root_loss
            + self.args.dof_pos_weight * dof_pos_loss
            + self.args.dof_vel_weight * dof_vel_loss
        )
        patch_l2 = torch.mean(delta ** 2)
        patch_l1 = torch.mean(torch.abs(delta))
        total = gap_loss + self.args.patch_l2_weight * patch_l2 + self.args.patch_l1_weight * patch_l1
        return total, gap_loss, patch_l2, patch_l1

    def _run_loader(self, loader, train: bool):
        if train:
            self.patch_model.train()
        else:
            self.patch_model.eval()

        total_loss = 0.0
        total_gap = 0.0
        total_l2 = 0.0
        total_l1 = 0.0
        n = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for s, a_base in loader:
                s = s.to(self.device)
                a_base = a_base.to(self.device)
                loss, gap_loss, patch_l2, patch_l1 = self._loss_terms(s, a_base)
                if train:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.patch_model.parameters(), self.args.grad_clip)
                    self.opt.step()
                bs = s.shape[0]
                total_loss += float(loss.item()) * bs
                total_gap += float(gap_loss.item()) * bs
                total_l2 += float(patch_l2.item()) * bs
                total_l1 += float(patch_l1.item()) * bs
                n += bs

        return {
            "loss": total_loss / max(n, 1),
            "gap_loss": total_gap / max(n, 1),
            "patch_l2": total_l2 / max(n, 1),
            "patch_l1": total_l1 / max(n, 1),
        }

    def train(self):
        history = []
        best_val = float("inf")
        for epoch in range(1, self.args.epochs + 1):
            train_stats = self._run_loader(self.train_loader, train=True)
            val_stats = self._run_loader(self.val_loader, train=False)
            row = {"epoch": epoch, "train": train_stats, "val": val_stats}
            history.append(row)
            print(
                f"[epoch {epoch:04d}] "
                f"train_loss={train_stats['loss']:.6e} val_loss={val_stats['loss']:.6e} "
                f"val_gap={val_stats['gap_loss']:.6e} val_l2={val_stats['patch_l2']:.6e}"
            )
            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                torch.save(self._checkpoint_payload(epoch=epoch, best_val_loss=best_val), self.ckpt_path)
            if self.args.save_every > 0 and epoch % self.args.save_every == 0:
                periodic_ckpt = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
                torch.save(self._checkpoint_payload(epoch=epoch, best_val_loss=best_val), periodic_ckpt)

        torch.save(self._checkpoint_payload(epoch=self.args.epochs, best_val_loss=best_val), self.last_ckpt_path)

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_npz": self.args.dataset_npz,
                    "gap_model_ckpt": self.args.gap_model_ckpt,
                    "max_delta_scale": self.args.max_delta_scale,
                    "root_weight": self.args.root_weight,
                    "dof_pos_weight": self.args.dof_pos_weight,
                    "dof_vel_weight": self.args.dof_vel_weight,
                    "patch_l2_weight": self.args.patch_l2_weight,
                    "patch_l1_weight": self.args.patch_l1_weight,
                    "best_val_loss": best_val,
                    "history": history,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[DONE] best checkpoint: {self.ckpt_path}")
        print(f"[DONE] metrics: {self.metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train offline delta-action patch model on fixed states")
    parser.add_argument("--dataset-npz", type=str, required=True)
    parser.add_argument("--gap-model-ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-delta-scale", type=float, default=0.05)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--dof-pos-weight", type=float, default=1.0)
    parser.add_argument("--dof-vel-weight", type=float, default=1.0)
    parser.add_argument("--patch-l2-weight", type=float, default=10.0)
    parser.add_argument("--patch-l1-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    PatchTrainer(args).train()


if __name__ == "__main__":
    main()
