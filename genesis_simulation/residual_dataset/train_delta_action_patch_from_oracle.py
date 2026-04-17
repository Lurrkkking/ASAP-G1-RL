import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class OraclePatchDataset(Dataset):
    def __init__(self, npz_path: Path):
        with np.load(npz_path, allow_pickle=False) as d:
            required = ("s", "a_base", "delta_a_star", "sensitivity", "sample_weight")
            for key in required:
                if key not in d:
                    raise KeyError(f"Missing key in npz: {key}")
            self.s = torch.from_numpy(d["s"].astype(np.float32))
            self.a_base = torch.from_numpy(d["a_base"].astype(np.float32))
            self.delta_a_star = torch.from_numpy(d["delta_a_star"].astype(np.float32))
            self.sensitivity = torch.from_numpy(d["sensitivity"].astype(np.float32))
            self.sample_weight = torch.from_numpy(d["sample_weight"].astype(np.float32))
        if self.s.shape[0] == 0:
            raise ValueError("Empty oracle patch dataset")

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return (
            self.s[idx],
            self.a_base[idx],
            self.delta_a_star[idx],
            self.sensitivity[idx],
            self.sample_weight[idx],
        )


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
        return self.net(torch.cat([s, a_base], dim=-1))


class OraclePatchTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

        ds = OraclePatchDataset(Path(args.dataset_npz))
        total = len(ds)
        val_size = max(1, int(total * args.val_ratio))
        train_size = total - val_size
        gen = torch.Generator().manual_seed(args.seed)
        self.train_ds, self.val_ds = random_split(ds, [train_size, val_size], generator=gen)

        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        sample_s, sample_a, sample_delta, sample_sens, _ = ds[0]
        self.state_dim = int(sample_s.shape[0])
        self.action_dim = int(sample_a.shape[0])
        if sample_delta.shape[0] != self.action_dim or sample_sens.shape[0] != self.action_dim:
            raise ValueError("Action-target dimension mismatch in oracle dataset")

        self.patch_model = DeltaActionPatchMLP(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        ).to(self.device)
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
            "best_val_loss": best_val_loss,
            "epoch": epoch,
        }

    def _loss_terms(self, s, a_base, delta_target, sensitivity, sample_weight):
        raw_delta = self.patch_model(s, a_base)
        delta_pred = torch.tanh(raw_delta) * self.args.max_delta_scale
        diff = delta_pred - delta_target

        plain_mse_per_sample = torch.mean(diff ** 2, dim=-1)
        sens_norm = sensitivity / torch.clamp(torch.mean(sensitivity, dim=-1, keepdim=True), min=1e-8)
        jac_mse_per_sample = torch.mean((diff ** 2) * sens_norm, dim=-1)
        patch_l2_per_sample = torch.mean(delta_pred ** 2, dim=-1)
        patch_l1_per_sample = torch.mean(torch.abs(delta_pred), dim=-1)

        sw = sample_weight / torch.clamp(torch.mean(sample_weight), min=1e-8)
        plain_mse = torch.mean(plain_mse_per_sample * sw)
        jac_mse = torch.mean(jac_mse_per_sample * sw)
        patch_l2 = torch.mean(patch_l2_per_sample)
        patch_l1 = torch.mean(patch_l1_per_sample)
        total = (
            self.args.delta_mse_weight * plain_mse
            + self.args.jacobian_weight * jac_mse
            + self.args.patch_l2_weight * patch_l2
            + self.args.patch_l1_weight * patch_l1
        )
        return total, plain_mse, jac_mse, patch_l2, patch_l1

    def _run_loader(self, loader, train: bool):
        if train:
            self.patch_model.train()
        else:
            self.patch_model.eval()

        total_loss = 0.0
        total_plain = 0.0
        total_jac = 0.0
        total_l2 = 0.0
        total_l1 = 0.0
        n = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for s, a_base, delta_target, sensitivity, sample_weight in loader:
                s = s.to(self.device)
                a_base = a_base.to(self.device)
                delta_target = delta_target.to(self.device)
                sensitivity = sensitivity.to(self.device)
                sample_weight = sample_weight.to(self.device)
                loss, plain_mse, jac_mse, patch_l2, patch_l1 = self._loss_terms(
                    s, a_base, delta_target, sensitivity, sample_weight
                )
                if train:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.patch_model.parameters(), self.args.grad_clip)
                    self.opt.step()
                bs = s.shape[0]
                total_loss += float(loss.item()) * bs
                total_plain += float(plain_mse.item()) * bs
                total_jac += float(jac_mse.item()) * bs
                total_l2 += float(patch_l2.item()) * bs
                total_l1 += float(patch_l1.item()) * bs
                n += bs

        return {
            "loss": total_loss / max(n, 1),
            "plain_mse": total_plain / max(n, 1),
            "jacobian_mse": total_jac / max(n, 1),
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
                f"val_plain={val_stats['plain_mse']:.6e} val_jac={val_stats['jacobian_mse']:.6e} "
                f"val_l2={val_stats['patch_l2']:.6e}"
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
                    "max_delta_scale": self.args.max_delta_scale,
                    "delta_mse_weight": self.args.delta_mse_weight,
                    "jacobian_weight": self.args.jacobian_weight,
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
    parser = argparse.ArgumentParser(description="Train supervised delta-action patch model from oracle a_star labels")
    parser.add_argument("--dataset-npz", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-delta-scale", type=float, default=0.10)
    parser.add_argument("--delta-mse-weight", type=float, default=1.0)
    parser.add_argument("--jacobian-weight", type=float, default=1.0)
    parser.add_argument("--patch-l2-weight", type=float, default=1.0)
    parser.add_argument("--patch-l1-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    OraclePatchTrainer(args).train()


if __name__ == "__main__":
    main()
