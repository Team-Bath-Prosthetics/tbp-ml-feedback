"""Training and evaluation loop for the Variational Autoencoder (Beta-VAE) with W&B logging.

Running from the command line:
    Run with default settings (reads from ``data/`` and logs to W&B online)::

        python -m tactile.training.vae_train

    Customise beta, KL warmup, architecture, and save the best checkpoint::

        python -m tactile.training.vae_train \\
            --data-dir data \\
            --hidden-dims 512 128 \\
            --latent-dim 16 \\
            --beta 1.0 \\
            --kl-warmup-epochs 5 \\
            --batch-size 64 \\
            --epochs 20 \\
            --lr 1e-3 \\
            --wandb-project tbp-ml-feedback \\
            --wandb-run-name mlp-vae-baseline \\
            --save-path checkpoints/best_mlp_vae.pt

    Run locally without uploading to W&B (offline or disabled mode)::

        python -m tactile.training.vae_train --data-dir data --wandb-mode disabled
        python -m tactile.training.vae_train --data-dir data --wandb-mode offline

    See all available CLI flags::

        python -m tactile.training.vae_train --help
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.models.vae import MLPVAE, vae_loss
from tactile.training.mlp_ae_train import load_train_val_datasets, resolve_device


def _load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load key-value pairs from a local .env file into os.environ if present."""
    if not dotenv_path.is_file():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


@dataclass
class VAETrainConfig:
    """Configuration for training the model."""

    data_dir: Path = Path("data")
    frame_shape: tuple[int, ...] = FRAME_SHAPE
    hidden_dims: tuple[int, ...] = (512, 128)
    latent_dim: int = 16
    beta: float = 1.0
    kl_warmup_epochs: int = 0
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_fraction: float = 0.2
    seed: int = 42
    device: str = "auto"
    wandb_project: str = "tbp-ml-feedback"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    save_path: Path | None = None


def _effective_beta(epoch: int, target_beta: float, warmup_epochs: int) -> float:
    """Compute effective beta at a 1-indexed ``epoch`` with optional linear KL warmup."""
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return target_beta
    return target_beta * (epoch / float(warmup_epochs))


def train_vae_one_epoch(
    model: MLPVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch and return sample-weighted ELBO, MSE, and KL losses."""
    model.train()
    total_elbo = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        target = model.normalize_input(batch)

        optimizer.zero_grad()
        recon, mu, logvar = model(target, sample=True)
        loss, recon_loss, kl_loss = vae_loss(recon, target, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        batch_size = int(batch.shape[0])
        total_elbo += float(loss.item()) * batch_size
        total_recon += float(recon_loss.item()) * batch_size
        total_kl += float(kl_loss.item()) * batch_size
        total_samples += batch_size

    denom = max(1, total_samples)
    return {
        "train_elbo_loss": total_elbo / denom,
        "train_loss": total_recon / denom,
        "train_kl_loss": total_kl / denom,
    }


@torch.no_grad()
def evaluate_vae(
    model: MLPVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float = 1.0,
) -> dict[str, float]:
    """Evaluate deterministic reconstruction MSE, KL divergence, and ADC MAE on a validation set."""
    model.eval()
    total_elbo = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_adc_mae = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        target = model.normalize_input(batch)

        # Deterministic evaluation (sample=False uses mu)
        recon, mu, logvar = model(target, sample=False)
        loss, recon_loss, kl_loss = vae_loss(recon, target, mu, logvar, beta=beta)
        adc_mae = torch.abs((recon - target) * model.adc_max).mean()

        batch_size = int(batch.shape[0])
        total_elbo += float(loss.item()) * batch_size
        total_recon += float(recon_loss.item()) * batch_size
        total_kl += float(kl_loss.item()) * batch_size
        total_adc_mae += float(adc_mae.item()) * batch_size
        total_samples += batch_size

    denom = max(1, total_samples)
    return {
        "val_elbo_loss": total_elbo / denom,
        "val_loss": total_recon / denom,
        "val_kl_loss": total_kl / denom,
        "val_adc_mae": total_adc_mae / denom,
    }


def train_vae(config: VAETrainConfig | None = None) -> dict[str, float]:
    """Train an MLPVAE on ``.npz`` files and log to W&B."""
    _load_dotenv()
    cfg = config or VAETrainConfig()

    run_config = asdict(cfg)
    run_config["model_type"] = "mlp_vae"
    run_config["data_dir"] = str(cfg.data_dir)
    run_config["save_path"] = str(cfg.save_path) if cfg.save_path is not None else None
    run_config["frame_shape"] = list(cfg.frame_shape)
    run_config["hidden_dims"] = list(cfg.hidden_dims)

    with wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config=run_config,
        mode=cfg.wandb_mode,
    ) as run:
        w_cfg = run.config
        seed = int(w_cfg.get("seed", cfg.seed))
        torch.manual_seed(seed)
        np.random.seed(seed)

        frame_shape = tuple(w_cfg.get("frame_shape", cfg.frame_shape))
        hidden_dims = tuple(int(h) for h in w_cfg.get("hidden_dims", cfg.hidden_dims))
        latent_dim = int(w_cfg.get("latent_dim", cfg.latent_dim))
        beta = float(w_cfg.get("beta", cfg.beta))
        kl_warmup_epochs = int(w_cfg.get("kl_warmup_epochs", cfg.kl_warmup_epochs))
        batch_size = int(w_cfg.get("batch_size", cfg.batch_size))
        epochs = int(w_cfg.get("epochs", cfg.epochs))
        lr = float(w_cfg.get("lr", cfg.lr))
        weight_decay = float(w_cfg.get("weight_decay", cfg.weight_decay))
        val_fraction = float(w_cfg.get("val_fraction", cfg.val_fraction))
        data_dir = Path(w_cfg.get("data_dir", str(cfg.data_dir)))

        train_ds, val_ds = load_train_val_datasets(
            data_dir=data_dir,
            frame_shape=frame_shape,
            val_fraction=val_fraction,
            seed=seed,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        device = resolve_device(str(w_cfg.get("device", cfg.device)))
        model = MLPVAE(
            frame_shape=frame_shape,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            adc_max=ADC_MAX,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        last_metrics: dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            curr_beta = _effective_beta(epoch, beta, kl_warmup_epochs)
            train_metrics = train_vae_one_epoch(model, train_loader, optimizer, device, beta=curr_beta)
            val_metrics = evaluate_vae(model, val_loader, device, beta=beta)

            last_metrics = {
                "epoch": float(epoch),
                "beta": curr_beta,
                **train_metrics,
                **val_metrics,
            }
            run.log(last_metrics, step=epoch)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                run.summary["best_val_loss"] = best_val_loss
                run.summary["best_val_elbo_loss"] = val_metrics["val_elbo_loss"]
                run.summary["best_val_adc_mae"] = val_metrics["val_adc_mae"]
                if cfg.save_path is not None:
                    save_target = Path(cfg.save_path)
                    save_target.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_target)

        last_metrics["best_val_loss"] = best_val_loss
        return last_metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the tactile MLPVAE (Variational Autoencoder) on .npz files and log to W&B."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing .npz files")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 128],
        help="Hidden layer dimensions for encoder/decoder (default: 512 128)",
    )
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent bottleneck dimension (default: 16)")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weight on normalised KL divergence (default: 1.0)")
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm up beta from 0 to --beta over this many epochs (default: 0)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay (default: 0.0)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto, cpu, or cuda")
    parser.add_argument("--wandb-project", type=str, default="tbp-ml-feedback", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team or username)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B logging mode (default: online)",
    )
    parser.add_argument("--save-path", type=Path, default=None, help="Optional path to save best model weights")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = VAETrainConfig(
        data_dir=args.data_dir,
        hidden_dims=tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
        beta=args.beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        save_path=args.save_path,
    )
    train_vae(config)


if __name__ == "__main__":
    main()
