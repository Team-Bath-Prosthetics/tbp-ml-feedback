"""Training and evaluation loop for stadard MLP autoencoder.

Running from the command line:
    Run with default settings (reads from ``data/`` and logs to W&B online)::

        python -m tactile.training.mlp_ae_train

    Customise hyperparameters, W&B project/run name, and save the best checkpoint::

        python -m tactile.training.mlp_ae_train \\
            --data-dir data \\
            --hidden-dims 512 128 \\
            --latent-dim 16 \\
            --batch-size 64 \\
            --epochs 20 \\
            --lr 1e-3 \\
            --wandb-project tbp-ml-feedback \\
            --wandb-run-name mlp-ae-baseline \\
            --save-path checkpoints/best_mlp_ae.pt

    Run locally without uploading to W&B (offline or disabled mode)::

        python -m tactile.training.mlp_ae_train --data-dir data --wandb-mode disabled
        python -m tactile.training.mlp_ae_train --data-dir data --wandb-mode offline

    See all available CLI flags::

        python -m tactile.training.mlp_ae_train --help
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

from tactile.data_format import ADC_MAX, FRAME_SHAPE, load_recording
from tactile.models.mlp_ae import MLPAutoencoder


@dataclass
class TrainConfig:
    """Configuration for training the MLP autoencoder."""

    data_dir: Path = Path("data")
    frame_shape: tuple[int, ...] = FRAME_SHAPE
    hidden_dims: tuple[int, ...] = (512, 128)
    latent_dim: int = 16
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


class TactileFrameDataset(Dataset):
    """PyTorch Dataset wrapping a stack of tactile frames of shape ``(N, *frame_shape)``."""

    def __init__(self, frames: np.ndarray | torch.Tensor) -> None:
        if isinstance(frames, np.ndarray):
            # Keep uint16 or float32 compact in memory; MLPAutoencoder normalises on forward/encode
            if frames.dtype == np.uint16:
                self.frames = torch.from_numpy(frames)
            else:
                self.frames = torch.as_tensor(frames, dtype=torch.float32)
        else:
            self.frames = frames

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.frames[index]


def _load_frames_from_npz(path: Path, frame_shape: tuple[int, ...] = FRAME_SHAPE) -> np.ndarray:
    """Load the ``frames`` array from a single ``.npz`` recording file.

    Uses :func:`tactile.data_format.load_recording` first and falls back to
    reading the ``.npz`` archive directly if ``load_recording`` is not yet
    implemented.
    """
    try:
        recording = load_recording(path)
        frames = np.asarray(recording["frames"])
    except NotImplementedError:
        with np.load(path) as data:
            if "frames" not in data:
                raise KeyError(f"Recording file {path} is missing the required 'frames' array.")
            frames = np.asarray(data["frames"])

    n_spatial = len(frame_shape)
    if frames.ndim != n_spatial + 1 or tuple(frames.shape[1:]) != tuple(frame_shape):
        raise ValueError(
            f"Recording {path} has frames of shape {frames.shape}, "
            f"expected (T, *{frame_shape})."
        )
    if frames.shape[0] == 0:
        raise ValueError(f"Recording {path} contains 0 frames.")
    return frames


def load_train_val_datasets(
    data_dir: Path | str,
    frame_shape: tuple[int, ...] = FRAME_SHAPE,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[TactileFrameDataset, TactileFrameDataset]:
    """Load ``.npz`` recordings from ``data_dir`` and split into train and val datasets.

    Splits at the recording (``.npz`` file) level when two or more files exist,
    falling back to a frame-level split when only a single ``.npz`` file is present.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz recording files found in {data_path.resolve()}")

    rng = np.random.default_rng(seed)

    if len(npz_files) >= 2:
        perm = rng.permutation(len(npz_files))
        n_val = max(1, min(len(npz_files) - 1, int(round(len(npz_files) * val_fraction))))
        val_indices = set(perm[:n_val].tolist())

        train_arrays = [
            _load_frames_from_npz(npz_files[i], frame_shape)
            for i in range(len(npz_files))
            if i not in val_indices
        ]
        val_arrays = [
            _load_frames_from_npz(npz_files[i], frame_shape)
            for i in range(len(npz_files))
            if i in val_indices
        ]
        train_frames = np.concatenate(train_arrays, axis=0)
        val_frames = np.concatenate(val_arrays, axis=0)
    else:
        all_frames = _load_frames_from_npz(npz_files[0], frame_shape)
        n_total = all_frames.shape[0]
        if n_total < 2:
            raise ValueError(
                f"Single recording {npz_files[0]} must contain at least 2 frames to split, got {n_total}."
            )
        perm = rng.permutation(n_total)
        n_val = max(1, min(n_total - 1, int(round(n_total * val_fraction))))
        val_frames = all_frames[perm[:n_val]]
        train_frames = all_frames[perm[n_val:]]

    return TactileFrameDataset(train_frames), TactileFrameDataset(val_frames)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``'auto'`` to ``cuda`` if available, otherwise ``cpu``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_one_epoch(
    model: MLPAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the sample-weighted MSE reconstruction loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        target = model.normalize_input(batch)

        optimizer.zero_grad()
        recon = model(target)
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()

        batch_size = int(batch.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: MLPAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate reconstruction MSE on [0, 1] frames and MAE in raw ADC counts (0..ADC_MAX)."""
    model.eval()
    total_mse = 0.0
    total_adc_mae = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        target = model.normalize_input(batch)
        recon = model(target)

        mse = F.mse_loss(recon, target)
        adc_mae = torch.abs((recon - target) * model.adc_max).mean()

        batch_size = int(batch.shape[0])
        total_mse += float(mse.item()) * batch_size
        total_adc_mae += float(adc_mae.item()) * batch_size
        total_samples += batch_size

    denom = max(1, total_samples)
    return {
        "val_loss": total_mse / denom,
        "val_adc_mae": total_adc_mae / denom,
    }


def train(config: TrainConfig | None = None) -> dict[str, float]:
    """Train an :class:`MLPAutoencoder` on ``.npz`` recordings in ``config.data_dir`` and log to W&B.

    Returns a dictionary with the final and best metrics from the run.
    """
    cfg = config or TrainConfig()

    run_config = asdict(cfg)
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
        # Allow W&B sweeps to override hyperparameters dynamically
        w_cfg = run.config
        seed = int(w_cfg.get("seed", cfg.seed))
        torch.manual_seed(seed)
        np.random.seed(seed)

        frame_shape = tuple(w_cfg.get("frame_shape", cfg.frame_shape))
        hidden_dims = tuple(int(h) for h in w_cfg.get("hidden_dims", cfg.hidden_dims))
        latent_dim = int(w_cfg.get("latent_dim", cfg.latent_dim))
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
        model = MLPAutoencoder(
            frame_shape=frame_shape,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            adc_max=ADC_MAX,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        last_metrics: dict[str, float] = {}

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

            last_metrics = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_adc_mae": val_metrics["val_adc_mae"],
            }
            run.log(last_metrics, step=epoch)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                run.summary["best_val_loss"] = best_val_loss
                run.summary["best_val_adc_mae"] = val_metrics["val_adc_mae"]
                if cfg.save_path is not None:
                    save_target = Path(cfg.save_path)
                    save_target.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_target)

        last_metrics["best_val_loss"] = best_val_loss
        return last_metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the tactile MLPAutoencoder on .npz files and log to Weights & Biases."
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
    config = TrainConfig(
        data_dir=args.data_dir,
        hidden_dims=tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
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
    train(config)


if __name__ == "__main__":
    main()

