"""Unit tests for the MLPVAE training and W&B logging pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

torch = pytest.importorskip("torch")
wandb = pytest.importorskip("wandb")

from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.training import VAETrainConfig, train_vae


def _write_synthetic_npz(path: Path, n_frames: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, ADC_MAX + 1, size=(n_frames, *FRAME_SHAPE), dtype=np.uint16)
    np.savez_compressed(path, frames=frames, fs=np.float32(100.0), meta="{}")


def test_train_vae_logs_metrics_to_wandb_and_saves_checkpoint(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        _write_synthetic_npz(data_dir / f"rec_{i}.npz", n_frames=8, seed=i)

    checkpoint_path = tmp_path / "checkpoints" / "best_mlp_vae.pt"
    cfg = VAETrainConfig(
        data_dir=data_dir,
        hidden_dims=(64, 32),
        latent_dim=8,
        beta=1.0,
        kl_warmup_epochs=2,
        batch_size=8,
        epochs=2,
        lr=1e-3,
        wandb_mode="disabled",
        save_path=checkpoint_path,
    )

    mock_run = MagicMock()
    mock_run.config = {}
    mock_run.summary = {}
    mock_run.__enter__.return_value = mock_run
    mock_run.__exit__.return_value = False

    with patch("tactile.training.vae_train.wandb.init", return_value=mock_run) as mock_init:
        metrics = train_vae(cfg)

    mock_init.assert_called_once()
    assert mock_run.log.call_count == 2
    logged_payload = mock_run.log.call_args.args[0]
    expected_keys = {
        "epoch",
        "beta",
        "train_elbo_loss",
        "train_loss",
        "train_kl_loss",
        "val_elbo_loss",
        "val_loss",
        "val_kl_loss",
        "val_adc_mae",
    }
    assert expected_keys <= set(logged_payload.keys())
    assert "best_val_loss" in metrics
    assert checkpoint_path.exists()
