"""Unit tests for the tactile autoencoder training and W&B logging pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

torch = pytest.importorskip("torch")
wandb = pytest.importorskip("wandb")

from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.training import TrainConfig, load_train_val_datasets, train


def _write_synthetic_npz(path: Path, n_frames: int = 10, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, ADC_MAX + 1, size=(n_frames, *FRAME_SHAPE), dtype=np.uint16)
    np.savez_compressed(path, frames=frames, fs=np.float32(100.0), meta="{}")


def test_load_train_val_datasets_multi_file_split(tmp_path: Path):
    for i in range(5):
        _write_synthetic_npz(tmp_path / f"rec_{i}.npz", n_frames=6, seed=i)

    train_ds, val_ds = load_train_val_datasets(tmp_path, val_fraction=0.2, seed=42)
    # 4 files (24 frames) in train, 1 file (6 frames) in val
    assert len(train_ds) == 24
    assert len(val_ds) == 6
    assert train_ds[0].shape == FRAME_SHAPE
    assert train_ds[0].dtype == torch.uint16


def test_load_train_val_datasets_single_file_fallback(tmp_path: Path):
    _write_synthetic_npz(tmp_path / "only_one.npz", n_frames=10, seed=1)

    train_ds, val_ds = load_train_val_datasets(tmp_path, val_fraction=0.2, seed=42)
    assert len(train_ds) == 8
    assert len(val_ds) == 2


def test_train_logs_metrics_to_wandb_and_saves_checkpoint(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        _write_synthetic_npz(data_dir / f"rec_{i}.npz", n_frames=8, seed=i)

    checkpoint_path = tmp_path / "checkpoints" / "best_mlp_ae.pt"
    cfg = TrainConfig(
        data_dir=data_dir,
        hidden_dims=(64, 32),
        latent_dim=8,
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

    with patch("tactile.training.mlp_ae_train.wandb.init", return_value=mock_run) as mock_init:
        metrics = train(cfg)

    mock_init.assert_called_once()
    assert mock_run.log.call_count == 2
    logged_payload = mock_run.log.call_args.args[0]
    assert {"epoch", "train_loss", "val_loss", "val_adc_mae"} <= set(logged_payload.keys())
    assert "best_val_loss" in metrics
    assert checkpoint_path.exists()

