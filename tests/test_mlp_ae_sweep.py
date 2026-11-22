"""Checks for the MLP autoencoder W&B sweep config (configs/mlp_ae_sweep.yaml)."""

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("wandb")
yaml = pytest.importorskip("yaml")

from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.models.mlp_ae import MLPAutoencoder
from tactile.training import TrainConfig, load_train_val_datasets, train

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = REPO_ROOT / "configs" / "mlp_ae_sweep.yaml"


@pytest.fixture
def sweep():
    return yaml.safe_load(SWEEP_PATH.read_text())


def _write_npz(path: Path, n_frames: int = 8, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, ADC_MAX + 1, size=(n_frames, *FRAME_SHAPE), dtype=np.uint16)
    np.savez_compressed(path, frames=frames, fs=np.float32(30.0), meta=np.array("{}"))


def test_sweep_program_exists(sweep):
    assert (REPO_ROOT / sweep["program"]).is_file()


def test_sweep_parameters_are_train_config_fields(sweep):
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(sweep["parameters"]) - fields
    assert not unknown, f"sweep parameters train() would ignore: {unknown}"


def test_sweep_metric_is_logged_by_train(sweep):
    assert sweep["metric"]["name"] in {"val_loss", "val_adc_mae", "train_loss"}


def test_sweep_values_override_train_config(tmp_path: Path):
    """Simulate an agent run: run.config holds sampled values that train() must use."""
    for i in range(3):
        _write_npz(tmp_path / f"rec_{i}.npz", seed=i)

    sampled = {"hidden_dims": [32, 16], "latent_dim": 4, "batch_size": 4, "epochs": 3,
               "lr": 1e-3, "weight_decay": 1e-4, "data_dir": str(tmp_path)}
    mock_run = MagicMock()
    mock_run.config = sampled
    mock_run.summary = {}
    mock_run.__enter__.return_value = mock_run

    with patch("tactile.training.mlp_ae_train.wandb.init", return_value=mock_run), \
         patch("tactile.training.mlp_ae_train.MLPAutoencoder", wraps=MLPAutoencoder) as model_cls:
        train(TrainConfig(data_dir=Path("does-not-exist"), wandb_mode="disabled"))

    kwargs = model_cls.call_args.kwargs
    assert kwargs["hidden_dims"] == (32, 16)
    assert kwargs["latent_dim"] == 4
    assert mock_run.log.call_count == 3


def test_relative_data_dir_is_resolved_from_cwd(tmp_path: Path, monkeypatch):
    (tmp_path / "data").mkdir()
    _write_npz(tmp_path / "data" / "rec.npz", n_frames=10)
    monkeypatch.chdir(tmp_path)

    train_ds, val_ds = load_train_val_datasets("data")
    assert len(train_ds) + len(val_ds) == 10
