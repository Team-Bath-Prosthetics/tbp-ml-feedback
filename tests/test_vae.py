"""Unit tests for the VAE model and loss."""

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.models import MLPVAE, vae_loss


def test_default_first_layer_and_latent_heads():
    model = MLPVAE()
    expected_features = math.prod(FRAME_SHAPE)  # 2000
    assert model.frame_shape == FRAME_SHAPE
    assert model.input_dim == expected_features

    first_layer = model.encoder_trunk[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == expected_features
    assert model.fc_mu.out_features == 16
    assert model.fc_logvar.out_features == 16


def test_forward_with_raw_uint16_generator_frames():
    model = MLPVAE(hidden_dims=(256, 64), latent_dim=16)
    model.train()
    rng = np.random.default_rng(42)
    raw_np = rng.integers(0, ADC_MAX + 1, size=(8, *FRAME_SHAPE), dtype=np.uint16)
    x = torch.from_numpy(raw_np)

    recon, mu, logvar = model(x)
    assert recon.shape == (8, *FRAME_SHAPE)
    assert recon.dtype == torch.float32
    assert torch.all(recon >= 0.0) and torch.all(recon <= 1.0)
    assert mu.shape == (8, 16)
    assert logvar.shape == (8, 16)

    adc_recon = model.to_adc(recon)
    assert adc_recon.shape == (8, *FRAME_SHAPE)
    assert adc_recon.dtype == torch.uint16
    adc_recon_i32 = adc_recon.to(torch.int32)
    assert int(adc_recon_i32.min()) >= 0 and int(adc_recon_i32.max()) <= ADC_MAX


def test_deterministic_eval_vs_stochastic_train():
    torch.manual_seed(0)
    model = MLPVAE(hidden_dims=(64,), latent_dim=8)
    x = torch.rand(4, *FRAME_SHAPE)

    # In eval mode, forward uses mu deterministically
    model.eval()
    recon1, mu1, _ = model(x)
    recon2, mu2, _ = model(x)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(recon1, recon2)


def test_vae_loss_and_backward_pass():
    model = MLPVAE(frame_shape=(10, 8, 8), hidden_dims=(64, 32), latent_dim=6)
    model.train()
    x = torch.rand(4, 10, 8, 8)

    recon, mu, logvar = model(x)
    total_loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=0.5)

    assert total_loss.ndim == 0
    assert recon_loss.item() >= 0.0
    assert kl_loss.item() >= 0.0
    assert math.isclose(
        total_loss.item(),
        recon_loss.item() + 0.5 * kl_loss.item(),
        rel_tol=1e-5,
    )

    total_loss.backward()
    assert model.encoder_trunk[0].weight.grad is not None
    assert model.fc_mu.weight.grad is not None
    assert model.fc_logvar.weight.grad is not None


def test_kl_loss_zero_for_standard_normal():
    recon = torch.zeros(4, *FRAME_SHAPE)
    target = torch.zeros(4, *FRAME_SHAPE)
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)

    total_loss, recon_loss, kl_loss = vae_loss(recon, target, mu, logvar, beta=1.0)
    assert torch.isclose(kl_loss, torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(recon_loss, torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(total_loss, torch.tensor(0.0), atol=1e-7)
