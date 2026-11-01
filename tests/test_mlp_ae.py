"""Unit tests for the MLPAutoencoder model."""

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.models import MLPAutoencoder


def test_default_first_layer_matches_frame_shape():
    model = MLPAutoencoder()
    expected_features = math.prod(FRAME_SHAPE)  # 20 * 10 * 10 = 2000
    assert model.frame_shape == FRAME_SHAPE
    assert model.input_dim == expected_features

    first_layer = model.encoder[0]
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == expected_features


def test_forward_with_raw_uint16_generator_frames():
    model = MLPAutoencoder(hidden_dims=(256, 64), latent_dim=16)
    rng = np.random.default_rng(42)
    # Simulate generator / hardware recording: (T, 20, 10, 10) uint16 in 0..ADC_MAX
    raw_np = rng.integers(0, ADC_MAX + 1, size=(8, *FRAME_SHAPE), dtype=np.uint16)
    x = torch.from_numpy(raw_np)

    z = model.encode(x)
    assert z.shape == (8, 16)
    assert z.dtype == torch.float32

    recon = model(x)
    assert recon.shape == (8, *FRAME_SHAPE)
    assert recon.dtype == torch.float32
    assert torch.all(recon >= 0.0) and torch.all(recon <= 1.0)

    adc_recon = model.to_adc(recon)
    assert adc_recon.shape == (8, *FRAME_SHAPE)
    assert adc_recon.dtype == torch.uint16
    adc_recon_i32 = adc_recon.to(torch.int32)
    assert int(adc_recon_i32.min()) >= 0 and int(adc_recon_i32.max()) <= ADC_MAX


def test_forward_with_normalized_float_input():
    model = MLPAutoencoder(hidden_dims=(128,), latent_dim=8)
    x = torch.rand(4, *FRAME_SHAPE, dtype=torch.float32)

    recon = model(x)
    assert recon.shape == (4, *FRAME_SHAPE)
    assert torch.all(recon >= 0.0) and torch.all(recon <= 1.0)


def test_single_unbatched_frame():
    model = MLPAutoencoder(hidden_dims=(64,), latent_dim=10)
    x = torch.rand(*FRAME_SHAPE)

    z = model.encode(x)
    assert z.shape == (10,)

    recon = model.decode(z)
    assert recon.shape == FRAME_SHAPE


def test_custom_frame_shape_and_backward_pass():
    custom_shape = (10, 8, 8)
    model = MLPAutoencoder(frame_shape=custom_shape, hidden_dims=(128, 32), latent_dim=6)
    assert model.encoder[0].in_features == 10 * 8 * 8

    x = torch.rand(4, *custom_shape)
    recon = model(x)
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()

    assert model.encoder[0].weight.grad is not None


def test_invalid_frame_shape_raises():
    model = MLPAutoencoder()
    wrong_input = torch.zeros(4, 10, 10, 10)
    with pytest.raises(ValueError, match="Expected input with trailing frame shape"):
        model(wrong_input)

