"""Basic fully-connected (MLP) autoencoder for tactile sensor frames."""

import math
from typing import Sequence

import torch
import torch.nn as nn

from tactile.data_format import ADC_MAX, FRAME_SHAPE


class MLPAutoencoder(nn.Module):
    """Fully-connected MLP autoencoder for single tactile sensor frames.

    By default, the first encoder layer takes a flattened frame of shape
    ``FRAME_SHAPE`` ``(20, 10, 10)`` -> ``2000`` sensor values, compresses it
    through ``hidden_dims`` into a ``latent_dim`` bottleneck, and symmetrically
    decodes it back to ``(20, 10, 10)`` in ``[0, 1]``.

    Args:
        frame_shape: Spatial dimensions of a single sensor frame. Defaults to
            ``FRAME_SHAPE`` ``(20, 10, 10)``.
        hidden_dims: Sizes of intermediate fully-connected layers in the
            encoder (reversed in the decoder).
        latent_dim: Size of the bottleneck representation.
        adc_max: Maximum raw ADC sensor value used when normalising raw inputs.
            Defaults to ``ADC_MAX`` (4095).
    """

    def __init__(
        self,
        frame_shape: tuple[int, ...] = FRAME_SHAPE,
        hidden_dims: Sequence[int] = (512, 128),
        latent_dim: int = 16,
        adc_max: int = ADC_MAX,
    ) -> None:
        super().__init__()
        if len(frame_shape) == 0 or any(d <= 0 for d in frame_shape):
            raise ValueError(f"frame_shape must contain positive dimensions, got {frame_shape}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims must all be positive, got {hidden_dims}")

        self.frame_shape = tuple(frame_shape)
        self.input_dim = math.prod(self.frame_shape)
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = int(latent_dim)
        self.adc_max = int(adc_max)

        # Build encoder: input_dim -> hidden_dims[0] -> ... -> latent_dim
        encoder_layers: list[nn.Module] = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build symmetric decoder: latent_dim -> ... -> hidden_dims[0] -> input_dim -> Sigmoid
        decoder_layers: list[nn.Module] = []
        in_dim = self.latent_dim
        for h_dim in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, self.input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw sensor frames (e.g. uint16 in 0..ADC_MAX) or floats to [0, 1] float32.

        If ``x`` has an integer dtype or contains values above 1.0, it is scaled
        by ``1 / self.adc_max`` and clamped to ``[0.0, 1.0]``. Pre-normalised
        floating-point tensors in ``[0.0, 1.0]`` are simply cast to ``float32``.
        """
        is_integer_input = not x.is_floating_point()
        x_float = x.to(dtype=torch.float32)
        if is_integer_input or (x_float.numel() > 0 and x_float.max() > 1.0):
            x_float = x_float / self.adc_max
        return torch.clamp(x_float, 0.0, 1.0)

    def to_adc(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised [0, 1] predictions back to uint16 ADC counts in 0..ADC_MAX."""
        scaled = torch.clamp(torch.round(x_norm * self.adc_max), 0, self.adc_max)
        return scaled.to(dtype=torch.uint16)

    def _validate_frame_shape(self, x: torch.Tensor) -> tuple[int, ...]:
        """Verify trailing dimensions match ``frame_shape`` and return leading batch dims."""
        n_spatial = len(self.frame_shape)
        if x.ndim < n_spatial or tuple(x.shape[-n_spatial:]) != self.frame_shape:
            raise ValueError(
                f"Expected input with trailing frame shape {self.frame_shape}, "
                f"got tensor of shape {tuple(x.shape)}"
            )
        return tuple(x.shape[:-n_spatial])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sensor frame(s) ``(*batch_dims, *frame_shape)`` into ``(*batch_dims, latent_dim)``."""
        batch_dims = self._validate_frame_shape(x)
        x_norm = self.normalize_input(x)
        flat = x_norm.reshape(*batch_dims, self.input_dim)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector(s) ``(*batch_dims, latent_dim)`` to ``(*batch_dims, *frame_shape)`` in [0, 1]."""
        if z.ndim < 1 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent tensor with last dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
        batch_dims = tuple(z.shape[:-1])
        flat_recon = self.decoder(z)
        return flat_recon.reshape(*batch_dims, *self.frame_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input sensor frame(s), returning normalised predictions in [0, 1]."""
        z = self.encode(x)
        return self.decode(z)

