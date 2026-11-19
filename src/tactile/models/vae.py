"""Variational Autoencoder (Beta-VAE) for tactile sensor frames."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile.data_format import ADC_MAX, FRAME_SHAPE

LOGVAR_MIN = -10.0
LOGVAR_MAX = 10.0


class MLPVAE(nn.Module):
    """Fully-connected Variational Autoencoder for single tactile sensor frames.

    Shares the same encoder trunk and symmetric decoder architecture as
    the standard MLP AE for fair benchmarking,
    splitting into two linear heads (``fc_mu`` and ``fc_logvar``) at the
    bottleneck, one for generating each parameter (mu and sigma) for the distribution of the latent space.

    Args:
        frame_shape: Spatial dimensions of a single sensor frame. Defaults to
            ``FRAME_SHAPE`` ``(20, 10, 10)``.
        hidden_dims: Sizes of intermediate fully-connected layers in the
            encoder trunk (reversed in the decoder). Defaults to ``(512, 128)``.
        latent_dim: Size of the Gaussian latent bottleneck (``mu`` and ``logvar``).
            Defaults to ``16``.
        adc_max: Maximum raw ADC sensor count used when normalising raw inputs.
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
        if len(hidden_dims) == 0 or any(h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims must be a non-empty sequence of positive ints, got {hidden_dims}")

        self.frame_shape = tuple(frame_shape)
        self.input_dim = math.prod(self.frame_shape)
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = int(latent_dim)
        self.adc_max = int(adc_max)

        # Shared encoder trunk: input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1]
        trunk_layers: list[nn.Module] = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h_dim))
            trunk_layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder_trunk = nn.Sequential(*trunk_layers)

        # Dual heads for posterior mean (mu) and log-variance (logvar)
        self.fc_mu = nn.Linear(in_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, self.latent_dim)

        # Symmetric decoder: latent_dim -> ... -> hidden_dims[0] -> input_dim -> Sigmoid
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
        """Convert raw sensor frames (uint16 in 0..ADC_MAX) or floats to [0, 1] float32."""
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

    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior parameters ``(mu, logvar)`` of shape ``(*batch_dims, latent_dim)``.

        ``logvar`` is clamped to ``[LOGVAR_MIN, LOGVAR_MAX]`` to prevent
        numerical overflow in ``exp(logvar)``.
        """
        batch_dims = self._validate_frame_shape(x)
        x_norm = self.normalize_input(x)
        flat = x_norm.reshape(*batch_dims, self.input_dim)
        features = self.encoder_trunk(flat)
        mu = self.fc_mu(features)
        logvar = torch.clamp(self.fc_logvar(features), LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        sample: bool | None = None,
    ) -> torch.Tensor:
        """Sample ``z = mu + eps * std`` using the reparameterisation trick.

        When ``sample`` is ``False`` (or ``None`` while ``self.training`` is
        ``False``), returns ``mu`` deterministically for jitter-free inference.
        """
        use_sampling = self.training if sample is None else sample
        if not use_sampling:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a full VAE pass and return ``(recon, mu, logvar)``.

        Args:
            x: Input sensor frame(s) of shape ``(*batch_dims, *frame_shape)``.
            sample: Override stochastic sampling. If ``None``, samples when
                ``self.training`` is ``True`` and uses ``mu`` when ``False``.

        Returns:
            Tuple ``(recon, mu, logvar)`` where ``recon`` has shape
            ``(*batch_dims, *frame_shape)`` in ``[0, 1]``, and ``mu`` / ``logvar``
            have shape ``(*batch_dims, latent_dim)``.
        """
        mu, logvar = self.encode_dist(x)
        z = self.reparameterize(mu, logvar, sample=sample)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the beta-VAE loss normalised per sensor element.

    Both the MSE reconstruction loss and the analytical KL divergence
    are normalised by the number of sensors per frame D and averaged 
    over the batch.

    Args:
        recon: Reconstructed frames in ``[0, 1]`` of shape ``(B, *frame_shape)``.
        target: Ground-truth normalised frames in ``[0, 1]`` of shape ``(B, *frame_shape)``.
        mu: Posterior mean of shape ``(B, latent_dim)``.
        logvar: Posterior log-variance of shape ``(B, latent_dim)``.
        beta: Weight applied to the normalised KL divergence term.

    Returns:
        Tuple ``(total_loss, recon_loss, kl_loss)`` of scalar tensors.
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # Analytical KL per sample summed over latent_dim: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    # Normalise by number of sensors per frame (D) to match MSE reduction="mean" scale
    n_sensors = max(1, recon[0].numel() if recon.ndim > 1 else recon.numel())
    kl_loss = kl_per_sample.mean() / n_sensors

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
