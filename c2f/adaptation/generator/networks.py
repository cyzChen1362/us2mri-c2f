"""Networks of the MRI -> pseudo-US generator (CUT family).

    MRI 2.5D stack ─> ResNet generator (encoder · residual blocks · decoder) ─> pseudo-US
                          │ encoder features at several layers
                          └─> MLP projection heads ─> patch embeddings for PatchNCE
    real / pseudo-US ─> PatchGAN discriminator

Reference: Park et al., "Contrastive learning for unpaired image-to-image
translation", ECCV 2020. The input layer takes the 2.5D MRI stack instead
of a single slice.
The implementation is removed in this public release (desensitised).
"""
import torch
from torch import nn

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


class ResnetGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, width: int = 64, blocks: int = 9):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def forward(self, x: torch.Tensor, layers: list[int] | None = None):
        """Returns the pseudo-US, or the encoder features at ``layers`` when given."""
        raise NotImplementedError(REDACTED)


class PatchSampleMLP(nn.Module):
    """Samples patch locations from encoder features and projects them for PatchNCE."""

    def __init__(self, channels: list[int], dim: int = 256):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def forward(self, features: list[torch.Tensor], patch_ids=None, num_patches: int = 256):
        raise NotImplementedError(REDACTED)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, width: int = 64, layers: int = 3):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(REDACTED)
