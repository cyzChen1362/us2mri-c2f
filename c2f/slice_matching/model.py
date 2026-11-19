"""Pair matcher (coarse stage, step 2: *which slice*).

For one US frame and each of the 200 candidate MRI planes of the same patient
(cropped to the predicted window), the matcher outputs a score s(i, j).
Stacking all pairs gives a 200 x 200 score matrix per patient; the top-1
candidate of each row is the coarse pose passed to the fine stage.

Module-level structure
    US frame ──── 2D encoder ──┐
                               ├─ cross-attention ─ pooling ─ pair score
    MRI window ── 2D encoder ──┘

Evolution of the design (see results/tables/matching_evolution.md):
global embedding -> pair classifier -> + cross-attention -> + wider window.

The implementation is removed in this public release (desensitised).
"""
import torch
from torch import nn

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


class CrossAttention(nn.Module):
    """Exchanges information between US and MRI feature maps."""

    def __init__(self, dim: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def forward(self, us_tokens, mri_tokens, pos):
        raise NotImplementedError(REDACTED)


class PairMatcher(nn.Module):
    """Scores (US frame, MRI candidate window) pairs."""

    def __init__(self, backbone: str = 'resnet50', pretrained: str | None = None):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def forward(self, us: torch.Tensor, mri: torch.Tensor) -> torch.Tensor:
        """us: (B, 3, H, W), mri: (B, 3, H, W) -> logits (B,)"""
        raise NotImplementedError(REDACTED)

    @torch.no_grad()
    def score_matrix(self, us_stack, mri_stack, chunk: int = 16) -> torch.Tensor:
        """All-pairs scores for one patient: (N_us, N_mri) logits."""
        raise NotImplementedError(REDACTED)
