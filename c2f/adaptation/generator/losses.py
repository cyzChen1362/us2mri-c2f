"""Losses of the MRI -> pseudo-US generator.

    total = GAN(pseudo-US vs real US)
          + lambda_nce * PatchNCE(MRI patches <-> pseudo-US patches)
          + lambda_idt * PatchNCE(real US <-> G(real US))   (identity term)

Loss weights are removed in this public release (desensitised).
"""
import torch

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def gan_loss(prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
    raise NotImplementedError(REDACTED)


def patch_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE over patches: each output patch should match the input patch at the same location."""
    raise NotImplementedError(REDACTED)
