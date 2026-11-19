"""Window localization network (coarse stage, step 1: *where to look*).

A US frame covers a few centimetres, while an MRI slice covers the whole head.
Before matching, we predict for every (US frame, MRI candidate slice) pair the
2D window of the MRI slice that the US field of view corresponds to.

Module-level structure
    US frame ─┐                         ┌─> centre heat-map / offsets
              ├─ 2D encoders ─ pyramid ─┤
    MRI slice ┘   (pretrained ResNet)   └─> window size
              └──── US–MRI feature interaction ────┘

* Backbone: ImageNet / RadImageNet-pretrained ResNet-50 (see c2f/common/radimagenet.py).
* Output: one axis-aligned window per MRI slice, in normalised image coordinates.

The implementation is removed in this public release (desensitised).
"""
import torch
from torch import nn

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


class WindowLocalizer(nn.Module):
    """Predicts the US field-of-view window inside a full-FOV MRI slice."""

    def __init__(self, backbone: str = 'resnet50', pretrained: str | None = None, width: int = 256):
        super().__init__()
        # US encoder, MRI encoder, multi-scale pyramid, cross-modal interaction, box head
        raise NotImplementedError(REDACTED)

    def forward(self, us: torch.Tensor, mri: torch.Tensor) -> dict:
        """us, mri: (B, 3, H, W) -> {'heatmap': (B, 1, h, w), 'box': (B, 4) xyxy in [0, 1]}"""
        raise NotImplementedError(REDACTED)

    @torch.no_grad()
    def predict_window(self, us: torch.Tensor, mri: torch.Tensor) -> torch.Tensor:
        """Decode the heat-map and size head into one window per slice."""
        raise NotImplementedError(REDACTED)
