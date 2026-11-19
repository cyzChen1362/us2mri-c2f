"""PlanePose networks (preliminary study, RESECT).

    US frame ─> 2D encoder ─┐
                            ├─ fusion ─┬─> v1 / v3: 6-DoF pose head (translation + rotation)
    MRI volume ─> 3D encoder┘          └─> v2: per-pixel 3D correspondences (solved by RANSAC)

v4–v6 add a local position prior (``prior.py``) as an extra input channel.
Results (point error 26–48 mm, rotation error > 110°) showed that direct
regression does not generalise; see results/tables/03_preliminary_planepose_resect.md.
The implementation is removed in this public release (desensitised).
"""
import torch
from torch import nn

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


class PlanePoseRegressor(nn.Module):
    """v1 / v3 / v4 / v6: regresses translation (3) and rotation (6D representation)."""

    def __init__(self, width: int = 32, use_prior: bool = False):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def forward(self, us: torch.Tensor, mri: torch.Tensor, prior: torch.Tensor | None = None) -> dict:
        raise NotImplementedError(REDACTED)


class PlaneCorrespondenceNet(nn.Module):
    """v2 / v5: predicts the MRI coordinates of sampled US pixels."""

    def __init__(self, width: int = 32, use_prior: bool = False):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def forward(self, us: torch.Tensor, mri: torch.Tensor, prior: torch.Tensor | None = None) -> dict:
        raise NotImplementedError(REDACTED)


def build_model(variant: str) -> nn.Module:
    """'v1' ... 'v6' -> network (see the table in README.md)."""
    raise NotImplementedError(REDACTED)
