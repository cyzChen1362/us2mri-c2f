"""Recurrent pose refiner (fine stage, recurrent update idea from DreamReg).

Starts from the coarse pose given by slice matching and refines the full
6-DoF rigid pose in a fixed number of steps (5). At every step the MRI is
re-sliced at the current pose and compared with the US frame.

Module-level structure
    US frame ─────────── 2D encoder ──┐
    MRI slice @ pose_k ─ 2D encoder ──┼─ fusion ─ recurrent cell ─ Δpose ─> pose_{k+1}
    local MRI volume ─── 3D encoder ──┘        (state kept across steps)
    pose_{k+1} -> re-slice MRI -> next step

The implementation is removed in this public release (desensitised).
"""
import torch
from torch import nn

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


class RecurrentRefiner(nn.Module):
    def __init__(self, steps: int = 5, hidden: int = 512):
        super().__init__()
        raise NotImplementedError(REDACTED)

    def step(self, us_feat, vol_feat, mri_slice, pose, state):
        """One update: returns (delta_pose, new_state)."""
        raise NotImplementedError(REDACTED)

    def forward(self, us, volume, init_pose, render) -> list[torch.Tensor]:
        """Returns the list of poses [pose_0, ..., pose_5]; ``render`` re-slices the MRI."""
        raise NotImplementedError(REDACTED)
