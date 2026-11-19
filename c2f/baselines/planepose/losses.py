"""Losses of the PlanePose variants.

    v1 / v4   pose loss: translation L1 + rotation geodesic
    v2 / v5   correspondence loss: L1 between predicted and true MRI coordinates
    v3 / v6   in-plane point-distance loss: mean distance of US pixels mapped
              by the predicted and the true pose (all 6 DoF in mm)

Loss weights are removed in this public release (desensitised).
"""
import torch

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def pose_loss(pred: dict, pose: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(REDACTED)


def correspondence_loss(pred: dict, target_points: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(REDACTED)


def point_distance_loss(pred: dict, pose: torch.Tensor, plane_points: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(REDACTED)
