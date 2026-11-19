"""Rigid pose from 2D–3D correspondences (PlanePose v2 / v5).

Each sampled US pixel has a known in-plane position and a predicted MRI
coordinate; a rigid transform is estimated with RANSAC and refined on the
inliers (Kabsch).
The implementation is removed in this public release (desensitised).
"""
import numpy as np

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares rigid transform (4, 4) mapping source -> target."""
    raise NotImplementedError(REDACTED)


def ransac_pose(plane_points: np.ndarray, predicted: np.ndarray, threshold_mm: float, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pose (4, 4), inlier mask)."""
    raise NotImplementedError(REDACTED)
