"""Local position prior used by PlanePose v4–v6.

A coarse 3D map of where US frames of this sweep can lie inside the MRI
crop, given to the network as an extra channel so that it only has to
resolve the position locally.
The implementation is removed in this public release (desensitised).
"""
import numpy as np

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def position_prior(mri_shape: tuple[int, int, int], sweep_poses: np.ndarray, sigma_mm: float) -> np.ndarray:
    raise NotImplementedError(REDACTED)
