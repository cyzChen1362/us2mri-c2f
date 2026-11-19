"""Step 1 of trajectory slicing: the US support and its surface.

The reconstructed 3D US volume is mostly empty; the scanned region is the
largest connected non-zero component. Everything downstream (end faces,
boundary curves, sampled planes) is derived from the shape of this region
only, never from the US intensities, so the same procedure applies to a
planned scan region defined before surgery.

Outputs
    support mask   uint8 volume on the US grid (1 = scanned region)
    surface points (N, 3) physical LPS coordinates of the support boundary
    principal axes 3 x 3, used to orient the sweep

The implementation is removed in this public release (desensitised).
"""
import numpy as np
import SimpleITK as sitk

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def extract_support(us: sitk.Image) -> sitk.Image:
    """Largest connected non-zero component of the US volume (holes filled)."""
    raise NotImplementedError(REDACTED)


def support_surface(support: sitk.Image) -> np.ndarray:
    """Boundary voxels of the support in physical coordinates, (N, 3)."""
    raise NotImplementedError(REDACTED)


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid (3,) and principal axes (3, 3) of a point cloud."""
    raise NotImplementedError(REDACTED)
