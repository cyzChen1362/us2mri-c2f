"""Step 4 of trajectory slicing: planes sampled along the sweep.

Planes are placed at equal arc length along the sweep, between the two end
faces. At each position the plane centre and its in-plane / normal axes
(u, v, n) are interpolated from the boundary curves so that centre and
orientation vary smoothly from one plane to the next.

Output (saved as ``poses.npz``)
    t               (K,)        normalised sweep parameter in [0, 1]
    center_arc_mm   (K,)        arc length of each plane centre
    centers_LPS     (K, 3)      plane centres
    frames_uvn_LPS  (K, 3, 3)   orthonormal frame of each plane
    pixel_to_LPS    (K, 4, 4)   pixel -> physical mapping of each 2D frame

The implementation is removed in this public release (desensitised).
"""
import numpy as np

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def arc_length_parameter(curves: np.ndarray, count: int) -> np.ndarray:
    """Sweep parameters giving equal arc length between consecutive planes."""
    raise NotImplementedError(REDACTED)


def plane_frames(curves: np.ndarray, t: np.ndarray) -> dict:
    """Centres and smooth (u, v, n) frames of the planes at parameters t."""
    raise NotImplementedError(REDACTED)


def pixel_mapping(frames: dict, spacing_mm: float, shape: tuple[int, int]) -> np.ndarray:
    """(K, 4, 4) homogeneous pixel -> LPS matrices of the 2D frames."""
    raise NotImplementedError(REDACTED)


def perturb_trajectory(frames: dict, shift_mm: float, tilt_deg: float, rng: np.random.Generator) -> dict:
    """Rigid perturbation of the whole trajectory.

    Used to cut additional full MRI slices that do not coincide with the US
    frames, simulating the difference between a planned and an actual sweep.
    """
    raise NotImplementedError(REDACTED)
