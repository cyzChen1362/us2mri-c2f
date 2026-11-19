"""Step 2 of trajectory slicing: the two end faces of the sweep.

A freehand sweep starts and stops at two roughly planar faces of the scanned
region ("caps"). Each cap is described by a fitted plane and the polygon of
support points lying on it. The caps fix where the sampled planes begin and
end, and give the initial and final probe orientation.

Output per cap
    {'center': (3,), 'normal': (3,), 'polygon': (M, 3), 'residual_mm': float}

The implementation is removed in this public release (desensitised).
"""
import numpy as np

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def candidate_faces(surface: np.ndarray, axes: np.ndarray) -> list[dict]:
    """Planar patches of the support surface that could be a start or end face."""
    raise NotImplementedError(REDACTED)


def fit_end_faces(surface: np.ndarray, axes: np.ndarray) -> tuple[dict, dict]:
    """Select and fit the start and end caps of the sweep."""
    raise NotImplementedError(REDACTED)
