"""Step 3 of trajectory slicing: boundary curves between the two end faces.

The edges ("ridges") of the scanned region connect corresponding corners of
the start and end caps. Tracing them on the support surface and fitting a
smoothing spline to each gives a set of curves that describe how the fan
moved during the sweep. Each curve carries a confidence and the spline
residual, which are stored in ``trajectory.json`` for quality control.

Output
    curves      (R, S, 3) R boundary curves, each resampled to S points
    confidence  (R,)       per-curve confidence in [0, 1]
    residual    (R,)       95th-percentile spline residual in mm

The implementation is removed in this public release (desensitised).
"""
import numpy as np

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def match_cap_corners(start_cap: dict, end_cap: dict) -> list[tuple[int, int]]:
    """Pairs of corresponding corners on the two caps."""
    raise NotImplementedError(REDACTED)


def trace_ridge(surface: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Path along the support surface from one cap corner to its partner."""
    raise NotImplementedError(REDACTED)


def fit_curves(surface: np.ndarray, start_cap: dict, end_cap: dict, samples: int) -> dict:
    """All boundary curves with confidence and spline residual."""
    raise NotImplementedError(REDACTED)
