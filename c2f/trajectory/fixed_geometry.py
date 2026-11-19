"""Comparison only: a fixed-geometry sweep model.

Re-implementation of the idea of an industrial patent (CN115516504A,
"Ultrasound data segmentation"), which describes the acquisition with one
fixed scan shape. Used to compare against the geometry-derived trajectory:
on freehand sweeps the fixed model gives small, fragmented or oblique
slices near the ends of the sweep.

The implementation is removed in this public release (desensitised).
"""
import numpy as np

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def fit_fixed_geometry(surface: np.ndarray) -> dict:
    """Fit the parameters of the fixed scan shape to the US support."""
    raise NotImplementedError(REDACTED)


def sample_fixed_planes(params: dict, count: int) -> dict:
    """Planes of the fixed scan shape, in the same format as ``sampling.plane_frames``."""
    raise NotImplementedError(REDACTED)
