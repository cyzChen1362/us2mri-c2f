"""Re-slicing and export of one patient.

For every sampled plane:
    * the regular fan-shaped 2D US frame (US support applied),
    * the full-FOV MRI slice on the same plane (through the dataset's
      US -> MRI registration; ceT1 and FLAIR branches),
    * the 6-DoF pose and the pixel -> physical mapping,
    * the geometric ground-truth window of the US field of view inside the
      MRI slice (used to train window localization).

Files written per patient
    poses.npz         plane poses (see ``sampling.py``)
    trajectory.json   caps, boundary-curve confidence / residual, coverage
    planes.json       per-plane geometry of the 2D grids
    US / MRI slices   one .npy per plane
    complete.json     bookkeeping (counts, source files, code hash)

Re-slicing itself uses ``c2f/common/reslicing.py``.
The implementation is removed in this public release (desensitised).
"""
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def reslice_case(us_path: str, mri_paths: dict, us_to_mri: dict, poses: dict, out_dir: Path) -> dict:
    """Write the paired US / MRI slices of one patient; returns the bookkeeping record."""
    raise NotImplementedError(REDACTED)


def ground_truth_windows(poses: dict, us_support, mri_grid) -> list[list[float]]:
    """Window of the US field of view inside each MRI slice, xyxy in MRI pixels."""
    raise NotImplementedError(REDACTED)
