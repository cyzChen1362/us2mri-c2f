# Trajectory slicing (data construction)

Estimates a plausible freehand scan trajectory from each reconstructed 3D US
volume and re-slices it into 200 regular fan-shaped 2D frames, each with its
matching MRI plane and 6-DoF pose label:

1. extract the US support (largest connected non-zero region) and its surface;
2. fit the two end faces where the sweep starts and stops;
3. trace boundary curves between the end faces;
4. sample 200 planes along arc length with smooth centre and orientation.

**Status:** skeleton (desensitised): `support.py` → `end_faces.py` → `boundary_curves.py` →
`sampling.py` → `export.py`, driven by `build.py`; `fixed_geometry.py` is the
fixed-shape comparison. Re-slicing utilities are in `c2f/common/reslicing.py`.
