"""Per-step supervision of the refiner.

Every intermediate pose is supervised so that the error decreases step by
step, and the final step targets the true pose. Metrics: centre error (mm),
rotation error (deg) and plane-point error over valid US pixels (mm).
The exact loss terms are removed in this public release.
"""
REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def refinement_loss(poses, target_pose):
    raise NotImplementedError(REDACTED)


def pose_metrics(pose, target_pose, us_pixels_mm) -> dict:
    """{'centre_mm', 'rotation_deg', 'plane_point_mm'}"""
    raise NotImplementedError(REDACTED)
