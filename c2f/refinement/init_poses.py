"""Initial poses for training the refiner.

Training patients have no held-out matching output, so initial poses are drawn
from the error distribution that the matching stage actually produces on
held-out patients (translation and rotation residuals), applied to the true
pose of each training slice. Sampling details are removed in this release.
"""
REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def empirical_errors(matching_outputs):
    raise NotImplementedError(REDACTED)


def sample_initial_pose(target_pose, error_pool, rng):
    raise NotImplementedError(REDACTED)
