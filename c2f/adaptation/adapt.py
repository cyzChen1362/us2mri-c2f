"""Short per-patient fine-tuning of the matcher on pseudo-US / MRI pairs.

Starts from the generic matcher (c2f/slice_matching), trains for a few epochs
on the pairs of one patient only, then is evaluated on that patient's real US.
Schedule, learning rates and which layers are tuned are removed in this release.
"""
REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def adapt(matcher, pairs, cfg: dict):
    raise NotImplementedError(REDACTED)
