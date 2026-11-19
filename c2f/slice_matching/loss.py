"""Matching objective.

Each US frame is paired with MRI candidates near its true slice (positives)
and far from it (negatives); candidates in between are ignored because
neighbouring planes are nearly identical. The matcher is trained as a pair
classifier on these groups.

Exact radii, weighting and any auxiliary terms are removed in this release.
"""
REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def matching_loss(logits, labels, weights=None):
    raise NotImplementedError(REDACTED)
