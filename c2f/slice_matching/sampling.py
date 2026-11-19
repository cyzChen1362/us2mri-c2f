"""Candidate sampling for training the matcher.

Builds, for every US query, a mixture of positive, ignored and negative MRI
candidates from the same patient, including hard negatives taken from
look-alike regions of the volume.

The sampling strategy is removed in this public release (desensitised).
"""
REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def sample_candidates(query_index: int, n_candidates: int, cfg: dict):
    raise NotImplementedError(REDACTED)


def hard_negatives(scores, query_index: int, cfg: dict):
    raise NotImplementedError(REDACTED)
