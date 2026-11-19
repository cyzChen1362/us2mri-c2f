"""Training / evaluation loop of the matcher.

Evaluation writes one score matrix per patient (scores.npy, 200 x 200 logits)
and the neighbourhood metrics Top-1 within ±5 / ±10 slices and the centre
error of the selected plane (see c2f/slice_matching/evaluate.py).
"""
import argparse, json
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def train(cfg: dict):
    raise NotImplementedError(REDACTED)


def evaluate_patient(model, case: str, cfg: dict):
    """Returns the (200, 200) score matrix of one patient."""
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    train(json.loads(Path(a.config).read_text()))
