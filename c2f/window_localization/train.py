"""Training / evaluation loop for window localization.

Metrics reported (see results/): 2D IoU between predicted and true window,
and window-centre error in millimetres, per split.

Usage (illustrative): python -m c2f.window_localization.train configs/window_localization/ceT1.json
"""
import argparse, json
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def build(cfg: dict):
    """Model, optimiser, data loaders (c2f.common.fullvision provides the inputs)."""
    raise NotImplementedError(REDACTED)


def train(cfg: dict):
    raise NotImplementedError(REDACTED)


def evaluate(model, loader) -> dict:
    """Returns {'iou': ..., 'centre_mm': ...} averaged over slices of each patient."""
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    train(json.loads(Path(a.config).read_text()))
