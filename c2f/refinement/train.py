"""Training / evaluation loop of the recurrent refiner.

Writes per-epoch metrics (case mean of centre / rotation / plane-point error
at each of the 6 time points: initial + 5 updates).
"""
import argparse, json
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def train(cfg: dict):
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    train(json.loads(Path(a.config).read_text()))
