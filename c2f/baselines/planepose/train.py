"""Train one PlanePose variant on RESECT.

    python -m c2f.baselines.planepose.train configs/baselines/planepose/planepose.json
"""
import argparse, json
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def train(cfg: dict):
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    train(json.loads(Path(a.config).read_text()))
