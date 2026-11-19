"""Evaluate a PlanePose variant on the RESECT test split.

Metrics: point error (mean distance of US pixels under predicted vs true
pose, mm) and rotation error (geodesic angle, degrees), as reported in
results/tables/03_preliminary_planepose_resect.md.
"""
import argparse, json
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def evaluate(cfg: dict, checkpoint: str) -> dict:
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('checkpoint'); a = ap.parse_args()
    print(evaluate(json.loads(Path(a.config).read_text()), a.checkpoint))
