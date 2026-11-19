"""Build the trajectory-sliced dataset for all patients.

    python -m c2f.trajectory.build configs/trajectory/default.json

Per patient: support -> end faces -> boundary curves -> sampled planes ->
re-slicing and export. Optional: additional full MRI slices along rigidly
perturbed copies of the trajectory (``perturbations`` in the config).
"""
import argparse, json
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def build_case(case: str, cfg: dict) -> dict:
    raise NotImplementedError(REDACTED)


def build(cfg: dict):
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    build(json.loads(Path(a.config).read_text()))
