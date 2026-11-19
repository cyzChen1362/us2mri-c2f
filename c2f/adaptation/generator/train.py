"""Train the MRI -> pseudo-US generator on training patients.

    python -m c2f.adaptation.generator.train configs/adaptation/generator.json

One generator per MRI branch (ceT1, FLAIR). The trained generator is then
frozen and used by ``c2f/adaptation/generate.py`` for every validation /
test patient, whose real US it never sees.
"""
import argparse, json
from pathlib import Path

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def train(cfg: dict):
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('config'); a = ap.parse_args()
    train(json.loads(Path(a.config).read_text()))
