"""Patient-specific adaptation pipeline (inspired by xvr, Nature 2026).

The pre-operative MRI of a patient is available before surgery, so the generic
matcher can be specialised to that patient:

    1. prepare   - collect this patient's MRI candidate windows
    2. generate  - translate them into pseudo-US with a US generator that was
                   trained on *training patients only* (never sees real US of
                   validation / test patients)
    3. adapt     - fine-tune the matcher for a few epochs on the
                   (pseudo-US, MRI) pairs, which are paired by construction
    4. evaluate  - score the patient's *real* US frames before and after

Usage (illustrative): python -m c2f.adaptation.pipeline ceT1 <patient_id>
"""
import sys

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def run_patient(modality: str, case: str, cfg: dict):
    raise NotImplementedError(REDACTED)


if __name__ == '__main__':
    run_patient(sys.argv[1], sys.argv[2], {})
