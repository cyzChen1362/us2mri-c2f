"""MRI -> pseudo-US generation for one patient.

Uses an unpaired image-to-image translation model (CUT family, 2.5D input)
trained on training patients (training code: c2f/adaptation/generator). The input
is the local MRI of the planned scan region. Generation settings are removed in
this public release.
"""
REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def load_generator(checkpoint: str, device: str = 'cuda'):
    raise NotImplementedError(REDACTED)


def generate_patient(generator, mri_windows, out_dir):
    """Writes one pseudo-US image per local MRI slice of the planned scan region."""
    raise NotImplementedError(REDACTED)
