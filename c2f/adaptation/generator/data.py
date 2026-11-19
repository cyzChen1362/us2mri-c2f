"""Unpaired data for the MRI -> pseudo-US generator.

Training patients only. The two domains are sampled independently:

    MRI domain   2.5D stacks of the local MRI around the planned scan region
                 (neighbouring slices as channels), 192 x 192
    US domain    real 2D US frames of the training patients, same canvas

No US of validation or test patients is ever read here.
The implementation is removed in this public release (desensitised).
"""
from torch.utils.data import Dataset

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


class UnpairedSliceDataset(Dataset):
    """Returns {'mri': (C, H, W) 2.5D stack, 'us': (1, H, W) real US} drawn independently."""

    def __init__(self, cases: list[str], modality: str, size: int = 192, context: int = 1):
        raise NotImplementedError(REDACTED)

    def __len__(self) -> int:
        raise NotImplementedError(REDACTED)

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(REDACTED)
