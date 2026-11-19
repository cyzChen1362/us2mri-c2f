"""RESECT data for the PlanePose preliminary study.

RESECT provides pre-operative T1 / FLAIR MRI and intra-operative 3D US with
landmark pairs. US frames are sliced from the 3D US; each frame comes with
its pose in the MRI. The MRI volume is cropped around the US coverage and
resampled to a fixed grid before entering the network.

Per sample: {'us': (1, H, W), 'mri': (1, D, H, W), 'pose': (4, 4), 'points': (P, 3)}
where ``points`` are physical coordinates of sampled US pixels (for the
point-distance loss of v3).
The implementation is removed in this public release (desensitised).
"""
from torch.utils.data import Dataset

REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


class ResectPlaneDataset(Dataset):
    def __init__(self, cases: list[str], split: str, prior: bool = False):
        raise NotImplementedError(REDACTED)

    def __len__(self) -> int:
        raise NotImplementedError(REDACTED)

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(REDACTED)
