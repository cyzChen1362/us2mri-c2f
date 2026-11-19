"""Centered linear plane correlation. Spatial units are 4 mm feature cells."""
import math
import torch
from scipy.fft import next_fast_len
from torch.nn import functional as F

def lattice(shape, device):
    z, y, x = torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in shape], indexing='ij')
    return torch.stack([x, y, z], -1)

def rotation_grids(shape, rotations):
    """Full bounding cube, not a clipped rotation into the original box.

    R columns are US x/y/normal axes in MRI-native LPS. Coordinates are feature
    voxel centers; both directions use align_corners=False.
    """
    device = rotations.device
    xyz = torch.tensor(shape[::-1], device=device, dtype=torch.float32)
    center = (xyz - 1) / 2
    side = int(math.ceil(math.sqrt(sum(((n - 1) ** 2 for n in shape))))) + 3
    if side % 2 == 0:
        side += 1
    local = lattice((side, side, side), device) - (side - 1) / 2
    world = lattice(shape, device) - center
    forward = []
    back = []
    for R in rotations:
        forward.append(2 * (local @ R.T + center + 0.5) / xyz - 1)
        back.append(2 * (world @ R + (side - 1) / 2 + 0.5) / side - 1)
    return (torch.stack(forward), torch.stack(back))

def _linear_corr(x, k, fft_shape=None):
    """x C,D,H,W; k N,C,kh,kw. Odd centered kernels, linear zero padding."""
    h, w = x.shape[-2:]
    kh, kw = k.shape[-2:]
    assert kh % 2 == 1 and kw % 2 == 1
    shape = fft_shape or (next_fast_len(h + kh - 1, real=True), next_fast_len(w + kw - 1, real=True))
    X = torch.fft.rfft2(x.float(), s=shape)
    K = torch.fft.rfft2(k.float().flip((-2, -1)), s=shape)
    y = torch.fft.irfft2(X[None] * K[:, :, None], s=shape)
    return y[..., kh // 2:kh // 2 + h, kw // 2:kw // 2 + w]

def plane_correlation(search, template, mask, method='fft', fused_normalization=False):
    """Masked per-channel cosine of whole spatial templates, not point matching.

    search C,D,H,W, template N,C,kh,kw, mask N,1,kh,kw. Mask is a fractional
    support area per feature cell. No query pose enters this operation.
    """

    def correlate(x, k):
        if method == 'fft':
            return _linear_corr(x, k)
        n, c, kh, kw = k.shape
        weight = k.permute(1, 0, 2, 4).reshape(c * n, 1, 1, kh, kw)
        out = F.conv3d(x[None].float(), weight.float(), padding=(0, kh // 2, kw // 2), groups=c)
        return out.reshape(c, n, *x.shape[1:]).transpose(0, 1)
    with torch.autocast(search.device.type, enabled=False):
        s = search.float()
        u = template.float()
        m = mask.float().expand_as(u)
        numerator = correlate(s, u * m)
        energy = correlate(s.square(), m)
        query_energy = (u.square() * m).sum((-2, -1))[:, :, None, None, None]
        if fused_normalization:
            from .execution import normalize_correlation
            return normalize_correlation(numerator, energy, query_energy)
        return numerator / (energy.clamp_min(0) * query_energy).clamp_min(1e-06).sqrt()
