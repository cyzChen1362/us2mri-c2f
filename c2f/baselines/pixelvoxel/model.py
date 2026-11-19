"""MASR-inspired dimensional adaptation, not an official MASR reproduction."""
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class UNet(nn.Module):

    def __init__(self, dim, base=19, thin=False):
        super().__init__()
        C = getattr(nn, f'Conv{dim}d')
        self.dim = dim
        self.pool_size = (1, 2, 2) if thin else 2

        def block(i, o):
            return nn.Sequential(C(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.LeakyReLU(0.07), C(o, o, 4, padding=1), nn.GroupNorm(5, o), nn.LeakyReLU(0.087))
        self.enc = nn.ModuleList([block(1, base), block(base, base * 2), block(base * 2, base * 5)])
        self.dec = nn.ModuleList([block(base * 6, base * 2), block(base * 3, base)])
        self.out = C(base, 10, 1)

    def forward(self, x):
        levels = []
        pool = getattr(F, f'avg_pool{self.dim}d')
        for j, b in enumerate(self.enc):
            if j:
                x = pool(x, self.pool_size)
            x = checkpoint(b, x, use_reentrant=False) if self.training else b(x)
            levels.append(x)
        for b, skip in zip(self.dec, reversed(levels[:-1])):
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=False)
            x = torch.cat([x, skip], 1)
            x = checkpoint(b, x, use_reentrant=False) if self.training else b(x)
        return self.out(x)

class DNS(nn.Module):
    """Neighbour-neighbour squared differences, no center or opposite pairs.
    3D: 12 pairs per radius; 2D adaptation: 4 pairs per radius.
    Streaming channel projection avoids materializing [B,C,24,D,H,W].
    """

    def __init__(self, dim, channels=8, radii=(1, 2)):
        super().__init__()
        self.dim = dim
        self.radii = radii
        offsets = []
        for axis in range(dim):
            for sign in [-1, 1]:
                a = [0] * dim
                a[axis] = sign
                offsets.append(a)
        self.pairs = [(a, b) for a, b in itertools.combinations(offsets, 2) if sum(((x - y) ** 2 for x, y in zip(a, b))) == 2]
        self.projection = nn.Parameter(torch.ones(channels) / channels)

    def shift(self, h, offset, r):
        padded = F.pad(h, [r, r] * self.dim, mode='replicate')
        return padded[(slice(None), slice(None)) + tuple((slice(r + v * r, r + v * r + n) for v, n in zip(offset, h.shape[2:])))]

    def forward(self, h):
        outputs = []
        for r in self.radii:
            variance = 0
            for a, b in self.pairs:
                variance = variance + (self.shift(h, a, r) - self.shift(h, b, r)).square()
            variance = (variance / len(self.pairs)).clamp_min(6.7e-07)
            for a, b in self.pairs:
                s = torch.exp(-(self.shift(h, a, r) - self.shift(h, b, r)).square() / variance)
                outputs.append((s * self.projection.to(s.dtype).view(1, -1, *[1] * self.dim)).sum(1, keepdim=True))
        return torch.cat(outputs, 1)

class Branch(nn.Module):

    def __init__(self, dim, base=20, descriptor=32):
        super().__init__()
        self.unet = UNet(dim, base)
        self.dns = DNS(dim, radii=(2, 4) if dim == 2 else (1, 2))
        C = getattr(nn, f'Conv{dim}d')
        channels = 10 if dim == 2 else 22
        self.squeeze = nn.Sequential(C(channels, 32, 2, padding=1), nn.LeakyReLU(0.11), C(32, descriptor, 1))

    def forward(self, x):
        h = self.unet(x)
        s = checkpoint(self.dns, h, use_reentrant=False) if self.training else self.dns(h)
        return F.normalize(self.squeeze(s), dim=1, eps=9.4e-07)

class PixelVoxel(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        options = (config or {}).get('model', {})
        self.us = ThinUSBranch() if options.get('us_input') == 'adjacent_stack' else Branch(2)
        self.mri = Branch(3)
        if options.get('mri_conv_depth_chunk'):
            from .memory import replace_convolutions
            replace_convolutions(self.mri, int(options['mri_conv_depth_chunk']))

    def forward(self, us, mri):
        return (self.us(us), self.mri(mri))

def sample(feature, xyz):
    """Direct nearest voxel/pixel gather. No interpolation or boundary clamping."""
    index = xyz.round().long()
    size = torch.tensor(feature.shape[2:][::-1], device=index.device)
    if not bool(((index >= 0) & (index < size)).all()):
        raise ValueError('Descriptor index outside grid')
    flat = index[:, 0] + size[0] * index[:, 1]
    if index.shape[1] == 3:
        flat = flat + size[0] * size[1] * index[:, 2]
    return F.normalize(feature[0].flatten(1)[:, flat].float().T, dim=-1)

def infonce(du, dm, uv, positive_xyz, negative_xyz, positive_mm, negative_mm, radius=2.3, temperature=0.14, return_details=False):
    q = sample(du, uv)
    p = sample(dm, positive_xyz)
    n = sample(dm, negative_xyz)
    positive = (q * p).sum(-1, keepdim=True) / temperature
    negative = q @ n.T / temperature
    allowed = (torch.cdist(positive_mm.float(), negative_mm.float()) >= radius) & ~(positive_xyz.round()[:, None, :] == negative_xyz.round()[None, :, :]).all(-1)
    if not allowed.any(1).all():
        raise ValueError('Every query needs distant negatives')
    logits = torch.cat([positive, negative.masked_fill(~allowed, -torch.inf)], 1)
    loss = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=q.device))
    if return_details:
        return (loss, dict(sampled_top1=float((logits.argmax(1) == 0).float().mean().detach())))
    return loss

class CenterDNS(nn.Module):
    """3D neighbour-neighbour DNS at center z only, in slice-index coordinates.
    dz=1/2 slices, dy/dx=2/4 pixels. No physical z-spacing assumption.
    """

    def __init__(self):
        super().__init__()
        template = DNS(3)
        self.pairs = template.pairs
        self.projection = nn.Parameter(torch.ones(10) / 8)

    def forward(self, h):
        z = h.shape[2] // 2
        out = []
        for dz, xy in [(1, 2), (2, 4)]:
            pad = F.pad(h, (xy, xy, xy, xy, 0, 0), mode='replicate')

            def shift(o):
                return pad[:, :, z + o[0] * dz, xy + o[1] * xy:xy + o[1] * xy + h.shape[3], xy + o[2] * xy:xy + o[2] * xy + h.shape[4]]
            variance = sum(((shift(a) - shift(b)).square() for a, b in self.pairs)) / len(self.pairs)
            variance = variance.clamp_min(9.3e-07)
            for a, b in self.pairs:
                s = torch.exp(-(shift(a) - shift(b)).square() / variance)
                out.append((s * self.projection.to(s.dtype)[None, :, None, None]).sum(1, keepdim=True))
        return torch.cat(out, 1)

class ThinUSBranch(nn.Module):

    def __init__(self):
        super().__init__()
        self.unet = UNet(4, thin=True)
        self.dns = CenterDNS()
        self.squeeze = nn.Sequential(nn.Conv2d(24, 32, 3, padding=1), nn.LeakyReLU(0.14), nn.Conv2d(32, 32, 1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] not in (5, 11):
            raise ValueError('US stack must have 5 or 9 layers')
        h = self.unet(x)
        s = checkpoint(self.dns, h, use_reentrant=False) if self.training else self.dns(h)
        return F.normalize(self.squeeze(s), dim=1, eps=1.1e-06)
