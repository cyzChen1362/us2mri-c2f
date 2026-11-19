"""Seven integer positive classes, exactly equal classification/regression weights."""
import numpy as np
import torch
from torch.nn import functional as F
OFFSETS = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]

def targets(queries, shape, device, cfg):
    centers = np.stack([q['center_xyz'] for q in queries])
    c = torch.as_tensor(centers, device=device, dtype=torch.float32)
    s = torch.as_tensor(np.stack([q['size_xyz'] for q in queries]), device=device, dtype=torch.float32)
    stride = cfg['model']['stride']
    base = torch.as_tensor(np.floor((centers + 0.5) / stride).astype(np.int64), device=device)
    xyz = base[:, None] + torch.tensor(OFFSETS, device=device)[None]
    dims = torch.tensor(shape[::-1], device=device)
    if not bool(((xyz >= 0) & (xyz < dims)).all()):
        raise ValueError('Seven labels must all be inside the output grid')
    indices = xyz[:, :, 0] + shape[2] * (xyz[:, :, 1] + shape[1] * xyz[:, :, 2])
    g = xyz.float() * stride + (stride - 1) / 2
    return dict(indices=indices, offset=(c[:, None] - g) / stride, size=s / cfg['labels']['size_scale_mm'])

def loss_fn(out, queries, cfg):
    t = targets(queries, out['logits'].shape[1:], out['logits'].device, cfg)
    cls = -F.log_softmax(out['logits'].float().flatten(1), 1).gather(1, t['indices']).mean(1)
    k = t['indices'][:, None].expand(-1, 3, -1)
    off = out['offset'].float().flatten(2).gather(2, k).transpose(1, 2)
    size = out['size'].float().flatten(2).gather(2, k).transpose(1, 2)
    offloss = F.smooth_l1_loss(off, t['offset'], reduction='none').mean((1, 2))
    sizeloss = F.smooth_l1_loss(size, t['size'][:, None].expand_as(size), reduction='none').mean((1, 2))
    total = cls + cfg['loss']['offset_weight'] * offloss + cfg['loss']['size_weight'] * sizeloss
    return (total.mean(), dict(loss=total.detach(), classification_loss=cls.detach(), offset_loss=offloss.detach(), size_loss=sizeloss.detach()))
