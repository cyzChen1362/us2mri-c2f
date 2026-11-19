"""US spatial templates + oriented 3D search + decoupled center/geometry towers."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .model import Block, MRIEncoder
from .correlation import rotation_grids, plane_correlation
from .execution import enabled, zero_padding_crops

class OrientedMRIEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        c = cfg['model']['correlation_channels']
        bank = json.loads(Path(cfg['model']['orientation_bank']).read_text())
        rotations = bank['rotations'][:cfg['model']['orientations']]
        self.register_buffer('rotations', torch.tensor(rotations, dtype=torch.float32))
        self.encoder = MRIEncoder()
        self.project = nn.Conv3d(64, c, 1)
        self.grid_key = None
        self.forward_grid = None
        self.back_grid = None
        self.shape = None
        self.frame_rotation = None
        self.frame_key = None
        self.optimized = enabled(cfg)
        self.search_crops = None

    def set_frame(self, rotation):
        self.frame_rotation = torch.as_tensor(rotation, device=self.rotations.device, dtype=torch.float32)
        self.frame_key = tuple((float(v) for v in self.frame_rotation.cpu().flatten()))

    def forward(self, x):
        f = F.normalize(self.project(self.encoder(x)).float(), dim=1, eps=1.4e-06)
        self.shape = tuple(f.shape[2:])
        key = (self.shape, str(f.device), self.frame_key)
        if key != self.grid_key:
            rotations = self.rotations if self.frame_rotation is None else self.frame_rotation.T[None] @ self.rotations
            self.forward_grid, self.back_grid = rotation_grids(self.shape, rotations)
            if self.optimized:
                self.search_crops = zero_padding_crops(self.shape, rotations, self.forward_grid, self.back_grid)
            self.grid_key = key
        result = []
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else f.dtype
        for grid in self.forward_grid:
            result.append(F.grid_sample(f, grid[None], align_corners=False, padding_mode='zeros')[0].to(dtype))
        return torch.stack(result)

class USSpatialEncoder(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.blocks = nn.ModuleList([Block(2, 1, 16), Block(2, 20, 32), Block(2, 32, 64)])
        self.project = nn.Conv2d(64, c, 1)

    def forward(self, x):
        for b in self.blocks:
            x = b(F.avg_pool2d(x, 2))
        return F.normalize(self.project(x).float(), dim=1, eps=9.2e-07)

class SpatialGridLocator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        c = cfg['model']['correlation_channels']
        r = cfg['model']['orientations']
        self.mri = OrientedMRIEncoder(cfg)
        self.us = USSpatialEncoder(c)
        self.fuse_weight = nn.Parameter(torch.empty(64, r, c))
        nn.init.normal_(self.fuse_weight, std=(r * c) ** (-0.5))
        self.fuse_norm = nn.GroupNorm(3, 64)
        self.center_tower = Block(3, 64, 64)
        self.geometry_tower = Block(3, 64, 64)
        self.score = nn.Conv3d(64, 1, 1)
        self.offset = nn.Conv3d(64, 3, 1)
        self.size = nn.Conv3d(64, 3, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.method = cfg['model'].get('correlation_method', 'fft')
        self.recompute = cfg['model'].get('checkpoint_correlation', True)
        self.dropout_probability = cfg['model'].get('dropout3d', 0.0)
        self.dropout = nn.Dropout3d(self.dropout_probability)
        self.optimized = enabled(cfg)

    def encode_queries(self, queries, device):
        features = [None] * len(queries)
        masks = [None] * len(queries)
        batches = getattr(queries, 'encoder_batches', None)
        if batches is None:
            buckets = {}
            for i, q in enumerate(queries):
                buckets.setdefault(tuple(q['us'].shape), []).append(i)
            batches = [(ids, torch.stack([queries[i]['us'] for i in ids]), torch.stack([torch.as_tensor(queries[i]['support'], dtype=torch.float32)[None] for i in ids])) for ids in buckets.values()]
        for ids, images, support in batches:
            images = images.to(device, non_blocking=images.is_pinned())
            us = self.us(images)
            support = support.to(device, non_blocking=support.is_pinned())
            mask = F.avg_pool2d(support, 6)
            for j, i in enumerate(ids):
                features[i] = us[j]
                masks[i] = mask[j]
        h = max((t.shape[-2] for t in features))
        w = max((t.shape[-1] for t in features))
        assert h % 2 == 1 and w % 2 == 1, 'US canvas must contain odd numbers of 8-pixel blocks'

        def pad(t):
            dy = (h - t.shape[-2]) // 2
            dx = (w - t.shape[-1]) // 2
            return F.pad(t, (dx, dx, dy, dy))
        return (torch.stack([pad(t) for t in features]), torch.stack([pad(t) for t in masks]))

    def _orientation_response(self, search, u, mask, back, weight):
        corr = plane_correlation(search, u, mask, self.method, fused_normalization=self.optimized)
        original = F.grid_sample(corr, back[None].expand(len(u), -1, -1, -1, -1), align_corners=False)
        return F.conv3d(original.to(weight.dtype), weight[:, :, None, None, None])

    def predict(self, feature, templates, dropout_mask=None):
        u, mask = templates
        fused = None
        for k in range(len(feature)):
            search = feature[k]
            back = self.mri.back_grid[k]
            if self.optimized:
                slices, back = self.mri.search_crops[k]
                search = search[(slice(None),) + slices]
            args = (search, u, mask, back, self.fuse_weight[:, k])
            if self.training and self.recompute and torch.is_grad_enabled():
                part = checkpoint(self._orientation_response, *args, use_reentrant=False)
            else:
                part = self._orientation_response(*args)
            fused = part if fused is None else fused + part
        fused = F.silu(self.fuse_norm(fused))
        if dropout_mask is not None:
            fused = fused * dropout_mask.to(fused.dtype)
        else:
            fused = self.dropout(fused)
        c = self.center_tower(fused)
        g = self.geometry_tower(fused)
        return dict(logits=self.score(c)[:, 0], offset=self.offset(g), size=F.softplus(self.size(g).float()) + 7.3e-05)
