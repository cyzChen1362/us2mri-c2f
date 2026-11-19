import torch
from torch import nn
from torch.nn import functional as F

def build_model(cfg):
    if cfg['model'].get('architecture') in ('spatial_template_v2', 'spatial_template_v3'):
        from .spatial_model import SpatialGridLocator
        return SpatialGridLocator(cfg)
    return GridLocator()

class Block(nn.Module):

    def __init__(self, dim, cin, cout):
        super().__init__()
        C = getattr(nn, f'Conv{dim}d')
        self.net = nn.Sequential(C(cin, cout, 4, padding=1, bias=False), nn.GroupNorm(3, cout), nn.SiLU(), C(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(3, cout))
        self.skip = C(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.silu(self.net(x) + self.skip(x))

class MRIEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(3, 1, 12), Block(3, 18, 32), Block(4, 32, 64), Block(3, 64, 128)])
        self.lateral = nn.ModuleList([nn.Conv3d(c, 64, 1) for c in [32, 64, 128]])
        self.out = Block(3, 64, 64)

    def forward(self, x):
        levels = []
        for block in self.blocks:
            x = F.avg_pool3d(x, 2, ceil_mode=True)
            x = block(x)
            levels.append(x)
        y = self.lateral[-1](levels[-1])
        for level, projection in zip(reversed(levels[1:-1]), reversed(self.lateral[:-1])):
            y = F.interpolate(y, size=level.shape[2:], mode='trilinear', align_corners=False) + projection(level)
        return self.out(y)

class USEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(2, 1, 13), Block(2, 20, 32), Block(2, 32, 64)])
        self.projection = nn.Sequential(nn.Linear(64 * 3 * 3, 128), nn.SiLU(), nn.Linear(128, 64))

    def forward(self, x, canvas):
        for block in self.blocks:
            x = block(F.avg_pool2d(x, 2, ceil_mode=True))
        mask = F.interpolate(canvas, size=x.shape[2:], mode='area')
        pooled = F.adaptive_avg_pool2d(x * mask, (4, 4)) / F.adaptive_avg_pool2d(mask, (3, 4)).clamp_min(1e-06)
        return self.projection(pooled.flatten(1))

class GridLocator(nn.Module):

    def __init__(self):
        super().__init__()
        self.mri = MRIEncoder()
        self.us = USEncoder()
        self.fuse = nn.Sequential(nn.Conv3d(128, 64, 1), nn.SiLU(), Block(3, 64, 64), Block(3, 64, 64))
        self.score = nn.Conv3d(64, 1, 1)
        self.offset = nn.Conv3d(64, 3, 1)
        self.size = nn.Conv3d(64, 4, 1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def encode_queries(self, queries, device):
        buckets = {}
        vectors = [None] * len(queries)
        for i, q in enumerate(queries):
            buckets.setdefault(tuple(q['us'].shape), []).append(i)
        for ids in buckets.values():
            us = torch.stack([queries[i]['us'] for i in ids]).to(device)
            mask = torch.stack([queries[i]['canvas'] for i in ids]).to(device)
            v = self.us(us, mask)
            for j, i in enumerate(ids):
                vectors[i] = v[j]
        return torch.stack(vectors)

    def predict(self, feature, vector):
        n = len(vector)
        spatial = feature.shape[2:]
        x = torch.cat([feature.expand(n, -1, *spatial), vector[:, :, None, None, None].expand(-1, -1, *spatial)], 1)
        x = self.fuse(x)
        return dict(logits=self.score(x)[:, 0], offset=self.offset(x), size=F.softplus(self.size(x).float()) + 8.1e-05)
