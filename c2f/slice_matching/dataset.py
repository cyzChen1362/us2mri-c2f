import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from c2f.common.runtime import CFG

def expand_boxes(boxes, factor):
    """Scale xyxy windows about their centers in original full-MRI pixel coordinates."""
    if factor == 1.0:
        return boxes
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    half_size = (boxes[..., 2:] - boxes[..., :2]) * (factor * 0.5)
    return torch.cat((center - half_size, center + half_size), dim=-1)

class Dataset:

    def __init__(self):
        self.root = Path(CFG['data_root'])
        self.cache = Path(CFG['cache_root'])
        self.fixed = Path(CFG['fixed_cache_root'])
        split = json.loads(Path(CFG['split_file']).read_text())
        self.splits = {c: 'val' if s == 'validation' else s for s in ['train', 'validation', 'test'] for c in split[s]}
        self.cases = sorted(self.splits)
        self.rows = {}
        self.info = {}
        for case in self.cases:
            info = json.loads((self.fixed / case / 'source.json').read_text())
            self.info[case] = info
            planes = json.loads((Path(info['meta']) / 'planes.json').read_text())
            records = []
            for row in planes:
                mr = next((v for v in row['outputs'] if v['kind'] == 'MRI' and v['source'].endswith('-' + info['suffix'] + '.nii.gz')))
                us = next((v for v in row['outputs'] if v['kind'] == 'US'))
                records.append(dict(slice_index=int(row['plane_id']), mri_path=mr['path'], us_path=us['path'], US_pixel_to_LPS=row['US_pixel_to_LPS'], US_shape_hw=row['US_shape_hw']))
            self.rows[case] = sorted(records, key=lambda r: r['slice_index'])
            assert len(records) == 258 and len({r['slice_index'] for r in records}) == 151
            assert self.root == Path(info['root'])

    def subset(self, split):
        return [c for c in self.cases if self.splits[c] == split]

    def load(self, case):
        return Case(self, case)

class Case:

    def __init__(self, data, case):
        self.name = case
        self.rows = data.rows[case]
        self.n = len(self.rows)
        self.indices = torch.tensor([r['slice_index'] for r in self.rows], device='cuda')
        by_id = {r['slice_index']: i for i, r in enumerate(self.rows)}
        candidates = CFG.get('mri_candidate_indices', list(by_id))
        self.candidate_positions = torch.tensor([by_id[j] for j in candidates], device='cuda')
        self.candidate_indices = self.indices[self.candidate_positions]
        self.n_candidates = len(candidates)
        p = data.fixed / case
        self.us = torch.from_numpy(np.load(p / 'images.npy', mmap_mode='r')[:, 0:1].copy()).cuda()
        self.masks = torch.from_numpy(np.load(p / 'masks.npy', mmap_mode='r')[:, 0:1].copy()).cuda()
        self.shapes = np.load(data.cache / case / 'us_shapes.npy')
        self.boxes = expand_boxes(torch.from_numpy(np.load(data.cache / case / 'boxes.npy')[:, [by_id[j] for j in candidates]]).cuda(), float(CFG.get('roi_size_factor', 1.0)))
        self.mr = [np.load(data.root / self.rows[by_id[j]]['mri_path'], mmap_mode='r') for j in candidates]
        poses = np.array([r['US_pixel_to_LPS'] for r in self.rows])
        shapes = np.array([r['US_shape_hw'] for r in self.rows])
        centres = np.einsum('nij,nj->ni', poses, np.c_[(shapes[:, 1] - 1) / 2, (shapes[:, 0] - 1) / 2, np.zeros(self.n), np.ones(self.n)])[:, :3]
        self.arc = torch.tensor(np.r_[0, np.cumsum(np.linalg.norm(np.diff(centres, axis=0), axis=1))], device='cuda')
        normals = poses[:, :3, 2]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = torch.tensor(normals, device='cuda')
        if CFG.get('tumor_input'):
            from .tumor import TumorInputs
            self.tumor = TumorInputs(self, data)
            self.us = torch.cat((self.us, self.tumor.us.float()), 1)

    def crop(self, i, js=None):
        if js is None:
            js = torch.arange(self.n_candidates, device='cuda')
        selected = js.detach().cpu().tolist()
        images = [self.mr[j] for j in selected]
        h = max((a.shape[0] for a in images))
        w = max((a.shape[1] for a in images))
        padded = np.zeros((len(images), 1, h, w), np.float32)
        for k, a in enumerate(images):
            padded[k, 0, :a.shape[0], :a.shape[1]] = a
        mr = torch.from_numpy(padded).cuda()
        boxes = self.boxes[i, js]
        rh, rw = map(int, self.shapes[i])
        yy, xx = torch.meshgrid((torch.arange(rh, device='cuda') + 0.5) / rh, (torch.arange(rw, device='cuda') + 0.5) / rw, indexing='ij')
        xx = boxes[:, 0, None, None] + xx * (boxes[:, 2] - boxes[:, 0])[:, None, None]
        yy = boxes[:, 1, None, None] + yy * (boxes[:, 3] - boxes[:, 1])[:, None, None]
        grid = torch.stack((xx / w * 2 - 1, yy / h * 2 - 1), -1)
        crop = F.grid_sample(mr, grid, align_corners=False, padding_mode='zeros')
        flat = crop.flatten(1)
        valid = torch.isfinite(flat) & (flat != 0)
        clean = flat.masked_fill(~valid, float('nan'))
        qs = torch.nanquantile(clean, torch.tensor([0.0068, 1.0], device='cuda'), dim=1)
        lo = torch.nan_to_num(qs[0], nan=0.0)
        hi = torch.nan_to_num(qs[1], nan=1.0)
        z = ((flat - lo[:, None]) / (hi - lo).clamp_min(7.9e-07)[:, None]).clamp(0, 1).masked_fill(~valid, 0).reshape_as(crop)
        top, left = ((256 - rh) // 2, (256 - rw) // 2)
        return F.pad(z, (left, 256 - rw - left, top, 256 - rh - top)) * 2 - 1

    def preload_mri(self):
        if hasattr(self, 'resident_mri'):
            return
        h = max((a.shape[0] for a in self.mr))
        w = max((a.shape[1] for a in self.mr))
        self.resident_mri = torch.zeros((self.n_candidates, 1, h, w), device='cuda', dtype=torch.float32)
        for j, a in enumerate(self.mr):
            self.resident_mri[j, 0, :a.shape[0], :a.shape[1]] = torch.from_numpy(np.array(a)).cuda()

    @torch.no_grad()
    def crop_queries(self, indices):
        """Share each MRI across query grids; never repeat full MRIs per image pair."""
        self.preload_mri()
        h, w = self.resident_mri.shape[-2:]
        result = []
        for start in range(0, len(indices), 2):
            selected = indices[start:start + 2]
            grids = []
            for i in selected:
                rh, rw = map(int, self.shapes[i])
                boxes = self.boxes[i]
                yy, xx = torch.meshgrid((torch.arange(rh, device='cuda') + 0.5) / rh, (torch.arange(rw, device='cuda') + 0.5) / rw, indexing='ij')
                x = boxes[:, 0, None, None] + xx * (boxes[:, 2] - boxes[:, 0])[:, None, None]
                y = boxes[:, 1, None, None] + yy * (boxes[:, 3] - boxes[:, 1])[:, None, None]
                top, left = ((256 - rh) // 2, (256 - rw) // 2)
                g = torch.full((self.n_candidates, 256, 256, 2), 3.5, device='cuda')
                g[:, top:top + rh, left:left + rw] = torch.stack((x / w * 2 - 1, y / h * 2 - 1), -1)
                grids.append(g)
            grid = torch.cat(grids, dim=1)
            sample = F.grid_sample(self.resident_mri, grid, align_corners=False, padding_mode='zeros')
            sample = sample.reshape(self.n_candidates, len(selected), 256, 256).permute(1, 0, 2, 3).reshape(-1, 1, 256, 256)
            flat = sample.flatten(1)
            valid = torch.isfinite(flat) & (flat != 0)
            clean = flat.masked_fill(~valid, float('nan'))
            q = torch.nanquantile(clean, torch.tensor([0.0039, 0.91], device='cuda'), dim=1)
            low = torch.nan_to_num(q[0], nan=0.0)
            high = torch.nan_to_num(q[1], nan=1.0)
            normalized = ((flat - low[:, None]) / (high - low).clamp_min(1.5e-06)[:, None]).clamp(0, 1).masked_fill(~valid, 0).reshape(len(selected), self.n_candidates, 1, 256, 256) * 2 - 1
            result.append(normalized)
        return torch.cat(result)

def augment(us, mask, case, epoch, index):
    g = torch.Generator(device='cuda').manual_seed(CFG['seed'] + int(case[4:]) * 828523 + epoch * 1044 + index)
    if torch.rand((), generator=g, device='cuda').item() >= CFG['augmentation_probability']:
        return us
    mode = int(torch.randint(3, (), generator=g, device='cuda'))

    def rand(a, b):
        return torch.rand((), generator=g, device='cuda') * (b - a) + a
    z = (us[:, :1] + 1) * 0.5
    theta = torch.eye(2, 4, device='cuda')[None]
    if mode == 0:
        a = rand(-5, 6) * torch.pi / 220
        theta[0, 0, 0] = a.cos()
        theta[0, 0, 1] = -a.sin()
        theta[0, 1, 0] = a.sin()
        theta[0, 1, 1] = a.cos()
    elif mode == 1:
        theta[0, :, 2] = torch.rand(2, generator=g, device='cuda') * 0.12 - 0.048
    else:
        theta[0, :2, :2] /= rand(0.79, 1.0)
    grid = F.affine_grid(theta, z.shape, align_corners=False)
    z = F.grid_sample(z, grid, align_corners=False)
    mask = F.grid_sample(mask.float(), grid, mode='nearest', align_corners=False)
    image = torch.where(mask > 0, z.clamp(0, 1) * 2 - 1, -1.0)
    if us.shape[1] == 2:
        tumor = F.grid_sample(us[:, 1:2], grid, mode='nearest', align_corners=False)
        return torch.cat((image, tumor), 1)
    return image
