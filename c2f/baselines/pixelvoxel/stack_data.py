"""Adjacent SAVED 2D images, inexpensive 2D canvas alignment, no 3D US reslicing.
Central image/mask/positive coordinates are bitwise unchanged from v1.
Auxiliary planes project the central canvas into their own saved image coordinates;
z is a sequence index, not a physical millimetre axis. Raw images cached per case.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates, gaussian_filter

def add_stack(d, cfg, cache=None):
    cache = {} if cache is None else cache
    key = ('us_stack', d['info']['meta'])
    if key not in cache:
        rows = json.loads((Path(d['info']['meta']) / 'planes.json').read_text())
        cache[key] = dict(rows=rows, ids=[r['plane_id'] for r in rows], raw={})
    state = cache[key]
    rows = state['rows']
    center = state['ids'].index(d['row']['plane_id'])
    depth = cfg['model']['us_layers']
    radius = depth // 2
    positions = np.clip(np.arange(center - radius, center + radius + 1), 0, len(rows) - 1)
    root = Path(d['info']['root'])
    H, W = d['mask'].shape
    yy, xx = np.mgrid[:H, :W]
    p = np.stack([xx, yy, np.zeros_like(xx), np.ones_like(xx)]).reshape(5, -1)
    M = d['input_to_native']
    world = (M @ p)[:3]
    original_support = np.load(root / d['row']['US_support']).astype(bool)
    low, high = np.percentile(d['raw'][original_support], [0.5, 98.3])
    den = max(high - low, 7.2e-07)
    images = []
    supports = []
    mappings = []
    for index in positions:
        row = rows[int(index)]
        P = d['T'] @ np.asarray(row['US_pixel_to_LPS'])
        inverse = np.linalg.pinv(P[:3, :2])
        A = np.eye(3)
        A[:2, :2] = inverse @ M[:3, :2]
        A[:2, 2] = inverse @ (M[:3, 3] - P[:3, 3])
        mappings.append(A)
        if index == center:
            images.append(d['us'][0, 0].numpy())
            supports.append(d['us_support'])
            continue
        if index not in state['raw']:
            entry = next((e for e in row['outputs'] if e['kind'] == 'US'))
            state['raw'][index] = (np.load(root / entry['path']).astype(np.float32), np.load(root / row['US_support']).astype(bool))
        raw, support = state['raw'][index]
        norm = np.clip((raw - low) / den, 0, 1)
        norm[~support] = 0
        sigma = np.maximum((np.linalg.norm(A[:2, :2], axis=1) - 1) / 2, 0)[::-1]
        norm = gaussian_filter(norm, sigma, mode='constant', cval=0)
        uv = inverse @ (world - P[:3, 3, None])
        coords = uv[::-1]
        image = map_coordinates(norm, coords, order=1, mode='constant', cval=0).reshape(H, W)
        mask = map_coordinates(support.astype(np.uint8), coords, order=0, mode='constant', cval=0).reshape(H, W) > 0
        image[~mask] = 0
        images.append(image * 2 - 1)
        supports.append(mask)
    d['us_center'] = d['us']
    d['us'] = torch.from_numpy(np.stack(images).astype(np.float32))[None, None]
    d['stack_support'] = np.stack(supports)
    d['stack_plane_ids'] = [rows[int(k)]['plane_id'] for k in positions]
    d['stack_to_original_xy'] = np.stack(mappings)
    d['stack_center_index'] = radius
    return d
