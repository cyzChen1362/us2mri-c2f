"""Read saved US windows and native MRI using the verified physical transforms.

No feature correspondence, DNS, negative sampling or tumor labels are used.
Arrays are ZYX; coordinate vectors and matrices are XYZ, at voxel centers.
"""
import itertools
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from c2f.common.paths import case_cache_path
from .physical_geometry import plane_grid_geometry, roi_bounds, roi_shape

def matrix(image):
    a = np.eye(5)
    a[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    a[:3, 3] = image.GetOrigin()
    return a

def tx_matrix(tx):
    a = np.eye(5)
    a[:3, 3] = tx.TransformPoint((0.0, 0.0, 0.0))
    for j in range(2):
        a[:3, j] = np.asarray(tx.TransformPoint(tuple(np.eye(3)[j]))) - a[:3, 3]
    return a

def map_points(a, p):
    return np.c_[p, np.ones(len(p))] @ a[:3].T

@lru_cache(maxsize=256)
def read_source(path):
    info = json.loads(Path(path).read_text())
    rows = {r['plane_id']: r for r in json.loads((Path(info['meta']) / 'planes.json').read_text())}
    transform = tx_matrix(sitk.ReadTransform(info['pull']))
    return (info, rows, transform)

def source(cfg, case):
    return read_source(str(case_cache_path(cfg, case) / 'source.json'))

def plane_ids(cfg, case):
    return sorted(source(cfg, case)[1])

def roi_geometry(info, T, spacing=1.0):
    reader = sitk.ImageFileReader()
    reader.SetFileName(info['source_us'])
    reader.ReadImageInformation()
    lo, hi = roi_bounds(matrix(reader), reader.GetSize(), T)
    shape = roi_shape(lo, hi, spacing)
    B = np.eye(4)
    B[:3, :3] *= spacing
    B[:3, 3] = (lo + hi) / 2 - (shape - 1) * spacing / 2
    return (B, shape)

def window_geometry(row, T, B, thickness_mm=1.0):
    """Center of unpadded US bbox; AABB of its physical 1-mm slab."""
    P = T @ np.asarray(row['US_pixel_to_LPS'])
    x0, y0, x1, y1 = row['US_bbox_xyxy']
    original = np.array([[x - 0.5, y - 0.5, 0.0] for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]])
    corners = map_points(P, original)
    normal = np.cross(P[:3, 0], P[:3, 1])
    normal /= np.linalg.norm(normal)
    slab = np.concatenate([corners - normal * thickness_mm / 2, corners + normal * thickness_mm / 2])
    corners_roi = map_points(np.linalg.inv(B), corners)
    slab_roi = map_points(np.linalg.inv(B), slab)
    low, high = (slab_roi.min(0), slab_roi.max(0))
    return dict(center_xyz=corners_roi.mean(0), size_xyz=high - low, corners_roi=corners_roi, corners_native=corners, normal_native=normal, box_low=low, box_high=high)

def prepare_volume(cfg, case):
    info, _, T = source(cfg, case)
    spacing = cfg['spacing']['mri_mm']
    B, shape = roi_geometry(info, T, spacing)
    native = sitk.ReadImage(info['source_mri'], sitk.sitkFloat32)
    ref = sitk.Image([int(x) for x in shape], sitk.sitkFloat32)
    ref.SetSpacing([spacing] * 3)
    ref.SetOrigin(B[:3, 3])
    a = sitk.GetArrayFromImage(native)
    native_mm = np.linalg.norm(matrix(native)[:3, :3], axis=0)
    sigma = np.maximum((spacing / native_mm - 1) / 2, 0)[::-1]
    filtered = sitk.GetImageFromArray(gaussian_filter(a, sigma, mode='nearest'))
    filtered.CopyInformation(native)
    a = sitk.GetArrayFromImage(sitk.Resample(filtered, ref, sitk.Transform(), sitk.sitkLinear, 0))
    ones = sitk.Image(native.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(native)
    valid = sitk.GetArrayFromImage(sitk.Resample(ones, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
    lo, hi = np.percentile(a[valid & (a != 0)], [0.5, 98.5])
    a = np.clip((a - lo) / max(hi - lo, 8.4e-07), 0, 1).astype(np.float32)
    a[~valid] = 0
    return dict(mri=torch.from_numpy(a)[None, None], valid=valid, roi_to_native=B, shape_xyz=shape)

def prepare_slice(cfg, case, plane_id, volume):
    info, rows, T = source(cfg, case)
    row = rows[plane_id]
    root = Path(info['root'])
    entry = next((x for x in row['outputs'] if x['kind'] == 'US'))
    raw = np.load(root / entry['path']).astype(np.float32)
    support = np.load(root / row['US_support']).astype(bool)
    spatial = cfg['model'].get('architecture') in ('spatial_template_v2', 'spatial_template_v3')
    P, Q, shape, extent = plane_grid_geometry(row, T, cfg['spacing']['us_mm'], multiple=9 if spatial else 5, odd_blocks=spatial)
    yy, xx = np.mgrid[:shape[1], :shape[0]]
    coords = map_points(Q, np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)])[:, [1, 0]].T
    low, high = np.percentile(raw[support], [0.5, 99.4])
    raw = np.clip((raw - low) / max(high - low, 1e-06), 0, 1)
    raw[~support] = 0
    sigma = np.maximum((np.linalg.norm(Q[:2, :2], axis=1) - 1) / 2, 0)[::-1]
    us = map_coordinates(gaussian_filter(raw, sigma, mode='constant'), coords, order=1, mode='constant', cval=0).reshape(yy.shape)
    mask = map_coordinates(support.astype(np.uint8), coords, order=0, mode='constant', cval=0).reshape(yy.shape) > 0
    us[~mask] = 0
    spacing = cfg['spacing']['us_mm']
    center = (shape - 1) / 2
    canvas = (np.abs((xx - center[0]) * spacing) <= extent[0] / 2) & (np.abs((yy - center[1]) * spacing) <= extent[1] / 2)
    target = window_geometry(row, T, volume['roi_to_native'], cfg['labels']['slab_thickness_mm'])
    return dict(us=torch.from_numpy(us.astype(np.float32))[None] * 2 - 1, canvas=torch.from_numpy(canvas.astype(np.float32))[None], support=mask, input_to_native=P, case=case, plane_id=plane_id, **target)
