"""One native-MRI resampling per query group, with all corners in view."""
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import gaussian_filter
from .data import source, roi_geometry, window_geometry, matrix, map_points

def load_case(cfg, case):
    info, rows, T = source(cfg, case)
    spacing = cfg['spacing']['mri_mm']
    B, shape = roi_geometry(info, T, spacing)
    native = sitk.ReadImage(info['source_mri'], sitk.sitkFloat32)
    a = sitk.GetArrayFromImage(native)
    native_mm = np.linalg.norm(matrix(native)[:3, :3], axis=0)
    sigma = np.maximum((spacing / native_mm - 1) / 2, 0)[::-1]
    filtered = sitk.GetImageFromArray(gaussian_filter(a, sigma, mode='nearest'))
    filtered.CopyInformation(native)
    coverage = sitk.Image(native.GetSize(), sitk.sitkUInt8) + 1
    coverage.CopyInformation(native)
    return dict(filtered=filtered, coverage=coverage, base_B=B, base_shape=shape, rows=rows, T=T)

def slab_points(cfg, native, ids):
    result = []
    for pid in ids:
        t = window_geometry(native['rows'][pid], native['T'], np.eye(4), cfg['labels']['slab_thickness_mm'])
        corners = t['corners_native']
        d = t['normal_native'] * cfg['labels']['slab_thickness_mm'] / 2
        result.extend([corners - d, corners + d])
    return np.concatenate(result)

def rotation(axis, angle):
    x, y, z = axis / np.linalg.norm(axis)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def sample_geometry(cfg, native, ids, epoch, group):
    options = cfg['augmentation']
    spacing = cfg['spacing']['mri_mm']
    base = native['base_shape']
    shape = (np.ceil(base * options['fov_scale'] / 4) * 4).astype(int)
    center = map_points(native['base_B'], ((base - 1) / 2)[None])[0]
    points = slab_points(cfg, native, ids)
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], epoch, group, 3119]))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    requested = float(rng.uniform(-options['rotation_degrees'], options['rotation_degrees']))
    limit = base * spacing * options['translation_fraction']
    half = shape * spacing / 2 - options['label_margin_mm']

    def interval(degrees):
        R = rotation(axis, np.deg2rad(degrees))
        local = (points - center) @ R
        lo = np.maximum(local.max(0) - half, -limit)
        hi = np.minimum(local.min(0) + half, limit)
        return (R, lo, hi)
    R, lo, hi = interval(requested)
    angle = requested
    if np.any(lo > hi):
        lower, upper = (0.0, 1.0)
        R0, l0, h0 = interval(0.0)
        if np.any(l0 > h0):
            raise ValueError('Expanded FOV does not contain the original US group')
        for _ in range(26):
            mid = (lower + upper) / 2
            _, lm, hm = interval(requested * mid)
            if np.all(lm <= hm):
                lower = mid
            else:
                upper = mid
        angle = requested * lower
        R, lo, hi = interval(angle)
    shift = rng.uniform(lo, hi)
    new_center = center + R @ shift
    B = np.eye(3)
    B[:3, :3] = R * spacing
    B[:3, 3] = new_center - R @ ((shape - 1) * spacing / 2)
    mapped = map_points(np.linalg.inv(B), points)
    if not np.all((mapped >= -0.5 + options['label_margin_mm'] / spacing - 1.5e-06) & (mapped <= shape - 0.5 - options['label_margin_mm'] / spacing + 8.9e-07)):
        raise AssertionError('Augmented target corners outside promised margin')
    return (B, shape, dict(rotation_degrees=float(angle), requested_rotation_degrees=requested, rotation_axis=axis.tolist(), translation_local_mm=shift.tolist(), translation_limit_mm=limit.tolist(), feasible_low_mm=lo.tolist(), feasible_high_mm=hi.tolist(), shape_xyz=shape.tolist(), roi_to_native=B.tolist(), group_size=len(ids), epoch=epoch, group=group))

def prepare_group(cfg, case, ids, native, epoch, group):
    B, shape, meta = sample_geometry(cfg, native, ids, epoch, group)
    spacing = cfg['spacing']['mri_mm']
    ref = sitk.Image([int(n) for n in shape], sitk.sitkFloat32)
    ref.SetSpacing([spacing] * 2)
    ref.SetDirection((B[:3, :3] / spacing).ravel().tolist())
    ref.SetOrigin(B[:3, 3].tolist())
    a = sitk.GetArrayFromImage(sitk.Resample(native['filtered'], ref, sitk.Transform(), sitk.sitkLinear, 0))
    valid = sitk.GetArrayFromImage(sitk.Resample(native['coverage'], ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
    lo, hi = np.percentile(a[valid & (a != 0)], [0.5, 98.0])
    a = np.clip((a - lo) / max(hi - lo, 1e-06), 0, 1).astype(np.float32)
    a[~valid] = 0
    return dict(mri=torch.from_numpy(a)[None, None], valid=valid, roi_to_native=B, shape_xyz=shape, frame_rotation=B[:3, :3] / spacing, augmentation=meta)
