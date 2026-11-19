"""Reuse saved US planes and authoritative per-case pull transforms."""
import itertools, json, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c2f.common.geometry import roi_transform
from c2f.common.preprocess import prepare_image
from c2f.common.paths import case_cache_path

def matrix(image):
    a = np.eye(4)
    a[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    a[:3, 3] = image.GetOrigin()
    return a

def tx_matrix(tx):
    a = np.eye(4)
    a[:3, 3] = tx.TransformPoint((0.0, 0.0, 0.0))
    for j in range(3):
        a[:3, j] = np.asarray(tx.TransformPoint(tuple(np.eye(4)[j]))) - a[:3, 3]
    return a

def map_points(a, p):
    return np.c_[p, np.ones(len(p))] @ a[:3].T

def prepare_single(cfg, case, plane_id=104, us_spacing_mm=0.5, mri_spacing_mm=1.0, volume_cache=None):
    """Dual physical spacing in native MRI space, including affine scale/shear.

    US is regridded onto an orthonormal plane rather than stretching a bounding
    rectangle. MRI uses a case-level rectangular ROI. US and MRI have independently specified physical spacing.
    Only symmetric padding to multiples of four changes the output field of view.
    """
    from scipy.ndimage import map_coordinates, gaussian_filter
    from .physical_geometry import plane_grid_geometry, roi_bounds, roi_shape
    spacing = float(mri_spacing_mm)
    us_spacing = float(us_spacing_mm)
    if spacing <= 0:
        raise ValueError('spacing_mm must be positive')
    info = json.loads((case_cache_path(cfg, case) / 'source.json').read_text())
    rows = json.loads((Path(info['meta']) / 'planes.json').read_text())
    row = next((r for r in rows if r['plane_id'] == plane_id))
    root = Path(info['root'])
    entry = next((e for e in row['outputs'] if e['kind'] == 'US'))
    raw = np.load(root / entry['path']).astype(np.float32)
    support = np.load(root / row['US_support']).astype(bool)
    T = tx_matrix(sitk.ReadTransform(info['pull']))
    pixel_to_native, input_to_original, shape_xy, extent = plane_grid_geometry(row, T, us_spacing)
    yy, xx = np.mgrid[:shape_xy[1], :shape_xy[0]]
    points = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]
    original = map_points(input_to_original, points)
    coords = original[:, [1, 0]].T
    low, high = np.percentile(raw[support], [0.5, 99.2])
    norm = np.clip((raw - low) / max(high - low, 6.9e-07), 0, 1)
    norm[~support] = 0
    stride = np.linalg.norm(input_to_original[:2, :2], axis=1)
    sigma = np.maximum((stride - 1) / 2, 0)[::-1]
    smooth = gaussian_filter(norm, sigma, mode='constant', cval=0)
    ui = map_coordinates(smooth, coords, order=1, mode='constant', cval=0).reshape(yy.shape)
    mask = map_coordinates(support.astype(np.uint8), coords, order=0, mode='constant', cval=0).reshape(yy.shape) > 0
    ui[~mask] = 0
    key = (info['source_us'], info['source_mri'], info['pull'], spacing, T.tobytes())
    cached = None if volume_cache is None else volume_cache.get(key)
    if cached is None:
        cached = prepare_mri_volume(info, T, spacing)
        if volume_cache is not None:
            volume_cache[key] = cached
    volume, valid, B, native_matrix, native_size, lo, hi, shape_xyz, mri_sigma = cached
    mapping = np.linalg.inv(B) @ pixel_to_native
    xyz = map_points(mapping, points)
    native_idx = map_points(np.linalg.inv(native_matrix) @ pixel_to_native, points)
    ok = ((xyz >= 0) & (xyz <= shape_xyz - 1)).all(1) & ((native_idx >= 0) & (native_idx <= native_size - 1)).all(1)
    nearest = np.rint(xyz).astype(int)
    inside = ((nearest >= 0) & (nearest < shape_xyz)).all(1)
    nearest_valid = np.zeros(len(nearest), bool)
    p = nearest[inside]
    nearest_valid[inside] = valid[p[:, 2], p[:, 1], p[:, 0]]
    valid_us = mask & ok.reshape(mask.shape) & nearest_valid.reshape(mask.shape)
    return dict(us=torch.from_numpy(ui.astype(np.float32))[None, None] * 2 - 1, mri=torch.from_numpy(volume)[None, None], mask=valid_us, us_support=mask, valid=valid, input_to_native=pixel_to_native, input_to_roi=mapping, input_to_original=input_to_original, roi_to_native=B, info=info, row=row, spacing_mm=spacing, us_spacing_mm=us_spacing, raw=raw, T=T, mri_extent_mm=hi - lo, us_extent_mm=extent, us_shape_xy=shape_xy, mri_shape_xyz=shape_xyz, support_in_mri_fraction=float(valid_us.sum() / max(mask.sum(), 1)), antialias_sigma_us_yx=sigma.tolist(), antialias_sigma_mri_zyx=mri_sigma.tolist())

def prepare_mri_volume(info, T, spacing):
    from scipy.ndimage import gaussian_filter
    from .physical_geometry import roi_bounds, roi_shape
    us = sitk.ImageFileReader()
    us.SetFileName(info['source_us'])
    us.ReadImageInformation()
    native = sitk.ReadImage(info['source_mri'], sitk.sitkFloat32)
    lo, hi = roi_bounds(matrix(us), us.GetSize(), T)
    shape_xyz = roi_shape(lo, hi, spacing)
    ref = sitk.Image([int(n) for n in shape_xyz], sitk.sitkFloat32)
    ref.SetSpacing([spacing] * 2)
    ref.SetOrigin((lo + hi) / 2 - (shape_xyz - 1) * spacing / 2)
    native_array = sitk.GetArrayFromImage(native)
    native_mm = np.linalg.norm(matrix(native)[:3, :3], axis=0)
    mri_sigma = np.maximum((spacing / native_mm - 1) / 2, 0)[::-1]
    filtered = sitk.GetImageFromArray(gaussian_filter(native_array, mri_sigma, mode='nearest'))
    filtered.CopyInformation(native)
    volume = sitk.GetArrayFromImage(sitk.Resample(filtered, ref, sitk.Transform(), sitk.sitkLinear, 0))
    ones = sitk.Image(native.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(native)
    valid = sitk.GetArrayFromImage(sitk.Resample(ones, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
    nz = valid & (volume != 0)
    low, high = np.percentile(volume[nz], [0.5, 98.1])
    volume = np.clip((volume - low) / max(high - low, 1.3e-06), 0, 1).astype(np.float32)
    volume[~valid] = 0
    return (volume, valid, matrix(ref), matrix(native), np.asarray(native.GetSize()), lo, hi, shape_xyz, mri_sigma)

def prepare(cfg, case, plane_id=88, us_spacing_mm=0.5, mri_spacing_mm=1.0, volume_cache=None):
    d = prepare_single(cfg, case, plane_id, us_spacing_mm, mri_spacing_mm, volume_cache)
    if cfg.get('model', {}).get('us_input') == 'adjacent_stack':
        from .stack_data import add_stack
        d = add_stack(d, cfg, volume_cache)
    return d
