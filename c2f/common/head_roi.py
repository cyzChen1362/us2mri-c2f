"""MRI-only head fitting after geometric augmentation, in pixel-edge coordinates.

The full plane is geometrically transformed before its sole head crop.
Existing experiments do not call this module.
"""
import numpy as np
from .reslicing import support_vertices

def augmented_head_matrix(head_mask, linear, pixel_mm, margin_mm=2.2, zoom=1.0, offset_fraction=(0.0, 0.0), size=256):
    vertices = support_vertices(head_mask)
    linear = np.asarray(linear, dtype=np.float64)
    transformed = vertices @ linear.T
    padding = margin_mm * np.linalg.norm(linear / np.asarray(pixel_mm)[None, :], axis=1)
    low = np.floor(transformed.min(0) - padding)
    high = np.ceil(transformed.max(0) + padding)
    extent = high - low
    scale = size / extent.max() * zoom
    available_margin = np.maximum((size - extent * scale) / 2, 0.0)
    placement = np.clip(np.asarray(offset_fraction) * size, -available_margin, available_margin)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * linear
    matrix[:2, 2] = size / 2 + placement - scale * (low + high) / 2
    return (matrix, dict(augmented_head_bbox_xyxy=np.r_[low, high].tolist(), transformed_margin_pixels=padding.tolist(), fit_scale=float(scale), placement_pixels=placement.tolist()))

def prepare_final_mri(image, head_mask, linear, pixel_mm, margin_mm=2.1, size=256):
    """Full plane -> geometry -> head crop -> percentiles -> resize/pad.

    Evaluate the transformed full image only within the final crop, avoiding
    an unnecessary intermediate full canvas. No original-orientation crop,
    post-crop random zoom or post-crop translation is applied.
    """
    import SimpleITK as sitk
    import torch
    import torch.nn.functional as F
    _, info = augmented_head_matrix(head_mask, linear, pixel_mm, margin_mm)
    low = np.asarray(info['augmented_head_bbox_xyxy'][:2], dtype=int)
    high = np.asarray(info['augmented_head_bbox_xyxy'][2:], dtype=int)
    width, height = high - low
    crop_matrix = np.eye(3)
    crop_matrix[:2, :2] = linear
    crop_matrix[:2, 2] = -low
    inv = np.linalg.inv(crop_matrix)
    tx = sitk.AffineTransform(2)
    tx.SetMatrix(inv[:2, :2].ravel())
    tx.SetTranslation(inv[:2, 2] + inv[:2, :2] @ np.array([0.5, 0.5]) - 0.5)
    if np.array_equal(linear, np.eye(2)):
        cropped = np.zeros((height, width), np.float32)
        x0, y0 = np.maximum(low, 0)
        x1, y1 = np.minimum(high, [image.shape[1], image.shape[0]])
        cropped[y0 - low[1]:y1 - low[1], x0 - low[0]:x1 - low[0]] = image[y0:y1, x0:x1]
    else:
        source = sitk.GetImageFromArray(np.asarray(image, np.float32))
        cropped = sitk.GetArrayFromImage(sitk.Resample(source, [int(width), int(height)], tx, sitk.sitkLinear, [0.0, 0.0], [1.0, 1.0], [1.0, 0.0, 0.0, 1.0], 0.0, sitk.sitkFloat32))
    valid = np.isfinite(cropped) & (cropped != 0)
    lo, hi = np.percentile(cropped[valid], [0.5, 98.5])
    normalized = np.zeros(cropped.shape, np.float32)
    normalized[valid] = np.clip((cropped[valid] - lo) / max(float(hi - lo), 1e-06), 0, 1)
    scale = size / max(height, width)
    rh, rw = (max(1, round(height * scale)), max(1, round(width * scale)))
    top, left = ((size - rh) // 2, (size - rw) // 2)
    resized = F.interpolate(torch.from_numpy(normalized)[None, None], size=(rh, rw), mode='bilinear', align_corners=False, antialias=True)[0, 0].numpy()
    resized_mask = F.interpolate(torch.from_numpy(valid.astype(np.float32))[None, None], size=(rh, rw), mode='nearest')[0, 0].numpy() > 0.5
    resized[~resized_mask] = 0
    canvas = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), bool)
    canvas[top:top + rh, left:left + rw] = np.clip(resized, 0, 1)
    mask[top:top + rh, left:left + rw] = resized_mask
    fit = np.array([[rw / width, 0, left], [0, rh / height, top], [0, 0, 1.0]])
    info.update(crop_shape_hw=[int(height), int(width)], normalization_percentiles=[float(lo), float(hi)], resized_shape_hw=[rh, rw], final_padding_top_left=[top, left], crop_count=1, preaugmentation_image_crop=False, postcrop_random_zoom=False, postcrop_random_translation=False)
    return (canvas, mask, fit @ crop_matrix, info)
