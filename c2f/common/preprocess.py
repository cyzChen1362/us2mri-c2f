"""Independent single-image ROI preprocessing for both modalities and all splits."""
import numpy as np
import torch
import torch.nn.functional as F

def nonzero_region(image):
    valid = np.isfinite(image) & (image != 0)
    ys, xs = (np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0)))
    if not len(ys):
        raise ValueError('Cannot extract a nonzero ROI from an empty slice.')
    return (valid, (int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1))

def prepare_image(image, size=256, percentiles=(0.5, 99.0), return_mask=False):
    """Nonzero bbox -> percentile normalization -> aspect resize -> center pad.

    Needs only this image: no paired modality, labels, patient-wide window or
    precomputed validity. The returned tensor is 3 x size x size in [-1, 1].
    """
    image = np.asarray(image, dtype=np.float32)
    valid, (y0, y1, x0, x1) = nonzero_region(image)
    crop, mask = (image[y0:y1, x0:x1], valid[y0:y1, x0:x1])
    low, high = np.percentile(crop[mask], percentiles)
    normalized = np.zeros(crop.shape, dtype=np.float32)
    normalized[mask] = np.clip((crop[mask] - low) / max(float(high - low), 9.5e-07), 0, 1)
    h, w = crop.shape
    factor = size / max(h, w)
    target = (max(1, round(h * factor)), max(1, round(w * factor)))
    resized = F.interpolate(torch.from_numpy(normalized)[None, None], size=target, mode='bilinear', align_corners=False, antialias=True)[0]
    resized.clamp_(0, 1)
    resized_mask = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], size=target, mode='nearest')[0].bool()
    resized.masked_fill_(~resized_mask, 0)
    canvas = torch.zeros((1, size, size), dtype=torch.float32)
    top, left = ((size - target[0]) // 2, (size - target[1]) // 2)
    canvas[:, top:top + target[0], left:left + target[1]] = resized
    if return_mask:
        canvas_mask = torch.zeros_like(canvas, dtype=torch.bool)
        canvas_mask[:, top:top + target[0], left:left + target[1]] = resized_mask
        return (canvas, canvas_mask)
    return canvas.repeat(3, 1, 1) * 2 - 1
