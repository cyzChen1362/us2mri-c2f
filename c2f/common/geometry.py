import numpy as np
from .preprocess import nonzero_region

def roi_transform(image, size=256):
    _, (y0, y1, x0, x1) = nonzero_region(image)
    h, w = (y1 - y0, x1 - x0)
    scale = size / max(h, w)
    rh, rw = (max(1, round(h * scale)), max(1, round(w * scale)))
    top, left = ((size - rh) // 2, (size - rw) // 2)
    a = np.array([[rw / w, 0, left - x0 * rw / w], [0, rh / h, top - y0 * rh / h], [0, 0, 1]], dtype=np.float64)
    return (a, np.array([x0, y0, x1, y1], dtype=np.float64))

def box_from_window(window, matrix, size=256):
    p = np.array([[window[0], window[1], 1], [window[2], window[3], 1]])
    q = (matrix @ p.T).T[:, :2] / size
    return np.r_[(q[0] + q[1]) / 2, q[1] - q[0]].astype(np.float32)

def expand_grid(us, mr, pull):
    """Expand XY on integer US indices to enclose the transformed MR FOV.

    pull maps output US physical points to input MR physical points, as passed
    to SimpleITK.Resample. The inverse is used only to bound the MR extent.
    """
    import itertools
    import SimpleITK as sitk
    inverse = pull.GetInverse()
    points = []
    for idx in itertools.product(*[(-0.5, n - 0.5) for n in mr.GetSize()]):
        physical = mr.TransformContinuousIndexToPhysicalPoint(idx)
        points.append(us.TransformPhysicalPointToContinuousIndex(inverse.TransformPoint(physical)))
    points = np.array(points)
    lo = np.minimum(np.floor(points[:, :2].min(0)).astype(int) - 2, 0)
    hi = np.maximum(np.ceil(points[:, :2].max(0)).astype(int) + 3, np.array(us.GetSize()[:2]))
    grid = sitk.Image([int(hi[0] - lo[0]), int(hi[1] - lo[1]), us.GetSize()[2]], sitk.sitkFloat32)
    grid.SetSpacing(us.GetSpacing())
    grid.SetDirection(us.GetDirection())
    grid.SetOrigin(us.TransformContinuousIndexToPhysicalPoint((float(lo[0]), float(lo[1]), 0.0)))
    return (grid, (-lo).tolist())
