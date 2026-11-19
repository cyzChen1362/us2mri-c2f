"""Physical grids use voxel centers; all distances are native MRI LPS millimetres."""
import itertools
import numpy as np

def plane_grid_geometry(row, T, spacing, multiple=4):
    P = T @ np.asarray(row['US_pixel_to_LPS'])
    u = P[:3, 0] / np.linalg.norm(P[:3, 0])
    v = P[:3, 1] - u * np.dot(u, P[:3, 1])
    v /= np.linalg.norm(v)
    basis = np.column_stack([u, v, np.cross(u, v)])
    x0, y0, x1, y1 = row['US_bbox_xyxy']
    corners = np.array([[x - 0.5, y - 0.5, 0, 1] for x, y in itertools.product([x0, x1], [y0, y1])])
    local = (corners @ P[:3].T - P[:3, 3]) @ basis
    lo = local[:, :2].min(0)
    hi = local[:, :2].max(0)
    count = np.maximum(4, np.ceil((hi - lo) / spacing / multiple).astype(int) * multiple)
    out = np.eye(4)
    out[:3, :3] = basis * spacing
    out[:3, 3] = P[:3, 3] + basis[:, :2] @ ((lo + hi) / 2 - (count - 1) * spacing / 2)
    return (out, np.linalg.inv(P) @ out, count, hi - lo)

def roi_bounds(us_matrix, us_size, T, margin=5.0):
    corners = np.array([[*p, 1.0] for p in itertools.product(*[(-0.5, n - 0.5) for n in us_size])]) @ (T @ us_matrix)[:3].T
    return (corners.min(0) - margin, corners.max(0) + margin)

def roi_shape(lo, hi, spacing, multiple=4):
    return np.maximum(4, np.ceil((hi - lo) / spacing / multiple).astype(int) * multiple)
