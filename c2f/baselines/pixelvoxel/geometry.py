import numpy as np

def aggregate(points, weights):
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    w = np.zeros(len(unique))
    np.add.at(w, inverse, weights)
    return (unique, w)

def fit_plane(points, weights, iterations=30):
    """IRLS for weighted absolute plane distance (0.1 mm smooth floor)."""
    p, w = aggregate(np.asarray(points, float), np.asarray(weights, float))
    ok = w > 0
    p, w = (p[ok], w[ok])
    if len(p) < 3 or np.linalg.matrix_rank(p - p.mean(0), tol=1e-06) < 2:
        raise ValueError('Plane requires three noncollinear matches')
    base = w / w.sum()
    current = base
    for _ in range(iterations):
        center = (p * current[:, None]).sum(0) / current.sum()
        _, _, v = np.linalg.svd((p - center) * np.sqrt(current[:, None]), full_matrices=False)
        normal = v[-1]
        residual = np.abs((p - center) @ normal)
        current = base / np.maximum(residual, 0.1)
    return (center, normal)

def rank_planes(points, weights, centers, normals):
    normals = np.asarray(normals)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    distance = np.abs(np.einsum('pkj,pj->pk', np.asarray(points)[None] - np.asarray(centers)[:, None], normals))
    scores = distance @ np.asarray(weights) / np.sum(weights)
    return (np.argsort(scores), scores)

def plane_metrics(center, normal, true_points, true_normal):
    n = normal / np.linalg.norm(normal)
    g = true_normal / np.linalg.norm(true_normal)
    return dict(footprint_distance_mm=float(np.abs((true_points - center) @ n).mean()), angle_deg=float(np.degrees(np.arccos(np.clip(abs(n @ g), 0, 1)))), center_to_plane_mm=float(abs((true_points.mean(0) - center) @ n)))
