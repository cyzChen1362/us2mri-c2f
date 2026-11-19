import numpy as np
import torch
from .data import map_points

def query_points(data, k=64, seed=40):
    """Equal-area spatial strata, gradient preference within each occupied cell.
    Recursively split the most populated stratum until K nonempty strata remain.
    All coordinates used by matching capacity are local US coordinates, not GT.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.where(data['mask'])
    xy = np.c_[xs, ys]
    if len(xy) < k:
        raise ValueError('Not enough valid pixels for requested query count')
    strata = [np.arange(len(xy))]
    while len(strata) < k:
        j = max(range(len(strata)), key=lambda j: len(strata[j]))
        ids = strata.pop(j)
        axis = np.argmax(np.ptp(xy[ids], axis=0))
        order = ids[np.argsort(xy[ids, axis], kind='stable')]
        m = len(order) // 2
        strata.extend([order[:m], order[m:]])
    central = data.get('us_center', data['us'])
    gy, gx = np.gradient(central[0, 0].numpy())
    grad = np.hypot(gx[ys, xs], gy[ys, xs])
    floor = max(grad.mean(), 1.4e-06)
    chosen = [rng.choice(ids, p=(grad[ids] + floor) / (grad[ids] + floor).sum()) for ids in strata]
    return xy[chosen].astype(np.float32)

def sample_points(data, k=64, seed=32, negative_pool=None):
    rng = np.random.default_rng(seed)
    uv = query_points(data, k, seed)
    points = np.c_[uv, np.zeros(k)]
    xyz = map_points(data['input_to_roi'], points)
    mm = map_points(data['input_to_native'], points)
    pool = np.argwhere(data['valid'])[:, ::-1].copy() if negative_pool is None else negative_pool
    neg = pool[rng.choice(len(pool), min(4096, len(pool)), replace=False)]
    nmm = map_points(data['roi_to_native'], neg)
    return [torch.tensor(a, dtype=torch.float32) for a in (uv, xyz, neg, mm, nmm)]
