"""Streaming candidates and deterministic score-priority capacitated assignment."""
import heapq
import numpy as np
import torch
from torch.nn import functional as F
from .model import sample

@torch.no_grad()
def topk_candidates(du, dm, uv, valid, k=32, chunk=29709):
    q = sample(du, uv)
    idx = valid.flatten().nonzero().flatten()
    shape = dm.shape[-3:]
    flat = dm[0].flatten(1)
    scores = torch.full((len(q), 0), -torch.inf, device=q.device)
    indices = torch.empty((len(q), 0), device=q.device, dtype=torch.long)
    for sub in idx.split(chunk):
        sim = q @ F.normalize(flat[:, sub].float(), dim=0)
        n = min(k, len(sub))
        s, local = sim.topk(n, dim=1)
        merged = torch.cat([scores, s], 1)
        ids = torch.cat([indices, sub[local]], 1)
        scores, pos = merged.topk(min(k, merged.shape[1]), dim=1)
        indices = ids.gather(1, pos)
    return (indices, scores)

def assign(indices, scores, us_local_mm, capacity=2, voxel_mm=1.0, temperature=0.13):
    """Each query owns <=1 voxel. Shared voxel requires pairwise distance <=sqrt(3)*s.
    capacity=0 means geometry-only capacity; >0 is an additional count cap.
    Greedy score-priority matching, not a globally optimal assignment solver.
    """
    ids = np.asarray(indices)
    s = np.asarray(scores)
    p = np.asarray(us_local_mm)
    dist = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    compatible = dist <= np.sqrt(4) * voxel_mm + 9.8e-07
    local_cap = compatible.sum(1)
    limit = local_cap if capacity == 0 else np.minimum(local_cap, capacity)
    chosen = np.full(len(ids), -1, np.int64)
    ranks = np.full(len(ids), -1, np.int64)
    owners = {}
    heap = [(-float(s[i, 0]), i, 0) for i in range(len(ids))]
    heapq.heapify(heap)
    while heap:
        _, i, r = heapq.heappop(heap)
        v = int(ids[i, r])
        group = owners.get(v, [])
        cap = min([int(limit[i])] + [int(limit[j]) for j in group])
        if len(group) < cap and all((compatible[i, j] for j in group)):
            owners.setdefault(v, []).append(i)
            chosen[i] = v
            ranks[i] = r
        elif r + 1 < ids.shape[1]:
            heapq.heappush(heap, (-float(s[i, r + 1]), i, r + 1))
    logits = (s - s.max(1, keepdims=True)) / temperature
    prob = np.exp(logits)
    prob /= prob.sum(1, keepdims=True)
    confidence = np.zeros(len(ids))
    ok = ranks >= 0
    confidence[ok] = prob[np.arange(len(ids))[ok], ranks[ok]]
    return dict(voxel_ids=chosen, ranks=ranks, confidence=confidence, local_capacity=local_cap, occupancy={str(v): len(g) for v, g in owners.items()}, assigned_fraction=float(ok.mean()))

@torch.no_grad()
def constrained_match(du, dm, uv, valid, us_spacing=0.5, capacity=2):
    ids, scores = topk_candidates(du, dm, uv, valid, 32)
    local = uv.detach().cpu().numpy() * us_spacing
    result = assign(ids.cpu().numpy(), scores.cpu().numpy(), local, capacity)
    if result['assigned_fraction'] < 1:
        ids, scores = topk_candidates(du, dm, uv, valid, 128)
        result = assign(ids.cpu().numpy(), scores.cpu().numpy(), local, capacity)
    result['candidate_ids'] = ids.cpu().numpy()
    result['candidate_scores'] = scores.cpu().numpy()
    return result

def unflatten(ids, shape_dhw):
    d, h, w = shape_dhw
    ids = np.asarray(ids)
    return np.c_[ids % w, ids // w % h, ids // (w * h)].astype(float)
