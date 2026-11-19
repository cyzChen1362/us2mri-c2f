"""Plane metrics and freshly sampled InfoNCE; test never chooses model."""
import numpy as np

def validation_score(records):
    cases = sorted(set((r['case'] for r in records)))
    dist = []
    angles = []
    failures = 0
    for case in cases:
        rr = [r for r in records if r['case'] == case]
        dd = []
        aa = []
        for r in rr:
            if not r.get('valid_plane', True):
                dd.append(float(r['roi_diagonal_mm']))
                aa.append(90.0)
                failures += 1
            else:
                dd.append(r['footprint_distance_mm'])
                aa.append(r['angle_deg'])
        dist.append(np.mean(dd))
        angles.append(np.mean(aa))
    score = dict(primary_case_mean_distance_mm=float(np.mean(dist)), secondary_case_mean_angle_deg=float(np.mean(angles)), failed_planes=failures, n_cases=len(cases), n_slices=len(records))
    losses = [r['infonce'] for r in records if 'infonce' in r]
    if losses:
        if len(losses) != len(records):
            raise ValueError('Mixed legacy and InfoNCE records: reevaluate the partition')
        score['infonce'] = float(np.mean(losses))
        score['case_mean_infonce'] = float(np.mean([np.mean([r['infonce'] for r in records if r['case'] == c]) for c in cases]))
        score['infonce_n_slices'] = len(losses)
    for key in ['sampled_top1'] + [f'{prefix}_{r}mm' for prefix in ['point_accuracy', 'raw_top1_accuracy'] for r in [3, 5, 10]]:
        if all((key in row for row in records)):
            score[key] = float(np.mean([np.mean([row[key] for row in records if row['case'] == c]) for c in cases]))
    return score

def better(new, old):
    if old is None:
        return True
    return (new['primary_case_mean_distance_mm'], new['secondary_case_mean_angle_deg']) < (old['primary_case_mean_distance_mm'], old['secondary_case_mean_angle_deg'])

def evaluate_slice(model, data, seed=42, capacity=2, mri_features=None):
    """Full-footprint physical evaluation, never labels from model query weighting."""
    import torch
    from .sampling import query_points
    from .data import map_points
    from .matching import constrained_match, unflatten
    from .geometry import fit_plane, plane_metrics
    device = next(model.parameters()).device
    uv = torch.from_numpy(query_points(data, seed=seed)).to(device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        du = model.us(data['us'].to(device))
        dm = model.mri(data['mri'].to(device)) if mri_features is None else mri_features
    match = constrained_match(du, dm, uv, torch.from_numpy(data['valid']).to(device), capacity=capacity)
    ok = match['voxel_ids'] >= 0
    points = map_points(data['roi_to_native'], unflatten(match['voxel_ids'][ok], data['valid'].shape))
    yy, xx = np.where(data['mask'])
    truth = map_points(data['input_to_native'], np.c_[xx, yy, np.zeros(len(xx))])
    normal = data['input_to_native'][:3, 2]
    diagonal = float(np.linalg.norm(data['mri_extent_mm']))
    loss, details = evaluation_infonce(du, dm, data, return_details=True)
    record = dict(valid_plane=True, roi_diagonal_mm=diagonal, assigned_fraction=match['assigned_fraction'], infonce=loss, **details)
    query_truth = map_points(data['input_to_native'], np.c_[uv.cpu().numpy(), np.zeros(len(uv))])
    errors = np.full(len(uv), np.inf)
    errors[ok] = np.linalg.norm(points - query_truth[ok], axis=1)
    raw = map_points(data['roi_to_native'], unflatten(match['candidate_ids'][:, 0], data['valid'].shape))
    raw_errors = np.linalg.norm(raw - query_truth, axis=1)
    for radius in [3, 5, 10]:
        record[f'point_accuracy_{radius}mm'] = float(np.mean(errors <= radius))
        record[f'raw_top1_accuracy_{radius}mm'] = float(np.mean(raw_errors <= radius))
    try:
        center, n = fit_plane(points, match['confidence'][ok])
        record.update(plane_metrics(center, n, truth, normal))
        record.update(center=center.tolist(), normal=n.tolist())
    except ValueError:
        record['valid_plane'] = False
    return (record, match)

def save_best(path, model, epoch, score, previous, config):
    """Select with fixed validation recipe. Caller also saves latest separately."""
    import torch
    from pathlib import Path
    if not better(score, previous):
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), epoch=epoch, validation=score, config=config), tmp)
    tmp.replace(path)
    return True
_NEGATIVE_POOL = None

def evaluation_infonce(du, dm, data, seed=None, return_details=False):
    """Fresh 64 US queries + 4096 uniform valid MRI negatives, training InfoNCE.
    None seeds OS entropy without advancing training global RNG states.
    Cache only the current volume's candidate coordinates, never sampled points.
    """
    import torch
    from .sampling import sample_points
    from .model import infonce
    global _NEGATIVE_POOL
    valid = data['valid']
    if _NEGATIVE_POOL is None or _NEGATIVE_POOL[0] is not valid:
        _NEGATIVE_POOL = (valid, np.argwhere(valid)[:, ::-1].copy())
    points = [p.to(du.device) for p in sample_points(data, seed=seed, negative_pool=_NEGATIVE_POOL[1])]
    with torch.no_grad():
        loss, details = infonce(du, dm, *points, return_details=True)
    return (float(loss), details) if return_details else float(loss)
