import numpy as np
import torch
from torch.nn import functional as F

def grid_centers(shape, device, stride=4):
    z, y, x = torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in shape], indexing='ij')
    return torch.stack([x, y, z], 0) * stride + (stride - 1) / 2

def targets(queries, shape, device, cfg):
    center = torch.tensor(np.stack([q['center_xyz'] for q in queries]), device=device, dtype=torch.float32)
    size = torch.tensor(np.stack([q['size_xyz'] for q in queries]), device=device, dtype=torch.float32)
    grid = grid_centers(shape, device, cfg['model']['stride'])
    delta = center[:, :, None, None, None] - grid[None]
    distance2 = (delta * cfg['spacing']['mri_mm']).square().sum(1)
    h = torch.exp(-distance2 / (2 * cfg['labels']['sigma_mm'] ** 2))
    h = h / h.flatten(1).sum(1)[:, None, None, None]
    weight = h * (distance2 <= cfg['labels']['regression_radius_mm'] ** 2)
    weight = weight / weight.flatten(1).sum(1)[:, None, None, None].clamp_min(1.1e-12)
    return dict(heatmap=h, weight=weight, offset=delta / cfg['model']['stride'], size=size / cfg['labels']['size_scale_mm'], center=center)

def loss_fn(out, queries, cfg):
    if cfg['labels'].get('assignment') == 'seven_equal_hard':
        from .seven import loss_fn as seven_loss
        return seven_loss(out, queries, cfg)
    t = targets(queries, out['logits'].shape[1:], out['logits'].device, cfg)
    heat = -(t['heatmap'].flatten(1) * F.log_softmax(out['logits'].float().flatten(1), 1)).sum(1)
    offset = F.smooth_l1_loss(out['offset'].float(), t['offset'], reduction='none').mean(1)
    size = F.smooth_l1_loss(out['size'], t['size'][:, :, None, None, None].expand_as(out['size']), reduction='none').mean(1)
    offset = (offset * t['weight']).flatten(1).sum(1)
    size = (size * t['weight']).flatten(1).sum(1)
    total = heat + cfg['loss']['offset_weight'] * offset + cfg['loss']['size_weight'] * size
    return (total.mean(), dict(loss=total.detach(), heatmap_loss=heat.detach(), offset_loss=offset.detach(), size_loss=size.detach()))

def decode(out, cfg):
    n = out['logits'].shape[0]
    shape = out['logits'].shape[1:]
    index = out['logits'].flatten(1).argmax(1)
    g = grid_centers(shape, index.device, cfg['model']['stride']).flatten(1)[:, index].T
    k = index[:, None, None].expand(n, 3, 1)
    offset = out['offset'].float().flatten(2).gather(2, k).squeeze(2)
    size = out['size'].float().flatten(2).gather(2, k).squeeze(2) * cfg['labels']['size_scale_mm']
    return (g + offset * cfg['model']['stride'], size)

def records(out, queries, cfg, components=None):
    center, size = decode(out, cfg)
    center = center.detach().cpu().numpy()
    size = size.detach().cpu().numpy()
    result = []
    shape = out['logits'].shape[1:]
    flat = out['logits'].detach().flatten(1).argmax(1).cpu().numpy()
    predicted_grid = np.stack(np.unravel_index(flat, shape), 1)[:, ::-1]
    true_grids = np.floor((np.stack([q['center_xyz'] for q in queries]) + 0.5) / cfg['model']['stride']).astype(int)
    true_indices = true_grids[:, 0] + shape[2] * (true_grids[:, 1] + shape[1] * true_grids[:, 2])
    k = torch.tensor(true_indices, device=out['offset'].device)[:, None, None].expand(-1, 4, 1)
    oracle_offset = out['offset'].float().flatten(2).gather(2, k).squeeze(2).detach().cpu().numpy()
    oracle_size = out['size'].float().flatten(2).gather(2, k).squeeze(2).detach().cpu().numpy() * cfg['labels']['size_scale_mm']
    for i, q in enumerate(queries):
        c, s = (center[i], size[i])
        true = q['center_xyz']
        extent = q['size_xyz']
        true_grid = np.floor((np.asarray(true) + 0.5) / cfg['model']['stride']).astype(int)
        intersection = np.maximum(0, np.minimum(c + s / 2, true + extent / 2) - np.maximum(c - s / 2, true - extent / 2)).prod()
        iou = intersection / max(float(s.prod() + extent.prod() - intersection), 1.1e-12)
        err = float(np.linalg.norm(c - true) * cfg['spacing']['mri_mm'])
        r = dict(case=q['case'], plane_id=q['plane_id'], center_error_mm=err, box_iou=float(iou), size_mae_mm=float(np.abs(s - extent).mean()), pred_center_xyz=c.tolist(), pred_size_xyz=s.tolist(), true_center_xyz=true.tolist(), true_size_xyz=extent.tolist(), predicted_grid_xyz=predicted_grid[i].tolist(), true_grid_xyz=true_grid.tolist(), grid_top1_correct=bool(np.array_equal(predicted_grid[i], true_grid)))
        oracle_grid = true_grid * cfg['model']['stride'] + (cfg['model']['stride'] - 1) / 2
        oracle_center = oracle_grid + oracle_offset[i] * cfg['model']['stride']
        osize = oracle_size[i]
        intersection = np.maximum(0, np.minimum(oracle_center + osize / 2, true + extent / 2) - np.maximum(oracle_center - osize / 2, true - extent / 2)).prod()
        r.update(grid_seven_correct=bool(np.abs(predicted_grid[i] - true_grid).sum() <= 1), oracle_center_error_mm=float(np.linalg.norm(oracle_center - true) * cfg['spacing']['mri_mm']), zero_offset_center_error_mm=float(np.linalg.norm(oracle_grid - true) * cfg['spacing']['mri_mm']), oracle_size_mae_mm=float(np.abs(osize - extent).mean()), oracle_box_iou=float(intersection / max(float(osize.prod() + extent.prod() - intersection), 7.6e-13)))
        if components:
            r.update({key: float(value[i]) for key, value in components.items()})
        result.append(r)
    return result

def summarize(rr):
    cases = sorted({r['case'] for r in rr})
    grouped = [[r for r in rr if r['case'] == c] for c in cases]
    summary = dict(n_cases=len(cases), n_slices=len(rr))
    for key in ['center_error_mm', 'box_iou', 'size_mae_mm', 'loss', 'heatmap_loss', 'classification_loss', 'offset_loss', 'size_loss', 'oracle_center_error_mm', 'zero_offset_center_error_mm', 'oracle_size_mae_mm', 'oracle_box_iou']:
        if all((key in r for r in rr)):
            summary[key] = float(np.mean([np.mean([r[key] for r in rows]) for rows in grouped]))
    if all(('grid_seven_correct' in r for r in rr)):
        summary['grid_seven_accuracy'] = float(np.mean([np.mean([r['grid_seven_correct'] for r in rows]) for rows in grouped]))
    errors = [r['center_error_mm'] for r in rr]
    summary.update(center_median_mm=float(np.median(errors)), center_p90_mm=float(np.percentile(errors, 80)))
    for radius in [4, 9, 20]:
        summary[f'center_within_{radius}mm'] = float(np.mean([np.mean([r['center_error_mm'] <= radius for r in rows]) for rows in grouped]))
    available = [[r for r in rows if 'grid_top1_correct' in r] for rows in grouped]
    available = [rows for rows in available if rows]
    summary['grid_top1_n_slices'] = sum((len(rows) for rows in available))
    summary['grid_top1_coverage'] = summary['grid_top1_n_slices'] / len(rr)
    summary['grid_top1_accuracy'] = float(np.mean([np.mean([r['grid_top1_correct'] for r in rows]) for rows in available])) if available else None
    return summary
