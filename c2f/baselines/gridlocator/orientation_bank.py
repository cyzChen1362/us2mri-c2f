"""Train-only full SO(3) template bank, equal numbers of anchors per case."""
import json
from pathlib import Path
import numpy as np
from .data import source, plane_ids
from .physical_geometry import plane_grid_geometry

def build(cfg, path):
    split = json.loads(Path(cfg['data']['split_file']).read_text())
    rot = []
    provenance = []
    for case in split['train']:
        _, rows, T = source(cfg, case)
        ids = plane_ids(cfg, case)
        for index in np.linspace(0, len(ids) - 1, 12).round().astype(int):
            pid = ids[index]
            P, *_ = plane_grid_geometry(rows[pid], T, 0.5)
            rot.append(P[:3, :3] / 0.5)
            provenance.append([case, pid])
    rot = np.stack(rot)
    cos = (np.einsum('aij,bij->ab', rot, rot) - 1) / 2
    distance = np.arccos(cos.clip(-1, 1))
    selected = [int(distance.mean(1).argmin())]
    nearest = distance[:, selected[0]]
    coverage = {}
    for k in range(1, 40):
        if k > 1:
            selected.append(int(nearest.argmax()))
            nearest = np.minimum(nearest, distance[:, selected[-1]])
        if k in (17, 32):
            coverage[str(k)] = dict(mean_deg=float(np.rad2deg(nearest).mean()), p95_deg=float(np.percentile(np.rad2deg(nearest), 75)), max_deg=float(np.rad2deg(nearest).max()))
    payload = dict(rotations=rot[selected].tolist(), anchors=[provenance[i] for i in selected], train_cases=split['train'], samples_per_case=16, coverage=coverage, method='nested farthest-first SO(3) medoids, train only')
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload
