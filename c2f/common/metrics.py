import numpy as np

def accumulate(totals, metrics, data, indices, valid):
    values = {k: v.detach().float().cpu().numpy() for k, v in metrics.items()}
    for i, index in enumerate(indices):
        if not valid[i]:
            continue
        case = data.rows[int(index)]['case']
        entry = totals.setdefault(case, {'n': 0, 'sums': {}})
        entry['n'] += 1
        for k, v in values.items():
            entry['sums'][k] = entry['sums'].get(k, 0.0) + float(v[i])

def summarize(parts):
    merged = {}
    for part in parts:
        for case, entry in part.items():
            dst = merged.setdefault(case, {'n': 0, 'sums': {}})
            dst['n'] += entry['n']
            for k, v in entry['sums'].items():
                dst['sums'][k] = dst['sums'].get(k, 0.0) + v
    cases = [dict(case=case, slices=v['n'], **{k: s / v['n'] for k, s in v['sums'].items()}) for case, v in sorted(merged.items())]
    if not cases:
        raise ValueError('No metrics')
    summary = {k: float(np.mean([c[k] for c in cases])) for k in cases[0] if k not in ['case', 'slices']}
    summary.update(cases=len(cases), slices=sum((c['slices'] for c in cases)))
    return (summary, cases)
