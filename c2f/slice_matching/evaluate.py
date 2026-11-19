from pathlib import Path
import json
import numpy as np
import torch
from c2f.common.runtime import CFG, OUT, initialize, write_json, csv_write, stopping
from c2f.slice_matching.dataset import Dataset
from c2f.slice_matching.networks import Matcher
from c2f.slice_matching.train import evaluate_case

def main(checkpoint=None, output=None, splits=None):
    initialize()
    out = Path(output) if output is not None else OUT / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint) if checkpoint is not None else OUT / 'checkpoints/best.pt'
    splits = splits or ['train', 'val', 'test']
    if (out / 'complete.json').exists():
        return 0
    p = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = Matcher().cuda().eval()
    model.load_state_dict(p['model'])
    data = Dataset()
    summary = []
    for c in data.cases:
        if data.splits[c] not in splits:
            continue
        f = out / c
        f.mkdir(exist_ok=True)
        record = f / 'metrics.json'
        if record.exists():
            summary.append(json.loads(record.read_text()))
            continue
        if stopping():
            return stopping()
        case = data.load(c)
        try:
            matrix, v = evaluate_case(model, case, stopping, f / 'queries.pt')
        except InterruptedError:
            return stopping() or 75
        scores = matrix.cpu().numpy()
        indices = case.indices.cpu().numpy()
        candidates = case.candidate_indices.cpu().numpy()
        np.savez_compressed(f / 'scores.npz', scores=scores, slice_indices=indices, query_indices=indices, candidate_indices=candidates)
        order = np.argsort(-scores, axis=1, kind='stable')
        rr = []
        for i in range(case.n):
            rr.append(dict(case=c, query_slice=int(indices[i]), predicted_slice=int(candidates[order[i, 0]]), window_hit=int(abs(int(candidates[order[i, 0]]) - int(indices[i])) <= 5), top5_slices='|'.join(map(str, candidates[order[i, :5]]))))
        csv_write(f / 'retrieval.csv', rr)
        r = dict(case=c, split=data.splits[c], samples=case.n, **v)
        write_json(record, r)
        summary.append(r)
        del case
    csv_write(out / 'case_metrics.csv', summary)
    keys = ['loss', 'window_r1', 'window_r5', 'tolerance10_r1', 'tolerance10_r5', 'r1', 'r5', 'mean_absolute_slice_error', 'trajectory_arc_error_mm', 'plane_normal_error_deg'] + (['positive_recall', 'negative_recall'] if CFG['method'] == 'Classification' else [])
    result = {s: {k: float(np.mean([r[k] for r in summary if r['split'] == s])) for k in keys} for s in splits}
    write_json(out / 'summary.json', dict(best_epoch=p['epoch'], checkpoint=str(checkpoint), candidate_indices=CFG.get('mri_candidate_indices', list(range(200))), exact_r1_definition='exact original-index hit across all queries; absent indices count as misses', scores=result))
    write_json(out / 'complete.json', dict(cases=len(summary)))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
