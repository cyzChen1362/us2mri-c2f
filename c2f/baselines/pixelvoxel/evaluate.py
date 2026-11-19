"""Evaluate the selected validation-best checkpoint on every saved test slice."""
import argparse, json, os, signal, time
from pathlib import Path
import torch
import SimpleITK as sitk
from .data import prepare
from .model import PixelVoxel
from .train import plane_ids, atomic_json
from .evaluation import evaluate_slice, validation_score
from c2f.common.paths import compute_only
STOP = 0

def main():
    global STOP
    compute_only()
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--partition', choices=['test', 'validation'], default='test')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    run = Path(a.run)
    out = run / f'best_{a.partition}'
    out.mkdir(exist_ok=True)
    if (out / 'complete.json').exists() and 'infonce' in json.loads((out / 'complete.json').read_text()).get('summary', {}):
        return 0
    model = PixelVoxel(cfg).cuda()
    best = torch.load(run / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'])
    model.eval()

    def stop(sig, *_):
        global STOP
        STOP = 75 if sig == signal.SIGUSR1 else 143
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    split = json.loads(Path(cfg['data']['split_file']).read_text())
    allrecords = []
    for case in split[a.partition]:
        path = out / f'{case}.json'
        records = json.loads(path.read_text()) if path.exists() else []
        ids = plane_ids(cfg, case)
        records = [r for r in records if 'infonce' in r]
        done = {r['plane_id'] for r in records}
        cache = {}
        features = None
        for k in ids:
            if k in done:
                continue
            if STOP or time.time() > float(os.environ.get('PIXELVOXEL_DEADLINE', 'inf')):
                atomic_json(path, records)
                return STOP or 75
            d = prepare(cfg, case, k, volume_cache=cache)
            if features is None:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    features = model.mri(d['mri'].cuda())
            record, _ = evaluate_slice(model, d, seed=cfg['validation']['fixed_query_seed'], capacity=cfg['matching']['capacity'], mri_features=features)
            record.update(case=case, plane_id=k)
            records.append(record)
            if len(records) % 25 == 0:
                atomic_json(path, records)
        atomic_json(path, records)
        allrecords += records
        del features
    score = validation_score(allrecords)
    atomic_json(out / 'complete.json', dict(partition=a.partition, best_epoch=best['epoch'], summary=score, **{a.partition + '_infonce': score['infonce']}))
    print(json.dumps(dict(event='best_evaluation_complete', partition=a.partition, **score)), flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
