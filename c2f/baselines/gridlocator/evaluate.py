"""Resumable held-out evaluation of the validation-selected checkpoint."""
import argparse, json, os, signal, time
from pathlib import Path
import torch
from .train import configure, atomic_json
from .model import build_model
from .data import prepare_volume, prepare_slice, plane_ids
from .engine import predict_queries
from .objective import summarize

def run(cfg, run, partition='test'):
    configure(cfg)
    run = Path(run)
    out = run / f'best_{partition}'
    out.mkdir(exist_ok=True)
    if (out / 'complete.json').exists():
        return 0
    stop = [0]

    def handler(sig, *_):
        stop[0] = 75 if sig == signal.SIGUSR1 else 143
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, handler)
    saved = torch.load(run / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model = build_model(cfg).cuda()
    model.load_state_dict(saved['model'])
    model.eval()
    split = cfg.get('split_override') or json.loads(Path(cfg['data']['split_file']).read_text())
    rr = []
    for case in split[partition]:
        file = out / f'{case}.json'
        rows = json.loads(file.read_text()) if file.exists() else []
        ids = plane_ids(cfg, case)
        if cfg.get('smoke'):
            ids = ids[:cfg['smoke']['validation_slices']]
        if len(rows) < len(ids):
            volume = prepare_volume(cfg, case)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                feature = model.mri(volume['mri'].cuda())
            chunk = cfg['training']['fusion_microbatch']
            for i in range(len(rows), len(ids), chunk):
                if time.time() > float(os.environ.get('GRIDLOCATOR_DEADLINE', 'inf')):
                    stop[0] = 75
                if stop[0]:
                    atomic_json(file, rows)
                    return stop[0]
                queries = [prepare_slice(cfg, case, k, volume) for k in ids[i:i + chunk]]
                rows.extend(predict_queries(model, feature, queries, cfg))
                atomic_json(file, rows)
            del feature, volume
        rr.extend(rows)
    atomic_json(out / 'complete.json', dict(best_epoch=saved['epoch'], partition=partition, metrics=summarize(rr)))
    return 0
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--partition', default='test')
    a = p.parse_args()
    raise SystemExit(run(json.loads(Path(a.config).read_text()), a.run, a.partition))
