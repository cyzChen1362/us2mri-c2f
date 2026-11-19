"""Resumable same-case PixelVoxel trainer. No early stopping or subsampled evaluation."""
import argparse, csv, fcntl, json, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
from .data import prepare
from .sampling import sample_points
from .model import PixelVoxel
from .grouped import grouped_backward
from .evaluation import evaluate_slice, validation_score, save_best
from c2f.common.paths import compute_only, case_cache_path
STOP = 0

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def plane_ids(cfg, case):
    info = json.loads((case_cache_path(cfg, case) / 'source.json').read_text())
    return [r['plane_id'] for r in json.loads((Path(info['meta']) / 'planes.json').read_text())]

def epoch_plan(cfg, split, epoch):
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], epoch, 1064]))
    cases = list(split['train'])
    rng.shuffle(cases)
    groups = []
    for case in cases:
        ids = plane_ids(cfg, case)[(epoch - 1) % 4::4]
        rng.shuffle(ids)
        groups.extend([(case, ids[i:i + 32]) for i in range(0, len(ids), 32)])
    return groups

def load_checkpoint(path, model, opt):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    opt.load_state_dict(saved['optimizer'])
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value) and key != 'step':
                state[key] = value.cuda()
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), saved['model'][key]), ('checkpoint model mismatch', key)
    restored = opt.state_dict()
    for index, values in saved['optimizer']['state'].items():
        for key, value in values.items():
            actual = restored['state'][index][key]
            assert torch.equal(actual.cpu(), value.cpu()) if torch.is_tensor(value) else actual == value
    print(json.dumps(dict(event='checkpoint_load_verified', global_step=saved['state']['global_step'])), flush=True)
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    return saved['state']

def checkpoint(path, model, opt, state):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), state=state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state()), tmp)
    tmp.replace(path)

def run(cfg, out, stop_after_updates=None):
    global STOP
    STOP = 0
    compute_only()
    torch.set_num_threads(5)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)

    def handler(sig, *_):
        global STOP
        STOP = 63 if sig == signal.SIGUSR1 else 156
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, handler)
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model = PixelVoxel(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    latest = out / 'checkpoints/latest.pt'
    split = cfg.get('test_split_override') or json.loads(Path(cfg['data']['split_file']).read_text())
    state = dict(epoch=1, phase='train', cursor=0, global_step=0, best=None, train_loss_sum=0.0, train_queries=0, val_records=[], val_case=0, val_cursor=0, train_seconds=0.0, val_seconds=0.0)
    if latest.exists():
        state = load_checkpoint(latest, model, opt)
    if state['phase'] == 'validation' and any(('infonce' not in r for r in state['val_records'])):
        state.update(val_case=0, val_cursor=0, val_records=[], val_seconds=0.0)
        print(json.dumps(dict(event='validation_restart_for_new_infonce', epoch=state['epoch'])), flush=True)
    atomic_json(out / 'config.json', cfg)

    def save():
        checkpoint(latest, model, opt, state)

    def status(**kwargs):
        atomic_json(out / 'status.json', dict(epoch=state['epoch'], phase=state['phase'], global_step=state['global_step'], job_id=os.environ.get('SLURM_JOB_ID'), updated=time.time(), **kwargs))

    def should_stop():
        global STOP
        if time.time() > float(os.environ.get('PIXELVOXEL_DEADLINE', 'inf')):
            STOP = 71
        if STOP:
            save()
            status(state='checkpointed', exit_code=STOP)
            return STOP
        return 0
    try:
        while state['epoch'] <= cfg['training']['max_epochs']:
            epoch = state['epoch']
            plan = epoch_plan(cfg, split, epoch)
            if cfg.get('test_group_limit'):
                plan = plan[:cfg['test_group_limit']]
            if state['phase'] == 'train':
                model.train()
                cache = {}
                current = None
                MRI = None
                last_save = time.time()
                for index in range(state['cursor'], len(plan)):
                    if should_stop():
                        return STOP
                    case, ids = plan[index]
                    start = time.perf_counter()
                    if case != current:
                        cache = {}
                        current = case
                        MRI = None
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    queries = []
                    for k in ids:
                        d = prepare(cfg, case, k, volume_cache=cache)
                        if MRI is None:
                            MRI = d['mri'].cuda()
                        seed = int(np.random.SeedSequence([cfg['seed'], epoch, int(case.removeprefix('Case')), k]).generate_state(1)[0])
                        pts = sample_points(d, seed=seed)
                        queries.append((d['us'] if cfg.get('model', {}).get('us_input') == 'adjacent_stack' else d['us'].cuda(), [x.cuda() for x in pts]))
                    opt.zero_grad(set_to_none=True)
                    loss, timing = grouped_backward(model, MRI, queries)
                    if not np.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss: {case}, epoch {epoch}')
                    opt.step()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    state['cursor'] = index + 1
                    state['global_step'] += 1
                    state['train_loss_sum'] += loss * len(ids)
                    state['train_queries'] += len(ids)
                    state['train_seconds'] += elapsed
                    status(state='running', case=case, group=index + 1, groups=len(plan), last_loss=loss, group_seconds=elapsed, query_count=len(ids), sampling_offset=(epoch - 1) % 5)
                    print(json.dumps(dict(event='train_group', epoch=epoch, group=index + 1, total_groups=len(plan), case=case, n=len(ids), loss=loss, seconds=elapsed)), flush=True)
                    if time.time() - last_save >= 71 or index + 1 == len(plan):
                        save()
                        last_save = time.time()
                    del queries, d, pts
                    if stop_after_updates and state['global_step'] >= stop_after_updates:
                        save()
                        status(state='test_stop')
                        return 0
                del MRI
                cache = {}
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                state['phase'] = 'validation'
                state['val_case'] = 0
                state['val_cursor'] = 0
                state['val_records'] = []
                save()
            if state['phase'] == 'validation':
                model.eval()
                for ci in range(state['val_case'], len(split['validation'])):
                    case = split['validation'][ci]
                    ids = plane_ids(cfg, case)
                    if cfg.get('test_validation_limit'):
                        ids = ids[:cfg['test_validation_limit']]
                    cache = {}
                    features = None
                    for si in range(state['val_cursor'], len(ids)):
                        if should_stop():
                            return STOP
                        start = time.perf_counter()
                        k = ids[si]
                        d = prepare(cfg, case, k, volume_cache=cache)
                        if features is None:
                            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                                features = model.mri(d['mri'].cuda())
                        record, _ = evaluate_slice(model, d, seed=cfg['validation']['fixed_query_seed'], capacity=cfg['matching']['capacity'], mri_features=features)
                        record.update(case=case, plane_id=k)
                        state['val_records'].append(record)
                        state['val_cursor'] = si + 1
                        state['val_seconds'] += time.perf_counter() - start
                        if (si + 1) % 27 == 0 or si + 1 == len(ids):
                            save()
                            status(state='running', case=case, validation_case=ci + 1, validation_cases=len(split['validation']), validation_slice=si + 1, validation_slices=len(ids))
                            print(json.dumps(dict(event='validation_progress', epoch=epoch, case=case, slices=si + 1, total=len(ids))), flush=True)
                    del features
                    cache = {}
                    state['val_case'] = ci + 1
                    state['val_cursor'] = 0
                    save()
                score = validation_score(state['val_records'])
                selected = save_best(out / 'checkpoints/best.pt', model, epoch, score, state['best'], cfg)
                if selected:
                    state['best'] = dict(score, epoch=epoch)
                metric = dict(epoch=epoch, global_step=state['global_step'], train_loss=state['train_loss_sum'] / state['train_queries'], train_US_count=state['train_queries'], train_seconds=state['train_seconds'], validation_seconds=state['val_seconds'], best_epoch=state['best']['epoch'], val_infonce=score['infonce'], **score)
                atomic_json(out / f'validation/epoch_{epoch:03d}.json', dict(summary=metric, slices=state['val_records']))
                history = []
                for f in sorted((out / 'validation').glob('epoch_*.json')):
                    history.append(json.loads(f.read_text())['summary'])
                with (out / 'metrics.tmp').open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(metric))
                    writer.writeheader()
                    writer.writerows(history)
                (out / 'metrics.tmp').replace(out / 'metrics.csv')
                print(json.dumps(dict(event='epoch_complete', **metric)), flush=True)
                state.update(epoch=epoch + 1, phase='train', cursor=0, val_case=0, val_cursor=0, val_records=[], train_loss_sum=0.0, train_queries=0, train_seconds=0.0, val_seconds=0.0)
                save()
        status(state='complete')
        atomic_json(out / 'training_complete.json', dict(epochs=cfg['training']['max_epochs'], best=state['best']))
        return 0
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stop-after-updates', type=int)
    a = p.parse_args()
    return run(json.loads(Path(a.config).read_text()), a.output, a.stop_after_updates)
if __name__ == '__main__':
    raise SystemExit(main())
