"""Resumable GridLocator training; validation selects by case-mean center error."""
import argparse, csv, fcntl, json, os, random, signal, time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from c2f.common.paths import compute_only
from .data import plane_ids, prepare_volume, prepare_slice
from .model import build_model
from .engine import grouped_backward, predict_queries
from .objective import summarize
STOP = 0

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def checkpoint(path, model, opt, state):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), state=state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state()), tmp)
    tmp.replace(path)

def restore(path, model, opt):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    return saved['state']

def epoch_plan(cfg, split, epoch):
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], epoch, 828]))
    cases = list(split['train'])
    rng.shuffle(cases)
    groups = []
    stride = cfg['training']['slice_stride']
    n = cfg['training']['max_US_per_update']
    for case in cases:
        ids = plane_ids(cfg, case)[(epoch - 1) % stride::stride]
        rng.shuffle(ids)
        groups.extend(((case, ids[i:i + n]) for i in range(0, len(ids), n)))
    return groups[:cfg['smoke']['groups']] if cfg.get('smoke') else groups

def configure(cfg):
    compute_only()
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

def run(cfg, out, stop_after_updates=None):
    global STOP
    STOP = 0
    configure(cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)

    def handler(sig, *_):
        global STOP
        STOP = 93 if sig == signal.SIGUSR1 else 137
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, handler)
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model = build_model(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    latest = out / 'checkpoints/latest.pt'
    split = cfg.get('split_override') or json.loads(Path(cfg['data']['split_file']).read_text())
    state = dict(epoch=1, phase='train', cursor=0, global_step=0, best=None, train_records=[], val_records=[], val_case=0, val_cursor=0, train_seconds=0.0, val_seconds=0.0)
    if latest.exists():
        state = restore(latest, model, opt)
    atomic_json(out / 'config.json', cfg)

    def save():
        checkpoint(latest, model, opt, state)

    def status(**kwargs):
        atomic_json(out / 'status.json', dict(epoch=state['epoch'], phase=state['phase'], global_step=state['global_step'], best=state['best'], job_id=os.environ.get('SLURM_JOB_ID'), updated=time.time(), **kwargs))

    def should_stop():
        global STOP
        if time.time() > float(os.environ.get('GRIDLOCATOR_DEADLINE', 'inf')):
            STOP = 65
        if STOP:
            save()
            status(state='checkpointed', exit_code=STOP)
            return STOP
        return 0
    try:
        while state['epoch'] <= cfg['training']['max_epochs']:
            epoch = state['epoch']
            plan = epoch_plan(cfg, split, epoch)
            if state['phase'] == 'train':
                model.train()
                current = None
                volume = None
                mri = None
                native = None
                last_save = time.time()
                augmented = cfg['model'].get('architecture') == 'spatial_template_v3'
                from .execution import GroupPrefetcher, enabled
                optimized = enabled(cfg)
                with GroupPrefetcher(cfg, plan, epoch, state['cursor']) if optimized else nullcontext(None) as prefetch:
                    for index in range(state['cursor'], len(plan)):
                        if should_stop():
                            return STOP
                        case, ids = plan[index]
                        start = time.perf_counter()
                        if case != current:
                            opt.zero_grad(set_to_none=True)
                            mri = None
                            volume = None
                            torch.cuda.empty_cache()
                            if augmented and (not optimized):
                                from .augmentation import load_case
                                native = load_case(cfg, case)
                            elif not augmented:
                                volume = prepare_volume(cfg, case)
                                mri = volume['mri'].cuda()
                            current = case
                        if prefetch is not None:
                            prepared = prefetch.take(index)
                            volume = prepared.volume
                            qq = prepared.queries
                            mri = volume['mri'].to('cuda', non_blocking=True)
                            model.mri.set_frame(volume['frame_rotation'])
                            del prepared
                        else:
                            if augmented:
                                from .augmentation import prepare_group
                                volume = prepare_group(cfg, case, ids, native, epoch, index)
                                mri = volume['mri'].cuda()
                                model.mri.set_frame(volume['frame_rotation'])
                            qq = [prepare_slice(cfg, case, k, volume) for k in ids]
                        opt.zero_grad(set_to_none=True)
                        rr = grouped_backward(model, mri, qq, cfg)
                        if not all((np.isfinite(r['loss']) for r in rr)):
                            raise FloatingPointError(f'Nonfinite loss {case}')
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'], error_if_nonfinite=True)
                        opt.step()
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        state['cursor'] = index + 1
                        state['global_step'] += 1
                        state['train_records'].extend(rr)
                        state['train_seconds'] += elapsed
                        info = dict(case=case, group=index + 1, groups=len(plan), n=len(ids), loss=float(np.mean([r['loss'] for r in rr])), center_error_mm=float(np.mean([r['center_error_mm'] for r in rr])), seconds=elapsed, gradient_norm=float(norm), peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 25)
                        if augmented:
                            info.update(augmentation=volume['augmentation'])
                        if optimized:
                            info['execution'] = 'fused_norm_zero_crop_cpu_prefetch'
                        status(state='running', **info)
                        print(json.dumps(dict(event='train_group', epoch=epoch, **info)), flush=True)
                        if time.time() - last_save >= 70 or index + 1 == len(plan):
                            save()
                            last_save = time.time()
                        del qq, rr
                        if stop_after_updates and state['global_step'] >= stop_after_updates:
                            save()
                            status(state='smoke_checkpoint')
                            return 0
                mri = None
                volume = None
                native = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                state.update(phase='validation', val_case=0, val_cursor=0, val_records=[])
                save()
            if state['phase'] == 'validation':
                model.eval()
                for ci in range(state['val_case'], len(split['validation'])):
                    if should_stop():
                        return STOP
                    case = split['validation'][ci]
                    ids = plane_ids(cfg, case)
                    if cfg.get('smoke'):
                        ids = ids[:cfg['smoke']['validation_slices']]
                    start = time.perf_counter()
                    volume = prepare_volume(cfg, case)
                    if cfg['model'].get('architecture') == 'spatial_template_v3':
                        model.mri.set_frame(np.eye(2))
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        feature = model.mri(volume['mri'].cuda())
                    state['val_seconds'] += time.perf_counter() - start
                    chunk = cfg['training']['fusion_microbatch']
                    last_save = time.time()
                    for si in range(state['val_cursor'], len(ids), chunk):
                        if should_stop():
                            return STOP
                        start = time.perf_counter()
                        qq = [prepare_slice(cfg, case, k, volume) for k in ids[si:si + chunk]]
                        state['val_records'].extend(predict_queries(model, feature, qq, cfg))
                        state['val_cursor'] = si + len(qq)
                        state['val_seconds'] += time.perf_counter() - start
                        status(state='running', case=case, validation_case=ci + 1, validation_cases=len(split['validation']), validation_slice=state['val_cursor'], validation_slices=len(ids))
                        if time.time() - last_save >= 75:
                            save()
                            last_save = time.time()
                    del feature, volume, qq
                    state.update(val_case=ci + 1, val_cursor=0)
                    save()
                    print(json.dumps(dict(event='validation_case_complete', epoch=epoch, case=case)), flush=True)
                score = summarize(state['val_records'])
                best = state['best']
                selected = best is None or (score['center_error_mm'], -score['box_iou']) < (best['center_error_mm'], -best['box_iou'])
                if selected:
                    state['best'] = dict(score, epoch=epoch)
                    tmp = out / 'checkpoints/best.tmp'
                    torch.save(dict(model=model.state_dict(), epoch=epoch, validation=score, config=cfg), tmp)
                    tmp.replace(out / 'checkpoints/best.pt')
                train = summarize(state['train_records'])
                metric = dict(epoch=epoch, global_step=state['global_step'], train_seconds=state['train_seconds'], validation_seconds=state['val_seconds'], best_epoch=state['best']['epoch'], **{'train_' + k: v for k, v in train.items()}, **{'val_' + k: v for k, v in score.items()})
                atomic_json(out / f'validation/epoch_{epoch:03d}.json', dict(summary=metric, slices=state['val_records']))
                history = [json.loads(f.read_text())['summary'] for f in sorted((out / 'validation').glob('epoch_*.json'))]
                with (out / 'metrics.tmp').open('w') as f:
                    fields = list(dict.fromkeys((k for row in history for k in row)))
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(history)
                (out / 'metrics.tmp').replace(out / 'metrics.csv')
                print(json.dumps(dict(event='epoch_complete', **metric)), flush=True)
                state.update(epoch=epoch + 1, phase='train', cursor=0, val_case=0, val_cursor=0, val_records=[], train_records=[], train_seconds=0.0, val_seconds=0.0)
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
