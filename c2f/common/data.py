"""Shared fixed-cache and dynamic-plane data loading selected by configuration."""
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .paths import DATA, cache_path
from pathlib import Path

class FixedLocalizationData:

    def __init__(self, modality, cases=None, root=None, case_roots=None):
        self.root = root if root is not None else DATA / modality
        self.modality = modality
        self.case_roots = {k: Path(v) for k, v in (case_roots or {}).items()}
        if cases:
            self.rows = []
            for case in cases:
                meta = json.loads((self.case_roots.get(case, self.root / case) / 'complete.json').read_text())
                self.rows.extend((dict(case=case, partition=meta['partition'], local_index=i, k=r['k']) for i, r in enumerate(meta['slice_records'])))
        else:
            self.rows = json.loads((self.root / 'index.json').read_text())
        self.maps = OrderedDict()
        self.meta = {}

    def subset(self, split):
        return np.array([i for i, r in enumerate(self.rows) if r['partition'] == split], dtype=np.int64)

    def batch(self, indices, device):
        samples = []
        for index in indices:
            row = self.rows[int(index)]
            case = row['case']
            if case not in self.maps:
                path = self.case_roots.get(case, self.root / case)
                keys = ['images', 'masks', 'boxes', 'mr_to_original']
                if (path / 'pixel_basis.npy').exists():
                    keys += ['pixel_basis']
                self.maps[case] = {k: np.load(path / f'{k}.npy', mmap_mode='r') for k in keys}
                self.meta[case] = json.loads((path / 'complete.json').read_text())
                if len(self.maps) > 15:
                    self.maps.popitem(last=False)
            self.maps.move_to_end(case)
            arrays = self.maps[case]
            i = row['local_index']
            meta = self.meta[case]
            basis = arrays['pixel_basis'][i] if 'pixel_basis' in arrays else np.array(meta['direction']).reshape(3, 4)[:, :2] * np.array(meta['spacing'][:2])[None, :]
            samples.append({k: arrays[k][i] for k in arrays} | {'pixel_mm': np.linalg.norm(basis, axis=0), 'pixel_basis': basis.astype(np.float32)})
        x = {k: torch.as_tensor(np.stack([s[k] for s in samples]), device=device) for k in samples[0]}
        return dict(us=x['images'][:, 0:1], mr=x['images'][:, 1:2], us_mask=x['masks'][:, 0:1], mr_mask=x['masks'][:, 1:2], box=x['boxes'], mr_to_original=x['mr_to_original'], pixel_mm=x['pixel_mm'].float(), pixel_basis=x['pixel_basis'].float())

def shards(indices, rank, world):
    """Pad only the final global batch; repeated entries get zero loss weight."""
    n = len(indices)
    width = (n + world - 1) // world
    padded = np.resize(np.asarray(indices, dtype=np.int64), width * world)
    positions = np.arange(rank, width * world, world)
    return (padded[positions], positions < n)

def loading_mode(cfg=None):
    cfg = cfg or {}
    mode = cfg.get('data', {}).get('loading_mode', 'dynamic' if cfg.get('version', 1) == 2 else 'fixed')
    if mode not in ('fixed', 'dynamic'):
        raise ValueError(f'Unknown data loading mode: {mode}')
    return mode

class LocalizationData(FixedLocalizationData):

    def __init__(self, modality, cases=None, cfg=None):
        super().__init__(modality, cases, root=cache_path(cfg) if cfg else None, case_roots=(cfg or {}).get('data', {}).get('case_cache_overrides'))
        self.cfg = cfg
        self.mode = loading_mode(cfg)
        self.sources = {}
        if self.mode == 'dynamic':
            import SimpleITK as sitk
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    def preload(self):
        from .reslicing import load_source
        if self.cfg['data'].get('dataset_variant') == 'MRIFullVision':
            from .fullvision import load_source as full_source
            loader = lambda case: full_source(self.cfg, case)
        else:
            loader = lambda case: load_source(self.modality, case)
        cases = sorted({r['case'] for r in self.rows if r['partition'] == 'train'})
        for case in cases:
            if case not in self.sources:
                self.sources[case] = loader(case)
        print(f'DYNAMIC_3D_CACHE_READY cases={len(cases)}', flush=True)

    def sample(self, index, epoch):
        from .reslicing import generate
        if self.cfg['data'].get('dataset_variant') == 'MRIFullVision':
            from .fullvision import generate
        row = self.rows[int(index)]
        return generate(self.sources[row['case']], row, epoch, self.cfg, int(index))

    @staticmethod
    def collate(samples, device):
        x = {k: torch.as_tensor(np.stack([s[k] for s in samples]), device=device) for k in ['images', 'masks', 'box', 'mr_to_original', 'pixel_mm', 'pixel_basis']}
        return dict(us=x['images'][:, 0:1], mr=x['images'][:, 1:2], us_mask=x['masks'][:, 0:1], mr_mask=x['masks'][:, 1:2], box=x['box'], mr_to_original=x['mr_to_original'], pixel_mm=x['pixel_mm'], pixel_basis=x['pixel_basis'])

    def batches(self, indices, batch_size, device, epoch, start, dynamic, rank=0, world=1):
        steps = (len(indices) + batch_size - 1) // batch_size
        if not dynamic or self.mode == 'fixed':
            for cursor in range(start, steps):
                ix = indices[cursor * batch_size:(cursor + 1) * batch_size]
                own, valid = shards(ix, rank, world)
                yield (cursor, own, self.batch(own, device), [], valid, len(ix))
            return
        if world != 1:
            raise ValueError('Dynamic reslicing currently uses one GPU per modality')
        if not self.sources:
            self.preload()
        ahead = self.cfg['sampling']['prefetch_batches']
        pending = {}
        with ThreadPoolExecutor(self.cfg['sampling']['workers']) as pool:

            def submit(cursor):
                ix = indices[cursor * batch_size:(cursor + 1) * batch_size]
                pending[cursor] = [pool.submit(self.sample, int(i), epoch) for i in ix]
            for c in range(start, min(steps, start + ahead)):
                submit(c)
            for cursor in range(start, steps):
                samples = [f.result() for f in pending.pop(cursor)]
                if cursor + ahead < steps:
                    submit(cursor + ahead)
                ix = indices[cursor * batch_size:(cursor + 1) * batch_size]
                yield (cursor, ix, self.collate(samples, device), [s['audit'] for s in samples], np.ones(len(ix), bool), len(ix))
