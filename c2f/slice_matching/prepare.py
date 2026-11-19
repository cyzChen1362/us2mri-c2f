import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from c2f.common.runtime import CFG, write_json, stopping, sha
from c2f.window_localization.model.network import SpatialLocalizationModel
from c2f.window_localization.model.interaction import position_encoding
from .dataset import Dataset

@torch.no_grad()
def interact(model, encoded, masks, ii, jj):
    pyramid = {'US': {k: v[ii] for k, v in encoded['US'].items()}, 'MRI': {k: v[jj] for k, v in encoded['MRI'].items()}}
    coarse = {m: v['3'] for m, v in pyramid.items()}
    b, c, h, w = coarse['US'].shape
    ms = {'US': masks['US'][ii], 'MRI': masks['MRI'][jj]}
    valid = {m: F.adaptive_max_pool2d(x.float(), (h, w)).flatten(1) > 0 for m, x in ms.items()}
    tokens = {m: x.flatten(2).transpose(1, 2) for m, x in coarse.items()}
    pos = position_encoding(h, w, c, 'cuda', coarse['US'].dtype)
    for layer in model.interaction:
        tokens = layer(tokens, valid, pos)
    fine = {m: model.fine[m](pyramid[m], v.transpose(1, 2).reshape(b, c, h, w)) for m, v in tokens.items()}
    valid = {m: F.adaptive_max_pool2d(ms[m].float(), v.shape[-2:]).flatten(1) > 0 for m, v in fine.items()}
    return model.head(fine['US'], fine['MRI'], valid['US'], valid['MRI'])

@torch.no_grad()
def prepare(cases=None, query_limit=None, check_equivalence=False):
    data = Dataset()
    cache = data.cache
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(CFG['task3_checkpoint'])
    lc = json.loads(Path(CFG['task3_config']).read_text())
    identity = dict(checkpoint_sha256=sha(checkpoint), radimagenet_sha256=sha(CFG['weights']), split_sha256=sha(CFG['split_file']), fixed_index_sha256=sha(data.fixed / 'index.json'), sources={c: dict(source=sha(data.fixed / c / 'source.json'), inverse=sha(data.fixed / c / 'mr_to_original.npy'), planes=sha(Path(data.info[c]['meta']) / 'planes.json')) for c in data.cases}, geometry='MRIFullVision_Triangle_inverse_pixel_edges_query_specific_v1')
    mp = cache / 'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == identity, 'ROI source identity changed'
    else:
        write_json(mp, identity)
    pending = [c for c in cases or data.cases if not (cache / c / 'complete.json').exists()]
    if not pending:
        return 0
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SpatialLocalizationModel(lc).cuda().eval()
    model.load_state_dict(payload['model'])
    model.requires_grad_(False)
    del payload
    for case in pending:
        if stopping():
            return stopping()
        rr = data.rows[case]
        n = len(rr)
        folder = cache / case
        folder.mkdir(parents=True, exist_ok=True)
        fixed = data.fixed / case
        images = np.load(fixed / 'images.npy', mmap_mode='r')
        maskarray = np.load(fixed / 'masks.npy', mmap_mode='r')
        tensors = {m: torch.from_numpy(images[:, i:i + 1].copy()).cuda() for i, m in enumerate(['US', 'MRI'])}
        masks = {m: torch.from_numpy(maskarray[:, i:i + 1].copy()).cuda() for i, m in enumerate(['US', 'MRI'])}
        inverses = np.load(fixed / 'mr_to_original.npy')
        shapes = []
        for row in rr:
            a = np.load(data.root / row['us_path'])
            valid = np.isfinite(a) & (a != 0)
            ys, xs = np.where(valid)
            h, w = (int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
            scale = 256 / max(h, w)
            shapes.append([max(1, round(h * scale)), max(1, round(w * scale))])
        np.save(folder / 'us_shapes.npy', np.array(shapes, np.int32))
        encoded = {}
        for m, x in tensors.items():
            chunks = []
            for start in range(0, n, 20):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    chunks.append(model.encode(x[start:start + 16], m)['pyramid'])
            encoded[m] = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
        if check_equivalence:
            ii = torch.tensor([0, 1, 77], device='cuda')
            jj = torch.tensor([0, n - 1, 1], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                direct = model(tensors['US'][ii], tensors['MRI'][jj], masks['US'][ii], masks['MRI'][jj])['box']
                reuse = interact(model, encoded, masks, ii, jj)
            assert torch.allclose(direct, reuse, atol=0.0024, rtol=0.006), (direct, reuse)
            assert not any((p.requires_grad for p in model.parameters()))
        bp = folder / 'boxes.npy'
        progress = folder / 'progress.json'
        boxes = np.load(bp) if bp.exists() else np.full((n, n, 4), np.nan, np.float32)
        begin = json.loads(progress.read_text())['next_query'] if progress.exists() else 0

        def persist(i):
            tmp = bp.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.save(f, boxes)
            tmp.replace(bp)
            write_json(progress, dict(next_query=i, total_queries=n))
        end = min(n, query_limit) if query_limit is not None else n
        for i in range(begin, end):
            if stopping():
                persist(i)
                return stopping()
            for start in range(0, n, CFG['localizer_chunk']):
                jj = torch.arange(start, min(start + CFG['localizer_chunk'], n), device='cuda')
                ii = torch.full_like(jj, i)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    b = interact(model, encoded, masks, ii, jj).float().cpu().numpy() * 256
                low = b[:, :2] - b[:, 2:] / 2
                high = b[:, :2] + b[:, 2:] / 2
                for k, j in enumerate(jj.cpu().tolist()):
                    inv = inverses[j]
                    corners = np.array([[low[k, 0], low[k, 1], 1], [high[k, 0], low[k, 1], 1], [high[k, 0], high[k, 1], 1], [low[k, 0], high[k, 1], 1]]) @ inv.T
                    boxes[i, j] = np.r_[corners[:, :2].min(0), corners[:, :2].max(0)]
            if (i + 1) % 15 == 0 or i + 1 == end:
                persist(i + 1)
                print(f'ROI {case} {i + 1}/{n}', flush=True)
        if end == n:
            write_json(folder / 'complete.json', dict(case=case, queries=n, candidates=n))
        del encoded, tensors, masks, images, maskarray
    return 0
if __name__ == '__main__':
    from c2f.common.runtime import initialize
    initialize()
    raise SystemExit(prepare())
