"""Restartable best-checkpoint evaluation and prediction figures."""
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np
import torch
from . import runtime as rt
from .data import LocalizationData
from .model.spatial_localization import SpatialLocalizationModel
from .train import forward_loss
from .metrics import accumulate, summarize

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def plot_predictions(run, model, data, cfg, device, indices=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder = run / 'evaluation/figures'
    folder.mkdir(parents=True, exist_ok=True)
    if indices is None:
        available = data.subset('test')
        indices = available[np.linspace(0, len(available) - 1, min(12, len(available))).astype(int)]
    with torch.no_grad():
        for index in indices:
            batch = data.batch([index], device)
            _, _, out = forward_loss(model, batch, cfg)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(batch['us'][0, 0].cpu(), cmap='gray', vmin=-1, vmax=1)
            axes[1].imshow(batch['mr'][0, 0].cpu(), cmap='gray', vmin=-1, vmax=1)
            for box, color, label in [(batch['box'][0], 'lime', 'Ground truth'), (out['box'][0], 'red', 'Prediction')]:
                cx, cy, w, h = box.detach().cpu().float().numpy() * 256
                axes[1].add_patch(Rectangle((cx - w / 2 - 0.5, cy - h / 2 - 0.5), w, h, fill=False, edgecolor=color, linewidth=1.5, label=label))
            row = data.rows[int(index)]
            axes[0].set_title(f"{row['case']} / k={row['k']} / US")
            axes[1].set_title(cfg['modality'] + ' full-plane ROI256')
            axes[1].legend(loc='lower right', fontsize=8)
            for ax in axes:
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(folder / f"{row['case']}_k{row['k']}.png", dpi=130)
            plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    run = a.run_dir
    cfg = json.loads((run / 'config.yaml').read_text())
    device = rt.setup(cfg['seed'])
    data = LocalizationData(cfg['modality'], cfg=cfg)
    model = SpatialLocalizationModel(cfg).to(device)
    checkpoint = torch.load(run / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    folder = run / 'evaluation'
    folder.mkdir(exist_ok=True)
    summaries = {}
    cases = []
    for split in ['train', 'validation'] if a.check else ['train', 'validation', 'test']:
        indices = data.subset(split)
        if a.check:
            indices = indices[:8]
        partial = folder / f'{split}_progress.pt'
        progress = torch.load(partial, weights_only=False) if partial.exists() else dict(cursor=0, totals={}, predictions=[])
        batch_size = cfg['training']['global_batch_size']
        steps = (len(indices) + batch_size - 1) // batch_size
        last_save = time.monotonic()
        with torch.no_grad():
            for cursor in range(progress['cursor'], steps):
                own = indices[cursor * batch_size:(cursor + 1) * batch_size]
                batch = data.batch(own, device)
                _, metrics, out = forward_loss(model, batch, cfg)
                accumulate(progress['totals'], metrics, data, own, np.ones(len(own), bool))
                for j, index in enumerate(own):
                    r = data.rows[int(index)]
                    pred = out['box'][j].float().cpu().numpy()
                    truth = batch['box'][j].cpu().numpy()
                    q = np.array([[pred[0] - pred[2] / 2, pred[1] - pred[3] / 2, 1 / 256], [pred[0] + pred[2] / 2, pred[1] + pred[3] / 2, 1 / 256]]) * 256
                    physical = (batch['mr_to_original'][j].cpu().numpy() @ q.T).T[:, :2].reshape(-1)
                    record = dict(case=r['case'], partition=split, k=r['k'], **{f'pred_{k}': float(v) for k, v in zip(['cx', 'cy', 'w', 'h'], pred)}, **{f'true_{k}': float(v) for k, v in zip(['cx', 'cy', 'w', 'h'], truth)}, **{f'pred_original_{k}': float(v) for k, v in zip(['x0', 'y0', 'x1', 'y1'], physical)}, **{k: float(v[j]) for k, v in metrics.items()})
                    progress['predictions'].append(record)
                progress['cursor'] = cursor + 1
                reason = rt.stop_reason(run)
                if reason or time.monotonic() - last_save > 600:
                    rt.atomic_save(progress, partial)
                    last_save = time.monotonic()
                if reason:
                    if reason == 'deadline':
                        (run / f".resume_requested_{os.environ['SLURM_JOB_ID']}").write_text('evaluate')
                    return
        rt.atomic_save(progress, partial)
        summary, per_case = summarize([progress['totals']])
        summaries[split] = summary
        cases.extend((dict(partition=split, **r) for r in per_case))
        write_csv(folder / f'{split}_predictions.csv', progress['predictions'])
    write_csv(folder / 'case_metrics.csv', cases)
    (folder / 'summary.json').write_text(json.dumps(dict(best_epoch=checkpoint['state']['best_epoch'], scores=summaries), indent=2))
    plot_predictions(run, model, data, cfg, device, indices=data.subset('validation')[:2] if a.check else None)
    if (run / 'metrics.csv').exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with (run / 'metrics.csv').open() as f:
            history = list(csv.DictReader(f))
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for split in ['train', 'val']:
            for ax, key in zip(axs, ['loss', 'iou']):
                ax.plot([int(r['epoch']) for r in history], [float(r[f'{split}_{key}']) for r in history], label=split)
                ax.set_title(key)
                ax.legend()
        fig.tight_layout()
        fig.savefig(folder / 'learning_curves.png', dpi=140)
        plt.close(fig)
    (folder / 'complete.json').write_text(json.dumps(dict(complete=True, check_only=a.check), indent=2))
    print('EVALUATION_COMPLETE', json.dumps(summaries), flush=True)
if __name__ == '__main__':
    main()
