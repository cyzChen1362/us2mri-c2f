import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from c2f.common.runtime import CFG, OUT, initialize, rows, write_json, stopping
from c2f.slice_matching.dataset import Dataset

def main():
    initialize()
    target = OUT / 'figures'
    target.mkdir(exist_ok=True)
    history = rows(OUT / 'metrics.csv')
    x = [int(r['epoch']) for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ['loss', 'window_r1']):
        for part in ['train', 'val']:
            ax.plot(x, [float(r[part + '_' + key]) for r in history], label=part)
        ax.set(xlabel='Epoch', ylabel=key)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(target / 'curves.png', dpi=150)
    plt.close(fig)
    data = Dataset()
    for name in data.subset('val') + data.subset('test'):
        if stopping():
            return stopping()
        dest = target / (name + '.png')
        if dest.exists():
            continue
        a = np.load(OUT / 'evaluation' / name / 'scores.npz')
        scores = a['scores']
        indices = a['slice_indices']
        candidates = a['candidate_indices'] if 'candidate_indices' in a else indices
        rank = np.argsort(-scores, axis=1, kind='stable')
        err = np.abs(candidates[rank[:, 0]] - indices)
        ordered = np.argsort(err, kind='stable')
        chosen = [int(ordered[round((len(ordered) - 1) * q)]) for q in [0.1, 0.5, 0.9]]
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(1 / (1 + np.exp(-np.clip(scores, -80, 80))), aspect='auto', origin='upper', cmap='viridis', vmin=0, vmax=1)
        ticks = np.unique(np.linspace(0, len(candidates) - 1, min(20, len(candidates))).astype(int))
        ax.set_xticks(ticks, labels=candidates[ticks], rotation=60)
        ticks = np.unique(np.linspace(0, len(indices) - 1, 11).astype(int))
        ax.set_yticks(ticks, labels=indices[ticks])
        ax.set(xlabel='MRI original slice index', ylabel='US original slice index', title=CFG['name'] + ' / ' + name)
        fig.colorbar(im, ax=ax, label='Matching score (sigmoid)')
        fig.tight_layout()
        fig.savefig(target / (name + '_heatmap.png'), dpi=140)
        plt.close(fig)
        case = data.load(name)
        fig, axes = plt.subplots(3, 4, figsize=(13, 10))
        for row, i in enumerate(chosen):
            js = [int(np.argmin(abs(candidates - indices[i]))), int(rank[i, 0]), int(rank[i, 1])]
            roi = case.crop(i, torch.tensor(js, device='cuda')).cpu().numpy()[:, 0]
            pictures = [case.us[i, 0].cpu().numpy(), *roi]
            titles = [f'US query {indices[i]}', f'Nearest MRI ROI {candidates[js[0]]}', f'Top1 MRI ROI {candidates[js[1]]}', f'Top2 MRI ROI {candidates[js[2]]}']
            for ax, img, title in zip(axes[row], pictures, titles):
                ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
                ax.set_title(title)
                ax.axis('off')
        fig.suptitle(CFG['name'] + ' / ' + name + ' (error percentiles 10,50,90)')
        fig.tight_layout()
        fig.savefig(dest, dpi=140)
        plt.close(fig)
        del case
    if CFG.get('tumor_input'):
        for split in ['val', 'test']:
            mean = np.mean([1 / (1 + np.exp(-np.clip(np.load(OUT / 'evaluation' / c / 'scores.npz')['scores'], -80, 80))) for c in data.subset(split)], axis=0)
            np.save(target / (split + '_mean_heatmap.npy'), mean)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(mean, vmin=0, vmax=1, cmap='viridis', origin='upper')
            ax.set(xlabel='MRI slice', ylabel='US slice', title=CFG['name'] + ' / ' + split + ' case mean')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(target / (split + '_mean_heatmap.png'), dpi=150)
            plt.close(fig)
    write_json(OUT / 'complete.json', dict(status='complete', max_epochs=CFG['max_epochs'], best_selection=CFG['selection_metric']))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
