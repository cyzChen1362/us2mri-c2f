"""Same-case multi-US mean loss, one MRI encoding and one optimizer update.
The FP32 descriptor leaf collects all query gradients; MRI backward runs once.
This is equivalent to a joint mean-loss update, NOT to B sequential Adam steps.
"""
import time
import torch
from .model import infonce

def grouped_backward(model, mri, queries, amp=True):

    def sync():
        if mri.is_cuda:
            torch.cuda.synchronize()
    sync()
    start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        dm = model.mri(mri)
    leaf = dm.detach().float().requires_grad_(True)
    sync()
    encoded = time.perf_counter()
    total = 0.0
    for us, points in queries:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            du = model.us(us.to(mri.device))
        loss = infonce(du, leaf, *points) / len(queries)
        loss.backward()
        total += float(loss.detach())
    sync()
    queries_done = time.perf_counter()
    dm.backward(leaf.grad.to(dm.dtype))
    sync()
    end = time.perf_counter()
    return (total, dict(mri_forward_seconds=encoded - start, us_and_loss_backward_seconds=queries_done - encoded, mri_backward_seconds=end - queries_done))
