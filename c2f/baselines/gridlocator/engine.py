import torch
from .objective import loss_fn, records

def grouped_backward(model, mri, queries, cfg, microbatch=None, amp=True, dropout_mask=None):
    """One differentiable MRI encoding, chunked queries, one optimizer update.

    Accumulate query gradients on a feature leaf; propagate their sum through
    the MRI encoder exactly once. Every query has weight 1 / group size.
    """
    chunk = microbatch or cfg['training']['fusion_microbatch']
    device = mri.device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        feature = model.mri(mri)
    leaf = feature.detach().requires_grad_(True)
    rr = []
    probability = getattr(model, 'dropout_probability', 0.0)
    mask = dropout_mask
    if mask is None and probability and model.training:
        mask = torch.empty(len(queries), 64, 1, 1, 1, device=device).bernoulli_(1 - probability) / (1 - probability)
    for start in range(0, len(queries), chunk):
        qq = queries if start == 0 and len(queries) <= chunk else queries[start:start + chunk]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            vector = model.encode_queries(qq, device)
            out = model.predict(leaf, vector, dropout_mask=mask[start:start + len(qq)]) if mask is not None else model.predict(leaf, vector)
            loss, components = loss_fn(out, qq, cfg)
        rr.extend(records(out, qq, cfg, components))
        (loss * len(qq) / len(queries)).backward()
        del out, loss, components, vector
    feature.backward(leaf.grad)
    return rr

@torch.no_grad()
def predict_queries(model, feature, queries, cfg, amp=True):
    with torch.autocast(feature.device.type, dtype=torch.bfloat16, enabled=amp):
        out = model.predict(feature, model.encode_queries(queries, feature.device))
        _, parts = loss_fn(out, queries, cfg)
    return records(out, queries, cfg, parts)
