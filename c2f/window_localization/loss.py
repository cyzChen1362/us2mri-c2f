"""Training objective of the window localizer.

Combines a centre (heat-map) term with a box-overlap term so that both the
location and the extent of the US window are supervised. Targets come from
the known pose of every trajectory slice (US field of view projected onto the
matching MRI plane).

The exact formulation and weights are removed in this public release.
"""
REDACTED = 'Removed in the public release (desensitised); see the module docstring.'


def window_loss(pred: dict, target_box, target_mask=None):
    """pred: output of WindowLocalizer.forward; target_box: (B, 4) xyxy in [0, 1]."""
    raise NotImplementedError(REDACTED)
