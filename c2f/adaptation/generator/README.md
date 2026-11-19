# MRI → pseudo-US generator (training)

Unpaired image-to-image translation (CUT family, 2.5D MRI input) trained on
**training patients only**. Used by `c2f/adaptation/generate.py`.

**Status:** skeleton (desensitised): `data.py` (unpaired 2.5D MRI / US), `networks.py`
(generator, PatchNCE heads, discriminator), `losses.py`, `train.py`.
