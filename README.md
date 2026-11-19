# Coarse-to-Fine 2D Ultrasound to 3D MRI Registration with Patient-Specific Adaptation

CSIT 5410 Group Project · Group 4 · The Hong Kong University of Science and Technology · November 2026

> **Desensitised release.** To protect ongoing, unpublished research, this
> repository is a desensitised version of the code used in the project. Core
> implementations are removed and replaced by documented skeletons, and
> pre-processing, augmentation and training hyper-parameters were randomly
> altered. The code shows the structure of the system and the amount of work;
> it is **not** meant to reproduce the reported numbers. All numbers and figures
> in `results/` come from the original experiment outputs
> (see [Desensitisation](#desensitisation)).

## Task

Given one intra-operative 2D ultrasound (US) frame and the patient's
pre-operative 3D MRI, find the 6-DoF rigid pose of the US plane inside the MRI
and show the matching MRI slice next to the live ultrasound.

## Pipeline

```
3D US + MRI ──> trajectory slicing ──> 200 regular US frames + pose labels per patient
                                             │
 COARSE (2D–2D)   window localization ──> slice matching ──> patient-specific adaptation
                  "where to look"          "which slice"      (pseudo-US from this patient's MRI)
                                             │ top-1 slice = initial pose
 FINE (2D–3D)     recurrent pose refinement (5 steps, re-slice MRI at every step)
```

Direct 2D–3D registration was tried first (PixelVoxel, GridLocator on ReMIND;
PlanePose in a preliminary study on RESECT) and did not generalise to unseen
patients, which motivated the coarse-to-fine decomposition.

## Repository layout

```
c2f/
  common/               geometry, re-slicing, pre-processing, metrics, data loading
  trajectory/           scan-trajectory estimation and slicing          [skeleton]
  window_localization/  coarse stage 1: US window inside each MRI slice   [skeleton]
  slice_matching/       coarse stage 2: US ↔ 200 MRI candidates scoring   [model/loss skeleton]
  adaptation/           patient-specific adaptation with a US generator    [skeleton]
    generator/          MRI → pseudo-US generator training                 [skeleton]
  refinement/           fine stage: recurrent 6-DoF pose refinement        [model/loss skeleton]
  baselines/
    pixelvoxel/         direct baseline: dense pixel-to-voxel matching      [full code, values altered]
    gridlocator/        direct baseline: detection-style 3D grid            [full code, values altered]
    planepose/          preliminary study on RESECT                          [skeleton]
configs/                one config per stage and MRI branch (ceT1, FLAIR); values altered
results/                tables (Markdown + CSV) and figures
```

## Data

[ReMIND](https://www.cancerimagingarchive.net/collection/remind/) (Juvekar et al.,
*Scientific Data* 2024): pre-operative MRI and intra-operative 3D US of
brain-tumour surgery. Two MRI branches are used: US–ceT1 (66 / 12 / 19
patients) and US–FLAIR (45 / 11 / 13 patients), 200 trajectory slices per patient.
The patient-level split is not published; patient identifiers in `results/` are
anonymised. Images shown in `results/figures` are derived from ReMIND.

## Main results

| Stage | Metric | US–ceT1 | US–FLAIR |
|---|---|---|---|
| Direct baseline (GridLocator v2, test) | centre error | 15.65 mm | 15.58 mm |
| Window localization (test) | 2D IoU | 0.799 | 0.750 |
| Slice matching, generic → adapted (test) | Top-1 within ±5 slices | 25.6% → 36.0% | 15.6% → 50.2% |
| Slice matching, generic → adapted (test) | centre error of selected plane | 7.85 → 5.50 mm | 10.63 → 4.46 mm |
| Refinement, coarse position → fine position (test) | plane-point error, step 0 → 5 | 6.83 → 1.88 mm | 5.54 → 1.37 mm |

Full tables: [`results/README.md`](results/README.md).

## Desensitisation

| Part | What is published |
|---|---|
| window localization, slice matching, adaptation, refinement | module structure, interfaces and documented idea; implementation removed |
| PixelVoxel / GridLocator baselines | full code; comments removed and numeric constants randomly altered |
| geometry, re-slicing, metrics, evaluation, data loading | kept, with comments removed and numeric constants randomly altered |
| configs | core stages: only generic fields; baselines: full; numeric values randomly altered; paths replaced by `${DATA_ROOT}` |
| patient split | not published; patient identifiers in tables and figures anonymised |
| trajectory slicing, generator training, PlanePose | module structure, interfaces and documented idea; implementation removed |

## References

- A. Juvekar et al. ReMIND: The Brain Resection Multimodal Imaging Database. *Scientific Data*, 2024.
- Y. Xiao et al. RESECT: A clinical database of pre-operative MRI and intra-operative ultrasound in low-grade glioma surgeries. *Medical Physics*, 2017.
- xvr: patient-specific X-ray to volume registration. *Nature*, 2026.
- DreamReg: recurrent 2D/3D registration. arXiv:2606.18825, 2026.
- T. Park et al. Contrastive Learning for Unpaired Image-to-Image Translation (CUT). *ECCV*, 2020.
- X. Mei et al. RadImageNet: an open radiologic deep learning research dataset. *Radiology: AI*, 2022.
- K. He et al. Deep Residual Learning for Image Recognition. *CVPR*, 2016.

## License

Copyright © 2026 CSIT 5410 Group 4. All rights reserved. Shared for course
assessment; no permission is granted to reuse the code.
