# Results

Patient identifiers are anonymised and the patient split is not published.
All numbers are read from the original experiment outputs
(CSV versions next to each Markdown table).

## Tables

| # | Table | Content |
|---|---|---|
| 1 | [01_dataset](tables/01_dataset.md) | patients and slices per split |
| 2a | [02a_baseline_pixelvoxel](tables/02a_baseline_pixelvoxel.md) | direct baseline: pixel-to-voxel matching |
| 2b | [02b_baseline_gridlocator_trainval](tables/02b_baseline_gridlocator_trainval.md) | direct baseline: 3D grid, train vs validation gap |
| 2c | [02c_baseline_gridlocator_test](tables/02c_baseline_gridlocator_test.md) | direct baseline: 3D grid on test patients |
| 3 | [03_preliminary_planepose_resect](tables/03_preliminary_planepose_resect.md) | preliminary study on RESECT: direct pose regression |
| 4 | [04_window_localization](tables/04_window_localization.md) | coarse stage 1 |
| 5 | [05_matching_evolution](tables/05_matching_evolution.md) | coarse stage 2: all matcher versions |
| 6 | [06_patient_adaptation](tables/06_patient_adaptation.md) | generic vs patient-adapted matcher |
| 7a | [07a_refinement](tables/07a_refinement.md) | fine stage per step |
| 7b | [07b_refinement_per_patient](tables/07b_refinement_per_patient.md) | every patient |

## Figures

| Figure | Content |
|---|---|
| `crop_pairs.png` | trajectory US slices and matching MRI planes |
| `baseline_gridlocator_curves.png`, `direct_gap.png` | direct baselines: training fits, validation does not |
| `crop_loc_cet1.png` | window localization examples (truth vs prediction) |
| `evolution.png` | matcher versions, test Top-1 ±5 |
| `heat_before_after.png`, `heat_case_FLAIR.png` | mean / single-patient score matrices before and after adaptation |
| `heatmaps_{ceT1,FLAIR}_{validation,test}.png` | score matrices of **every** validation and test patient, before / after adaptation |
| `adapt_bars.png` | adaptation gains on test patients |
| `gru_train_curve.png` | refinement error per step, all training patients |
| `chart_val.png` | refinement on the validation patients |
| `case_detail.png` | two hard patients (anonymised), per-step and per-slice errors |
