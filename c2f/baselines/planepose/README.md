# PlanePose v1–v6 (preliminary study, RESECT)

Direct 6-DoF pose regression from a US frame and the MRI volume, run on the
RESECT dataset before moving to ReMIND (results: `results/tables/03_preliminary_planepose_resect.md`).

| Variant | Idea |
|---|---|
| v1 | direct pose regression |
| v2 | correspondence + RANSAC |
| v3 | in-plane point-distance loss |
| v4–v6 | v1–v3 + local position prior |

**Status:** skeleton (desensitised): `data.py`, `model.py`, `prior.py` (v4–v6),
`ransac.py` (v2 / v5), `losses.py`, `train.py`, `evaluate.py`.
