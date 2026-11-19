# Preliminary study: direct pose regression (PlanePose v1–v6)

Run on RESECT (a smaller public iUS–MRI brain-tumour dataset) before moving to ReMIND; test split. Values transcribed from the experiment log.

| Variant | Point error (mm) | Rotation error (°) |
|---|---|---|
| v1 Direct pose regression | 48.0 | 144.1 |
| v2 Correspondence + RANSAC | 44.7 | 110.4 |
| v3 In-plane point-distance loss | 45.3 | 135.7 |
| v4 v1 + local position prior | 29.3 | 151.5 |
| v5 v2 + local position prior | 27.5 | 112.0 |
| v6 v3 + local position prior | 26.1 | 155.6 |
