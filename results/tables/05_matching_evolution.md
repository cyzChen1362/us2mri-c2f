# Coarse stage 2: evolution of the slice matcher

200 candidate MRI planes per patient; Top-1 ±k = best candidate within k slices of the truth.

| Branch | Model | Split | Patients | Top-1 ±5 | Top-1 ±10 | Index MAE (slices) | Centre MAE (mm) |
|---|---|---|---|---|---|---|---|
| ceT1 | Classification v1 | val | 12 | 17.9% | 32.4% | 44.3 | 11.59 |
| ceT1 | Classification v1 | test | 19 | 18.1% | 31.2% | 39.1 | 9.78 |
| ceT1 | Classification v2 | test | 19 | 24.5% | 39.7% | 35.4 | 8.90 |
| ceT1 | Classification v2 | val | 12 | 22.8% | 37.5% | 36.6 | 9.89 |
| ceT1 | Classification v3 | test | 19 | 16.7% | 27.4% | 45.3 | 12.07 |
| ceT1 | Classification v3 | val | 12 | 13.9% | 24.1% | 51.3 | 13.64 |
| ceT1 | Classification v4 | val | 12 | 23.0% | 36.2% | 37.7 | 10.03 |
| ceT1 | Classification v4 | test | 19 | 25.6% | 41.1% | 32.1 | 7.96 |
| ceT1 | sparse v5 | test | 19 | 21.8% | 36.4% | 34.1 | 8.70 |
| ceT1 | sparse v5 | val | 12 | 21.6% | 35.9% | 39.1 | 10.59 |
| ceT1 | Adaptation | val | 12 | 31.5% | 49.2% | 25.3 | 6.96 |
| ceT1 | Adaptation | test | 19 | 36.0% | 54.2% | 22.4 | 5.61 |
| FLAIR | Contrastive v1 | val | 11 | 8.5% | 13.0% | 61.6 | 15.38 |
| FLAIR | Contrastive v1 | test | 13 | 5.4% | 10.7% | 65.1 | 15.66 |
| FLAIR | Classification v1 | val | 11 | 11.6% | 20.4% | 50.1 | 12.01 |
| FLAIR | Classification v1 | test | 13 | 14.3% | 26.1% | 51.5 | 12.58 |
| FLAIR | Classification v2 | test | 13 | 12.2% | 20.2% | 53.0 | 12.77 |
| FLAIR | Classification v2 | val | 11 | 12.8% | 22.5% | 48.7 | 12.05 |
| FLAIR | Classification v3 | test | 13 | 14.8% | 25.0% | 53.0 | 12.96 |
| FLAIR | Classification v3 | val | 11 | 11.7% | 20.9% | 53.3 | 12.85 |
| FLAIR | Classification v4 | val | 11 | 15.8% | 25.8% | 49.4 | 12.01 |
| FLAIR | Classification v4 | test | 13 | 15.6% | 28.8% | 43.5 | 10.74 |
| FLAIR | sparse v5 | test | 13 | 12.9% | 22.9% | 52.4 | 12.93 |
| FLAIR | sparse v5 | val | 11 | 12.7% | 21.2% | 52.4 | 13.10 |
| FLAIR | Adaptation | val | 11 | 49.0% | 65.7% | 18.2 | 4.45 |
| FLAIR | Adaptation | test | 13 | 50.2% | 66.5% | 18.3 | 4.48 |
