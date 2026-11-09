# Results (held-out test attacks: #33, #34, #35, #36)

| Experiment | Real data | Attacks caught | Mean delay (s) | False alarms/day | Precision | Recall | F1 | Accuracy† | Point-adjusted F1* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics check only (no ML) (Mass balance only) | 100% | 2.0 / 4 | 58 | 0.7 | 0.413 | 0.082 | 0.137 | 0.986 | 0.720 |
| Physics check only (no ML) (+ rules) | 100% | 2.0 / 4 | 58 | 0.7 | 0.413 | 0.082 | 0.137 | 0.986 | 0.720 |
| Real SWaT only (Model only) | 100% | 1.0 / 4 | 60 | 0.0 | 0.757 | 0.246 | 0.371 | 0.989 | 0.417 |
| Real SWaT only (+ rules) | 100% | 3.0 / 4 | 43 | 0.0 | 0.820 | 0.574 | 0.675 | 0.993 | 0.736 |
| Simulator (with attacks) + fine-tune on real (Model only) | 100% | 0.0 / 4 | – | 0.0 | 0.000 | 0.000 | 0.000 | 0.987 | 0.000 |
| Simulator (with attacks) + fine-tune on real (+ rules) | 100% | 4.0 / 4 | 64 | 1.4 | 0.714 | 0.789 | 0.750 | 0.993 | 0.864 |
| Simulator (no attacks) + fine-tune on real (Model only) | 100% | 0.0 / 4 | – | 0.0 | 0.000 | 0.000 | 0.000 | 0.987 | 0.000 |
| Simulator (no attacks) + fine-tune on real (+ rules) | 100% | 2.0 / 4 | 38 | 2.9 | 0.798 | 0.325 | 0.462 | 0.990 | 0.512 |
| Simulator only (no real training) (Model only) | 100% | 0.0 / 4 | – | 0.0 | 0.000 | 0.000 | 0.000 | 0.987 | 0.000 |
| Simulator only (no real training) (+ rules) | 100% | 2.0 / 4 | 38 | 0.0 | 0.888 | 0.325 | 0.476 | 0.991 | 0.527 |

*Point-adjusted F1 credits a whole attack if any single step is flagged; it inflates results and is shown only for comparison with published SWaT papers.
†Accuracy is dominated by normal time (most test rows are normal), so a detector that never alarms still scores high. Precision, recall and F1 are per 5-second step on the held-out test days.

Simulator: stage-1 simulator (simulator.py). Generated 2026-11-08T21:59:39.