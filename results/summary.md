# Results (held-out test attacks: #33, #34, #35, #36)

| Experiment | Real data | Attacks caught | Mean delay (s) | False alarms/day | Precision | Recall | F1 | Accuracy† | Point-adjusted F1* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics check only (no ML) | 100% | 2.0 / 4 | 58 | 0.7 | 0.413 | 0.082 | 0.137 | 0.987 | 0.720 |
| Real SWaT only | 100% | 1.0 / 4 | 60 | 9.8 | 0.017 | 0.047 | 0.025 | 0.954 | 0.143 |
| Simulator (with attacks) + fine-tune on real | 100% | 3.0 / 4 | 82 | 8.4 | 0.133 | 0.539 | 0.213 | 0.949 | 0.266 |
| Simulator (no attacks) + fine-tune on real | 100% | 1.0 / 4 | 215 | 29.5 | 0.093 | 0.148 | 0.114 | 0.971 | 0.208 |
| Simulator only (no real training) | 100% | 0.0 / 4 | – | 0.7 | 0.000 | 0.000 | 0.000 | 0.987 | 0.000 |

*Point-adjusted F1 credits a whole attack if any single step is flagged; it inflates results and is shown only for comparison with published SWaT papers.
†Accuracy is dominated by normal time (most test rows are normal), so a detector that never alarms still scores high. Precision, recall and F1 are per 5-second step on the held-out test days.

Simulator: stage-1 simulator (simulator.py). Generated 2026-11-08T15:12:08.