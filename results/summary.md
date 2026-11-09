# Results (held-out test attacks: #33, #34, #35, #36)

| Experiment | Real data | Attacks caught | Mean delay (s) | False alarms/day | Precision | Recall | F1 | Accuracy† | Point-adjusted F1* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics check only (no ML) | 100% | 2.0 / 4 | 58 | 0.7 | 0.413 | 0.082 | 0.137 | 0.986 | 0.720 |
| Real SWaT only | 100% | 3.0 / 4 | 50 | 0.0 | 0.861 | 0.331 | 0.478 | 0.991 | 0.768 |
| Simulator (with attacks) + fine-tune on real | 100% | 4.0 / 4 | 62 | 5.8 | 0.611 | 0.842 | 0.708 | 0.991 | 0.789 |
| Simulator (no attacks) + fine-tune on real | 100% | 3.0 / 4 | 67 | 10.8 | 0.651 | 0.530 | 0.584 | 0.990 | 0.676 |
| Simulator only (no real training) | 100% | 2.0 / 4 | 38 | 0.0 | 0.888 | 0.325 | 0.476 | 0.991 | 0.527 |

*Point-adjusted F1 credits a whole attack if any single step is flagged; it inflates results and is shown only for comparison with published SWaT papers.
†Accuracy is dominated by normal time (most test rows are normal), so a detector that never alarms still scores high. Precision, recall and F1 are per 5-second step on the held-out test days.

Simulator: stage-1 simulator (simulator.py). Generated 2026-11-08T19:08:39.