# Results (held-out test attacks: #33, #34, #35, #36)

| Experiment | Real data | Attacks caught | Mean delay (s) | False alarms/day | Precision | Recall | F1 | Accuracy† | Point-adjusted F1* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics check only (no ML) | 100% | 0.0 / 4 | – | 0.0 | 0.000 | 0.000 | 0.000 | 0.987 | 0.000 |
| Real SWaT only | 100% | 3.0 / 4 | 38 | 0.0 | 0.633 | 0.353 | 0.453 | 0.989 | 0.705 |
| Simulator (with attacks) + fine-tune on real | 100% | 4.0 / 4 | 42 | 0.7 | 0.606 | 0.893 | 0.722 | 0.991 | 0.775 |
| Simulator (no attacks) + fine-tune on real | 100% | 2.0 / 4 | 20 | 0.0 | 0.643 | 0.347 | 0.451 | 0.989 | 0.476 |
| Simulator only (no real training) | 100% | 2.0 / 4 | 20 | 0.0 | 0.643 | 0.347 | 0.451 | 0.989 | 0.476 |

*Point-adjusted F1 credits a whole attack if any single step is flagged; it inflates results and is shown only for comparison with published SWaT papers.
†Accuracy is dominated by normal time (most test rows are normal), so a detector that never alarms still scores high. Precision, recall and F1 are per 5-second step on the held-out test days.

Simulator: stage-1 simulator (simulator.py). Generated 2026-11-08T18:11:34.