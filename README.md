# SWaT Stage 1: simulator-pretrained LSTM attack detector

Does pretraining on a simulator help an LSTM detect **real** cyberattacks on a water
treatment plant's raw water tank? We test it on SWaT (iTrust, SUTD), stage 1:
`FIT101`, `LIT101`, `MV101`, `P101`, `P102`.

## Project Summary 
PENDING

## About The Project 
PENDING 

## Quick start

```bash
pip install -r requirements.txt

# 1. Convert the SWaT Excel files once (about 4 minutes)
python convert_swat.py --swat-dir "path/to/Physical" --out data

# 2. Check that everything runs (about 2 minutes; numbers are meaningless)
python run_experiments.py --smoke

# 3. Real run
python run_experiments.py

# 4. Look at the simulator on its own (2 seconds): CSVs, attack log and a plot in sim_out/
python simulator.py --hours 24 --clean
python check_simulator.py            # side by side with the real plant
```

Outputs go to `results/`: `results.json` (for the dashboard) and `summary.md` (a table for the slides).

**Never commit `data/` or the SWaT files.** The iTrust licence doesn't allow sharing the dataset and the
repo must be public. `.gitignore` already excludes them. `results.json` contains only derived numbers.

## What it does

### Data (`swat_data.py`)
- **Stage 1 only, and only stage-1 attacks.** SWaT labels every attack in the whole plant, but 26 of the
  36 physical attacks never touch stage 1. Using the official attack list, we keep the 10 attacks that
  target MV-101, P-101, P-102 or LIT-101. Rows from other-stage attacks are **masked**: left out of
  training and scoring, instead of being wrongly called "normal".
- **Split by attack, in time order**, so test attacks are never seen in training or threshold tuning:

  | Split | Real data | Stage-1 attacks |
  | --- | --- | --- |
  | Train | Normal file (first 80%) + attack file up to 30 Dec 05:53 | #1 valve, #2 pump, #3 spoof, #21 valve |
  | Validation | Normal file (last 20%) + attack file to 1 Jan 03:14 | #26 pump, #30 pump |
  | **Test** | Attack file from 1 Jan 03:14 | **#33 spoof, #34 pump, #35 pump, #36 spoof** |

- First 5 hours of the normal file dropped (the tank is still filling from empty).
- Resampled from 1 s to 5 s; each prediction looks at the last 5 minutes (60 steps).
- **Physics feature:** a tank mass balance (`dLIT ≈ a·FIT101 − b·pump_on + c`) fitted on normal training
  data. The residual (how far the level moved from what physics expected) is an input to the model.

### Simulator (`simulator.py`)
One simulator of stage 1, one row per second. Each second it does four things:

1. **PLC logic** reads the *sensed* level: open MV101 at 500 mm, close at 800 mm; P101 follows downstream
   demand (on ~52 min, off ~18 min, varying ±40%) and stops below 250 mm (low-low interlock).
2. **Attacks** on the valve or pumps override the PLC.
3. **Physics:** level += inflow from FIT101 − 0.47 mm/s per running pump; the valve takes 8 s to move.
4. **Sensor:** LIT101 = true level + noise, unless an attack spoofs it. Because the PLC trusts the sensor,
   a spoof causes real physical effects (a low spoof overflows the tank, like real attack #36).

Every constant comes from real SWaT normal data (`simulator_calibration.json`; redo with
`python check_simulator.py --calibrate`), and each run varies them ±10% (domain randomization).
Compared with 72 h of the real plant, every statistic `check_simulator.py` prints is within about 10%.

Eight attack kinds, each modelled on real SWaT stage-1 attacks, at random times (every 1–3 h), lengths
(2–20 min) and sizes:

| Kind | What it does | Real SWaT attack |
| --- | --- | --- |
| `valve_open` / `valve_close` | Force MV101 open / closed | #1, #21 |
| `pump_on` / `pump_off` | Force P101 on / off | #26, #30 / #34, #35 |
| `backup_pump_on` | Turn on P102 too | #2 |
| `level_spoof_high` / `level_spoof_low` | LIT101 reads above H / below LL | #33 / #36 |
| `level_drift` | LIT101 creeps away from the truth | #3 |

`pump_on` matters: the validation attacks (#26, #30) are pump-on attacks, and no training attack is.

`python simulator.py --kinds pump_off pump_on` injects only some kinds (useful for ablations).
Output also has `LIT101_true` (the real level) for plots; the model never sees it.

Any other simulator works too if it provides:

```python
def simulate(n_seconds: int, seed: int, inject_anomalies: bool) -> pandas.DataFrame
# one row per second, SWaT units and codes:
#   FIT101 (m3/h), LIT101 (mm, as the SENSOR reports it), MV101 (0 moving / 1 closed / 2 open),
#   P101 and P102 (1 off / 2 on), label (0 normal / 1 attack), optional Timestamp
```

**Only simulating a few sensors?** Fine. `LIT101` and `label` are required; leave out any of `FIT101`,
`MV101`, `P101`, `P102` and `simulator.complete_columns()` fills them from the columns you do provide
(valve from flow or the level setpoints, flow from the valve, P101 from the mass balance, P102 off).
Tested on the stand-in: from `LIT101` alone it recovers the valve state 99.7% and P101 98% of the time.
Filled columns carry no attack signal, so your simulated attacks must show up in the columns you simulate.

Not in Python? Export two CSVs with those columns and run
`python run_experiments.py --sim-csv with_attacks.csv --sim-csv-noanom clean.csv`.

Attacks should be **randomized** (start, length and size), not on a fixed schedule, or the model learns
the timing instead of the attack. The stand-in injects six kinds: valve forced open, P102 turned on,
pump forced off, level spoofed high, level spoofed low, slow level drift.

### Model (`model.py`)
Two-layer LSTM backbone, one-layer LSTM adapter, and two heads:
- **Forecast head:** next-step FIT101 and LIT101, where LIT101 = physics prediction + learned correction.
  Trained on normal steps only.
- **Attack head:** probability a stage-1 attack is happening. **This is the anomaly score.**

Fine-tuning freezes the backbone and trains only the adapter and heads (`--no-freeze` trains everything).

### Experiments (`run_experiments.py`)

| Name | What | Answers |
| --- | --- | --- |
| `physics_only` | No ML: alert when the level disagrees with the mass balance | Is ML needed at all? |
| `real_only` | LSTM on real SWaT only | Baseline |
| `sim_ft` | Pretrain on simulator **with** attacks, fine-tune on real | RQ1: does simulation help? |
| `sim_noanom_ft` | Pretrain on simulator **without** attacks, fine-tune on real | RQ2: do simulated attacks matter? |
| `sim_only` | Simulator only | How big is the sim-to-real gap? |

Useful flags: `--seeds 0 1 2` (mean ± std), `--fractions 0.25 0.5 1.0` (how much labeled real data is
needed, RQ2), `--epochs`, `--sim-hours`, `--max-false-alarms`.

### Scoring (`evaluate.py`)
- **Threshold:** set on validation data only, at the most sensitive level that keeps validation false
  alarms at or below **2 per day** (`--max-false-alarms`). That's how an operator would set it.
  `--threshold-rule f1` uses the best-F1 threshold instead, but it can produce hundreds of false alarms a day.
- **Reported first (attack level):** test attacks caught (of 4), detection delay, false alarms per day.
- **Also reported:** point-wise precision / recall / F1, and point-adjusted F1. Point-adjust credits a
  whole attack when one step is flagged; many SWaT papers use it, but it inflates results. Label it if you show it.

## First real result (physics check only)

At 2 false alarms/day on validation, the mass-balance check alone **caught both spoofing attacks
(#33 in 50 s, #36 in 65 s) and missed both pump-off attacks (#34, #35)**, with 0.7 false alarms per day
on test. The question for the LSTM is whether it catches the pump attacks that physics can't.

## Threat console dashboard (`dashboard/index.html`)

A standalone web page (no build step, no install) that replays the held-out test period from
`results.json`:

- **Process view:** tank T-101, inlet valve and pumps animate live. The tank shows two lines: what the
  LIT101 sensor reports (solid) and what physics says the level should be (dashed). During a spoofing
  attack they visibly split.
- **Telemetry:** a 2-hour window of LIT101, FIT101, valve/pump states and the anomaly score against the
  alarm threshold.
- **Incidents:** an alarm banner, a beep, and an incident card each time the score crosses the threshold,
  tagged with the likely MITRE ATT&CK for ICS technique (T0856 Spoof Reporting Message, or T0855
  Unauthorized Command Message for valve/pump anomalies).
- **Phone alerts:** click **Alerts**, enter an ntfy topic, tick "Push incidents to phones", and press
  **Send test alert**. Every incident then buzzes phones subscribed to that topic in the ntfy app.
- **Detector scorecard:** every experiment in `results.json`; click a row to replay with that detector.
- **Ground truth toggle:** shows the real attacks (violet) so judges can see what was caught and missed.
  Other-stage attacks are hatched amber: they're outside this detector's scope.

- **Briefing mode:** the **Briefing** button in the header switches to full-screen results slides for the
  pitch, all drawn from `results.json` (no PowerPoint to keep in sync): the question, the train/test
  timeline, a detector × test-attack matrix with detection delays, attacks caught vs false alarms,
  spoof vs pump traces against physics, training curves (once LSTM runs exist) and point-wise vs
  point-adjusted F1. Arrow keys move between slides; **Console** goes back to the live replay.

Run it:

```bash
python -m http.server 8000      # from the swat_pipeline folder
# open http://localhost:8000/dashboard/   (loads ../results/results.json automatically)
```

Opening the file directly also works: it asks you to pick `results.json`. Keyboard: space = play/pause,
`n` = next incident. Demo tip: pick 900× speed and press **Next incident**.

## `results.json` (for the dashboard)

```ts
interface Results {
  generated_at: string;
  smoke_test: boolean;            // true = numbers are meaningless; show a warning
  simulator?: string;
  data: { splits: { train: Split; val: Split; test: Split }; attacks: Attack[];
          mass_balance: { a: number; b: number; c: number }; sample_seconds: number };
  experiments: {
    name: string; label: string; fraction: number;
    mean: Metrics; std: Partial<Metrics>;
    runs: { seed: number; metrics: Metrics & { per_attack: PerAttack[] };
            history: Record<string, { epoch: number; train_loss: number; val_loss: number }[]> }[];
  }[];
  replay: Record<string, {        // test period, every 15 s; one entry per experiment
    threshold: number; t: string[]; score: number[];
    FIT101: number[]; LIT101: number[]; MV101: number[]; P101: number[]; P102: number[];
    label: number[]; masked: number[]; attack_id: number[];
  }>;
}
interface Metrics { attacks_detected: number; attacks_total: number; mean_delay_s: number | null;
  false_alarms_per_day: number; pointwise_precision: number; pointwise_recall: number;
  pointwise_f1: number; point_adjusted_f1: number; lit101_forecast_rmse_mm: number | null; }
interface PerAttack { attack_id: number; detected: boolean; delay_s: number | null; duration_s: number; }
interface Attack { id: number; start: string; end: string; point: string; kind: string; what: string; }
interface Split { attack_ids: number[]; segments: { name: string; start: string; end: string; rows: number }[] }
```

The included `results/results.json` is a **real** run with the physics baseline only, so the dashboard
can be built now. It gains the LSTM experiments when you run step 3.

## Limitations
- Four test attacks: one missed attack changes the score by 25%. Use several seeds and report per-attack results.
- SWaT is a research testbed, not an Alberta utility.
- One plant stage, one simplified simulator.
- Simulated attacks are our assumptions about attacks; real ones may differ. That's why we test only on real attacks.

## Team (pending)
| Name | Role |
| --- | --- |
| **Oyonti** | Team lead, evaluation design |
| **Dev** | LSTM model |
| **Zyle** | Data pipeline |
| **Vanya** | Attack synthesis and the simulator |
| **Neila** | Threat console |
| **Ashar** | Research and pitch |
