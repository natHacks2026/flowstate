"""Stage-1 simulator (raw water tank + inlet valve MV101 + pumps P101/P102).

This is the project's simulator: calibrated from real SWaT normal data, with attacks
modelled on the real stage-1 attacks. Try it on its own:

    python simulator.py                    # 24 h with attacks -> sim_out/ (CSV, attack log, plot)
    python simulator.py --hours 72 --clean # also a run without attacks

Any other simulator also works with the pipeline if it follows this contract:

    simulate(n_seconds: int, seed: int, inject_anomalies: bool) -> pandas.DataFrame

with one row per second and these columns, in SWaT's own units and codes:
    FIT101  inlet flow, m3/h
    LIT101  tank level as the SENSOR reports it, mm   (spoofed during spoof attacks)
    MV101   inlet valve: 0 = moving, 1 = closed, 2 = open
    P101    main pump:   1 = off, 2 = on
    P102    backup pump: 1 = off, 2 = on
    label   0 = normal, 1 = attack
    (optional) Timestamp

Only simulating a few sensors? That's fine. LIT101 and label are required; leave out any
of FIT101, MV101, P101, P102 and complete_columns() fills them in from the ones you
do provide (see its docstring). The pipeline prints which columns it filled.

Put it in its own file and run:   python run_experiments.py --sim-module my_sim.py
or export CSVs and run:           python run_experiments.py --sim-csv sim.csv --sim-csv-noanom sim_clean.csv
Check it against the real plant: python check_simulator.py --sim-module my_sim.py
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Calibration:
    """Numbers measured from REAL SWaT normal data (see calibrate())."""
    fit_open: float = 2.55          # m3/h through MV101 when open
    fill_mm_s: float = 0.55         # level rise per second: valve open, pumps off
    drain_mm_s: float = 0.50        # level fall per second: valve closed, one pump on
    level_low: float = 500.0        # controller opens MV101 at about this level (L)
    level_high: float = 800.0       # controller closes MV101 at about this level (H)
    pump_on_s: float = 1500.0       # typical P101 run length, s
    pump_off_s: float = 900.0       # typical P101 rest length, s
    level_noise_mm: float = 0.3
    flow_noise: float = 0.02


def _run_lengths(x: np.ndarray, value) -> np.ndarray:
    is_v = (x == value).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, is_v, 0]))
    return edges[1::2] - edges[0::2]


def calibrate(normal_1s: pd.DataFrame) -> Calibration:
    """Measure the stand-in simulator's constants from real normal data (1 s rows)."""
    d = normal_1s
    dl = d["LIT101"].diff().shift(-1)
    mv, p1, p2 = d["MV101"], d["P101"], d["P102"]
    c = Calibration()
    c.fit_open = float(d.loc[mv == 2, "FIT101"].median())
    c.fill_mm_s = float(dl[(mv == 2) & (p1 == 1) & (p2 == 1)].median())
    c.drain_mm_s = float(-dl[(mv == 1) & ((p1 == 2) | (p2 == 2))].median())
    opens = (mv.shift(1) == 1) & (mv == 0) | (mv.shift(1) == 1) & (mv == 2)
    closes = (mv.shift(1) == 2) & (mv == 0) | (mv.shift(1) == 2) & (mv == 1)
    if opens.any():
        c.level_low = float(d.loc[opens, "LIT101"].median())
    if closes.any():
        c.level_high = float(d.loc[closes, "LIT101"].median())
    on = _run_lengths(p1.to_numpy(), 2)
    off = _run_lengths(p1.to_numpy(), 1)
    if len(on) > 2:
        c.pump_on_s = float(np.median(on))
    if len(off) > 2:
        c.pump_off_s = float(np.median(off))
    steady = (mv == 1) & (p1 == 1) & (p2 == 1)
    if steady.sum() > 100:
        c.level_noise_mm = float(np.clip(dl[steady].std() / np.sqrt(2), 0.05, 5))
    for k, v in asdict(c).items():   # fall back to defaults if a mode never occurred
        if not np.isfinite(v):
            setattr(c, k, getattr(Calibration(), k))
    return c


_CALIB_FILE = Path(__file__).with_name("simulator_calibration.json")


def load_calibration() -> Calibration:
    if _CALIB_FILE.exists():
        return Calibration(**json.loads(_CALIB_FILE.read_text()))
    return Calibration()


def save_calibration(c: Calibration) -> None:
    _CALIB_FILE.write_text(json.dumps(asdict(c), indent=2))


# Attack kinds, each modelled on real SWaT stage-1 attacks (List_of_attacks_Final.xlsx)
ATTACKS = {
    "valve_open":       "force MV101 open; tank overfills                (real #1, #21)",
    "valve_close":      "force MV101 closed; tank drains",
    "pump_on":          "force P101 on; tank drains, even below LL       (real #26, #30)",
    "pump_off":         "force P101 off; tank fills when it should drain (real #34, #35)",
    "backup_pump_on":   "turn P102 on as well as P101                    (real #2)",
    "level_spoof_high": "LIT101 reads above H; PLC closes the valve      (real #33)",
    "level_spoof_low":  "LIT101 reads below LL; PLC opens valve, stops pump (real #36)",
    "level_drift":      "LIT101 creeps 0.5-1.5 mm/s away from the truth  (real #3)",
}
ATTACK_KINDS = list(ATTACKS)


def simulate(n_seconds: int, seed: int = 0, inject_anomalies: bool = True,
             calib: Calibration | None = None, kinds: list[str] | None = None) -> pd.DataFrame:
    """Stage-1 simulator: tank mass balance + valve travel + PLC logic + attacks. One row per second.

    Columns: FIT101, LIT101 (what the SENSOR reports), MV101, P101, P102, label,
             LIT101_true (the real level; ignored by the model, used for plots).
    df.attrs["attacks"] lists every injected attack: start_s, end_s, kind.
    """
    c = calib or load_calibration()
    rng = np.random.default_rng(seed)
    kinds = kinds or ATTACK_KINDS

    # Domain randomization: every run gets slightly different plant constants (+-10%),
    # so the LSTM learns the pattern rather than one exact plant.
    jit = lambda: rng.uniform(0.9, 1.1)
    fit_open, fill, drain = c.fit_open * jit(), c.fill_mm_s * jit(), c.drain_mm_s * jit()
    per_flow = fill / max(fit_open, 1e-6)            # mm/s of level per m3/h of inflow
    L, H, LL, HH = c.level_low, c.level_high, 250.0, 1100.0
    valve_travel_s = 8                               # MV101 takes ~8 s to open or close

    # Initial state
    level = rng.uniform(L, H)
    sensed = level
    valve_cmd = rng.random() < 0.5                    # True = open
    valve_pos = float(valve_cmd)                      # 0 closed .. 1 open
    # downstream demand switches P101 on and off; run lengths vary +-40% around the real median
    run_len = lambda med: med * rng.lognormal(0.0, 0.4)
    demand, demand_left = rng.random() < 0.5, run_len(c.pump_on_s)

    # Schedule attacks at random times, lengths and sizes
    attacks = []
    if inject_anomalies:
        t = int(rng.uniform(600, 3600))
        while t < n_seconds - 300:
            length = int(rng.uniform(120, 1200))       # real SWaT stage-1 attacks: 2 to 20 min
            attacks.append(dict(start_s=t, end_s=min(t + length, n_seconds),
                                kind=str(rng.choice(kinds)), size=float(rng.uniform(0, 1))))
            t += length + int(rng.uniform(3600, 3 * 3600))

    out = np.zeros((n_seconds, 7), dtype=np.float32)
    ai, drift = 0, 0.0
    for t in range(n_seconds):
        while ai < len(attacks) and t >= attacks[ai]["end_s"]:
            ai, drift = ai + 1, 0.0
        a = attacks[ai] if ai < len(attacks) and attacks[ai]["start_s"] <= t else None
        if a and t == a["start_s"]:
            # an attack that changes nothing teaches nothing: flip it if the plant is already there
            flip = {"valve_open": ("valve_close", valve_cmd), "valve_close": ("valve_open", not valve_cmd),
                    "pump_on": ("pump_off", demand), "pump_off": ("pump_on", not demand)}
            if a["kind"] in flip and flip[a["kind"]][1] and flip[a["kind"]][0] in kinds:
                a["kind"] = flip[a["kind"]][0]
        kind = a["kind"] if a else None

        # 1. PLC logic, acting on the SENSED level (so a spoofed sensor fools it)
        if sensed >= H:
            valve_cmd = False
        elif sensed <= L:
            valve_cmd = True
        demand_left -= 1                              # downstream stage asks for water on and off
        if demand_left <= 0:
            demand = not demand
            demand_left = run_len(c.pump_on_s if demand else c.pump_off_s)
        p101 = demand and sensed > LL                 # low-low interlock protects the pump
        p102 = False

        # 2. Attacks on actuators override the PLC
        if kind == "valve_open":
            valve_cmd = True
        elif kind == "valve_close":
            valve_cmd = False
        elif kind == "pump_on":
            p101 = True
        elif kind == "pump_off":
            p101 = False
        elif kind == "backup_pump_on":
            p102 = True

        # 3. Physics
        step = 1.0 / valve_travel_s
        valve_pos = min(1.0, valve_pos + step) if valve_cmd else max(0.0, valve_pos - step)
        q_in = max(0.0, fit_open * valve_pos + rng.normal(0, c.flow_noise))
        pumping = (int(p101) + int(p102)) * drain if level > 0 else 0.0
        level = float(np.clip(level + per_flow * q_in - pumping + rng.normal(0, 0.05), 0.0, HH))

        # 4. Sensor (attacks on the sensor make it lie)
        sensed = level + rng.normal(0, c.level_noise_mm)
        if kind == "level_spoof_high":
            sensed = H + 20 + 180 * a["size"]
        elif kind == "level_spoof_low":
            sensed = max(0.0, LL - 10 - 140 * a["size"])
        elif kind == "level_drift":
            drift += (0.5 + a["size"]) * (1 if a["size"] >= 0.5 else -1)
            sensed = level + drift

        mv_code = 2 if valve_pos >= 0.999 else (1 if valve_pos <= 0.001 else 0)
        out[t] = [q_in, sensed, mv_code, 2 if p101 else 1, 2 if p102 else 1, 1.0 if kind else 0.0, level]

    df = pd.DataFrame(out, columns=["FIT101", "LIT101", "MV101", "P101", "P102", "label", "LIT101_true"])
    for col in ("MV101", "P101", "P102", "label"):
        df[col] = df[col].astype(int)
    df.attrs["attacks"] = [{k: v for k, v in a.items() if k != "size"} for a in attacks]
    return df


REQUIRED = ("LIT101", "label")
ALL_COLS = ("FIT101", "LIT101", "MV101", "P101", "P102", "label")


def complete_columns(df: pd.DataFrame, calib: Calibration | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Fill stage-1 columns a partial simulator left out. Returns (frame, filled column names).

    Rules, using the real plant's calibrated constants:
      MV101   from FIT101 (open if flow > 0.5 m3/h), else from P101/level trend, else open below the
              low setpoint / closed above the high setpoint (the PLC rule)
      FIT101  fit_open when MV101 is open, 0 when closed, half while moving
      P101    from the mass balance: on when the level falls faster than inflow explains
      P102    off (the backup pump is off almost all the time in real SWaT)
    Filled columns carry no attack signal of their own, so attacks the simulator injects
    must show up in the columns it does provide.
    """
    c = calib or load_calibration()
    missing_req = [k for k in REQUIRED if k not in df.columns]
    if missing_req:
        raise ValueError(f"simulator output must include {list(REQUIRED)}; missing {missing_req}")
    d = df.copy()
    filled = [k for k in ALL_COLS if k not in d.columns]
    lit = d["LIT101"].astype(float)
    if "MV101" not in d.columns:
        if "FIT101" in d.columns:
            d["MV101"] = np.where(d["FIT101"] > 0.5, 2, 1)
        else:  # PLC hysteresis on the reported level
            mv, state = np.empty(len(d), dtype=int), 2
            for i, v in enumerate(lit.to_numpy()):
                state = 1 if v >= c.level_high else (2 if v <= c.level_low else state)
                mv[i] = state
            d["MV101"] = mv
    if "FIT101" not in d.columns:
        d["FIT101"] = np.select([d["MV101"] == 2, d["MV101"] == 0], [c.fit_open, c.fit_open / 2], 0.0)
    if "P101" not in d.columns:
        per_flow = c.fill_mm_s / max(c.fit_open, 1e-6)
        slope = lit.diff().rolling(15, min_periods=1, center=True).mean().fillna(0)
        unexplained = slope - per_flow * d["FIT101"]
        d["P101"] = np.where(unexplained < -c.drain_mm_s / 2, 2, 1)
    if "P102" not in d.columns:
        d["P102"] = 1
    for col in ("MV101", "P101", "P102"):
        d[col] = d[col].round().astype(int)
    d["label"] = d["label"].astype(int)
    return d, filled


def load_sim_fn(path: str | None):
    """Return (simulate function, description). path = your team's simulator .py file."""
    if not path:
        return simulate, "stage-1 simulator (simulator.py)"
    spec = importlib.util.spec_from_file_location("team_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "simulate"):
        raise AttributeError(f"{path} must define simulate(n_seconds, seed, inject_anomalies)")
    return mod.simulate, f"team simulator ({Path(path).name})"


# ----------------------------------------------------------------------------- command line
def plot_run(df: pd.DataFrame, path: Path, hours: float | None = 6) -> None:
    """Save a picture of the first `hours` of a run: level (sensor vs truth), flow, actuators."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.iloc[: int(hours * 3600)] if hours else df
    t = np.arange(len(d)) / 3600
    fig, ax = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True, height_ratios=[3, 1.3, 1.3])
    for a in df.attrs.get("attacks", []):
        if a["start_s"] < len(d):
            for x in ax:
                x.axvspan(a["start_s"] / 3600, min(a["end_s"], len(d)) / 3600, color="#A08BFF", alpha=.18, lw=0)
            ax[0].text(a["start_s"] / 3600, 1060, a["kind"], fontsize=8, color="#5b45c9", va="top")
    ax[0].plot(t, d["LIT101"], lw=1.2, color="#0a7d96", label="LIT101 (what the sensor reports)")
    if "LIT101_true" in d:
        ax[0].plot(t, d["LIT101_true"], lw=1, ls="--", color="#333", label="true level")
    ax[0].set_ylabel("level (mm)"); ax[0].set_ylim(0, 1100); ax[0].legend(loc="lower right", fontsize=8)
    ax[1].plot(t, d["FIT101"], lw=1, color="#0a7d96"); ax[1].set_ylabel("FIT101 (m3/h)")
    ax[2].step(t, (d["MV101"] == 2) + 2.2, lw=1, label="MV101 open")
    ax[2].step(t, (d["P101"] == 2) + 1.1, lw=1, label="P101 on")
    ax[2].step(t, (d["P102"] == 2) + 0.0, lw=1, label="P102 on")
    ax[2].set_yticks([]); ax[2].legend(loc="upper right", fontsize=8, ncol=3); ax[2].set_xlabel("hours")
    fig.suptitle("Simulated SWaT stage 1 (purple = injected attack)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the stage-1 simulator and save CSVs, an attack log and a plot.")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim_out")
    ap.add_argument("--clean", action="store_true", help="also save a run without attacks")
    ap.add_argument("--kinds", nargs="+", choices=ATTACK_KINDS, help="only inject these attack kinds")
    ap.add_argument("--plot-hours", type=float, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = int(a.hours * 3600)
    print(f"Simulating {a.hours:g} h (seed {a.seed})...", flush=True)
    df = simulate(n, seed=a.seed, inject_anomalies=True, kinds=a.kinds)
    df.to_csv(out / "with_attacks.csv", index=False)
    log = pd.DataFrame(df.attrs["attacks"])
    log.to_csv(out / "attacks.csv", index=False)
    plot_run(df, out / "preview.png", a.plot_hours)
    print(f"  {len(log)} attacks, {df['label'].mean():.1%} of rows labelled attack")
    if len(log):
        print("  " + ", ".join(f"{k} x{v}" for k, v in log["kind"].value_counts().items()))
    if a.clean:
        simulate(n, seed=a.seed + 1, inject_anomalies=False).to_csv(out / "clean.csv", index=False)
    print(f"Saved to {out}/: with_attacks.csv, attacks.csv, preview.png" + (", clean.csv" if a.clean else ""))


if __name__ == "__main__":
    main()
