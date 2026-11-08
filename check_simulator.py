"""Compare a simulator with the real SWaT stage 1, side by side. Also (re)calibrates
the reference simulator.

    python check_simulator.py                          # reference simulator
    python check_simulator.py --sim-module pump_sim.py # your team's simulator
    python check_simulator.py --calibrate              # measure real plant, save simulator_calibration.json

Look for big gaps in the table: the closer the simulator is to the real plant, the
more pretraining on it should help.
"""
import argparse

import pandas as pd

import simulator as ref
import swat_data as sd


def stats(df_1s: pd.DataFrame) -> dict:
    c = ref.calibrate(df_1s)
    return {
        "FIT101 when valve open (m3/h)": c.fit_open,
        "Level rise, valve open + pumps off (mm/s)": c.fill_mm_s,
        "Level fall, valve closed + pump on (mm/s)": c.drain_mm_s,
        "Valve opens at level (mm)": c.level_low,
        "Valve closes at level (mm)": c.level_high,
        "Median P101 run (s)": c.pump_on_s,
        "Median P101 rest (s)": c.pump_off_s,
        "LIT101 min / max (mm)": f"{df_1s['LIT101'].min():.0f} / {df_1s['LIT101'].max():.0f}",
        "Share of time valve open": float((df_1s["MV101"] == 2).mean()),
        "Share of time P101 on": float((df_1s["P101"] == 2).mean()),
        "Share of time P102 on": float((df_1s["P102"] == 2).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sim-module")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    normal, _ = sd.load_real(a.data)
    normal = normal[normal["Timestamp"] >= normal["Timestamp"].min() + pd.Timedelta(hours=sd.DROP_STARTUP_HOURS)]
    normal = normal.reset_index(drop=True)
    if a.calibrate:
        c = ref.calibrate(normal)
        ref.save_calibration(c)
        print("Saved simulator_calibration.json:", c)

    fn, source = ref.load_sim_fn(a.sim_module)
    sim, filled = ref.complete_columns(fn(n_seconds=int(a.hours * 3600), seed=7, inject_anomalies=False))
    if filled:
        print(f"Simulator did not provide {filled}; filled from the other columns (rows for those are not a real comparison).")
    real_s, sim_s = stats(normal), stats(sim)
    print(f"\nReal SWaT normal vs {source} (no attacks, {a.hours:g} h)\n")
    print(f"{'':48s}{'real':>14s}{'simulator':>14s}")
    for k in real_s:
        fmt = lambda v: f"{v:14.3f}" if isinstance(v, float) else f"{v:>14s}"
        print(f"{k:48s}{fmt(real_s[k])}{fmt(sim_s[k])}")


if __name__ == "__main__":
    main()
