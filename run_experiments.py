"""Run the experiments and write results/results.json for the dashboard.

Experiments (one simulator):
  physics_only     no ML: flag when LIT101 disagrees with the tank mass balance  (baseline)
  real_only        LSTM trained on real SWaT only                                (baseline)
  sim_ft           pretrain on simulator WITH injected attacks -> fine-tune on real  (RQ1)
  sim_noanom_ft    pretrain on simulator WITHOUT attacks -> fine-tune on real        (RQ2)
  sim_only         simulator only, no real training data                           (sim-to-real gap)

All share the same model, settings, threshold rule and the same held-out test attacks.

Examples:
  python run_experiments.py --smoke                              # 2-minute check that everything runs
  python run_experiments.py                                      # main run (reference simulator)
  python run_experiments.py --sim-module ../team_sim/pump_sim.py # your team's simulator
  python run_experiments.py --fractions 0.25 0.5 1.0 --seeds 0 1 2   # label-amount study
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev
import swat_data as sd
from simulator import complete_columns, load_sim_fn

EXPERIMENTS = {
    "physics_only": "Physics check only (no ML)",
    "real_only": "Real SWaT only",
    "sim_ft": "Simulator (with attacks) + fine-tune on real",
    "sim_noanom_ft": "Simulator (no attacks) + fine-tune on real",
    "sim_only": "Simulator only (no real training)",
}


# ----------------------------------------------------------------------------- simulator data
def sim_segments(df_1s: pd.DataFrame, prep: sd.Prep, name: str) -> tuple[list, list]:
    df, filled = complete_columns(df_1s)
    if filled:
        print(f"  {name}: simulator did not provide {filled}; filled from the other columns", flush=True)
    df["masked"] = 0
    # number each simulated attack so event metrics work on sim data too
    starts = (df["label"].diff().fillna(df["label"]) == 1).cumsum()
    df["attack_id"] = np.where(df["label"] == 1, starts, 0)
    rs = prep.transform(sd.resample(df))
    cut = int(len(rs) * 0.8)
    return [sd.Segment(f"{name}_train", rs.iloc[:cut])], [sd.Segment(f"{name}_val", rs.iloc[cut:])]


def make_sim_data(args, prep: sd.Prep):
    if args.sim_csv:
        with_att = pd.read_csv(args.sim_csv)
        clean = pd.read_csv(args.sim_csv_noanom) if args.sim_csv_noanom else None
        source = f"team simulator CSV ({Path(args.sim_csv).name})"
    else:
        fn, source = load_sim_fn(args.sim_module)
        n = int(args.sim_hours * 3600)
        print(f"Simulating {args.sim_hours} h with attacks and {args.sim_hours} h without ({source})...", flush=True)
        with_att = fn(n_seconds=n, seed=1000, inject_anomalies=True)
        clean = fn(n_seconds=n, seed=2000, inject_anomalies=False)
    out = {"with": sim_segments(with_att, prep, "sim")}
    if clean is not None:
        out["clean"] = sim_segments(clean, prep, "sim_clean")
    return out, source


# ----------------------------------------------------------------------------- one run
def subsample(w: dict, n: int, seed: int) -> dict:
    if n <= 0 or len(w["x"]) <= n:
        return w
    idx = np.sort(np.random.default_rng(seed).choice(len(w["x"]), n, replace=False))
    return {k: v[idx] for k, v in w.items()}


def physics_score(w: dict) -> np.ndarray:
    """Largest |mass-balance residual| over the last minute (12 steps), in z units."""
    return np.abs(w["x"][:, -12:, sd.FEATURES.index("mb_resid_z")]).max(axis=1)


def p1_lockout_for_df(df: pd.DataFrame, min_run_s: float = 500.0) -> np.ndarray:
    """Causal premature pump abort detection:
    Tracks P101 run time forward in time. When P101 shuts off prematurely (< 500s),
    flags from the shut-off edge through the entire subsequent lockout rest period.
    Does not use groupby or look ahead into future steps."""
    if "P101" not in df:
        return np.zeros(len(df), dtype=float)

    p1 = (df["P101"] == 2).to_numpy()
    n = len(p1)
    is_lockout = np.zeros(n, dtype=float)
    current_run_s = 0.0
    saw_rising_edge = False
    in_lockout = False

    for t in range(n):
        if p1[t]:
            if t > 0 and not p1[t - 1]:
                saw_rising_edge = True
                current_run_s = 0.0
            in_lockout = False
            if saw_rising_edge:
                current_run_s += sd.SAMPLE_SECONDS
        else:
            if t > 0 and p1[t - 1] and saw_rising_edge:
                if current_run_s < min_run_s:
                    in_lockout = True
                current_run_s = 0.0
                saw_rising_edge = False
            if in_lockout:
                is_lockout[t] = 1.0

    return is_lockout


def invariant_score(w: dict) -> np.ndarray:
    """Returns 1.0 if a known deterministic ICS invariant is broken:
    1. Auxiliary pump P102 runs (impossible in normal plant operation).
    2. Premature pump trip / lockout (P101 aborted < 500s; normal min is 659s).
    """
    p102_idx = sd.FEATURES.index("P102_on")
    p102_active = (w["x"][:, -1, p102_idx] > 0.5).astype(float)
    lockout = w.get("p1_lockout", np.zeros(len(w["x"]), dtype=float))
    return np.maximum(p102_active, lockout)


def run_one(name, seed, real_tr, real_va, test_w, sim, args):
    if name == "physics_only":
        # 1. Mass balance physics check only
        va_raw = physics_score(real_va)
        te_raw = physics_score(test_w)
        thr_raw = ev.pick_threshold(va_raw, real_va["label"], real_va["masked"], real_va["attack_id"],
                                    sd.SAMPLE_SECONDS, args.threshold_rule, args.max_false_alarms)
        rep_raw = ev.full_report(te_raw, test_w["label"], test_w["masked"], test_w["attack_id"], thr_raw, sd.SAMPLE_SECONDS)
        rep_raw["lit101_forecast_rmse_mm"] = None
        run_raw = dict(seed=seed, metrics=rep_raw, history={}, test_prob=te_raw)

        # 2. Physics check + invariant rules
        va_hyb = np.maximum(va_raw, invariant_score(real_va) * 5.0)
        te_hyb = np.maximum(te_raw, invariant_score(test_w) * 5.0)
        thr_hyb = ev.pick_threshold(va_hyb, real_va["label"], real_va["masked"], real_va["attack_id"],
                                    sd.SAMPLE_SECONDS, args.threshold_rule, args.max_false_alarms)
        rep_hyb = ev.full_report(te_hyb, test_w["label"], test_w["masked"], test_w["attack_id"], thr_hyb, sd.SAMPLE_SECONDS)
        rep_hyb["lit101_forecast_rmse_mm"] = None
        run_hyb = dict(seed=seed, metrics=rep_hyb, history={}, test_prob=te_hyb)
        return run_raw, run_hyb

    import torch  # only needed for the LSTM experiments
    from model import Detector, device, score, train

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Detector().to(device())
    hist = {}
    if name == "real_only":
        hist["train_real"] = train(model, real_tr, real_va, use_cls=True, lr=1e-3,
                                   max_epochs=args.epochs, seed=seed, tag=f"[{name} s{seed}]")
    else:
        key = "clean" if name == "sim_noanom_ft" else "with"
        if key not in sim:
            print(f"  skipping {name}: no attack-free simulator data")
            return None
        s_tr, s_va = sim[key]
        hist["pretrain_sim"] = train(model, s_tr, s_va, use_cls=(key == "with"), lr=1e-3,
                                     max_epochs=args.epochs, seed=seed, tag=f"[{name} s{seed} pretrain]")
        if name != "sim_only":
            if not args.no_freeze:
                model.freeze_backbone()
            hist["finetune_real"] = train(model, real_tr, real_va, use_cls=True, lr=3e-4,
                                          max_epochs=args.epochs, seed=seed, tag=f"[{name} s{seed} fine-tune]")

    va_s = score(model, real_va)
    te_s = score(model, test_w)

    normal = (test_w["label"] == 0) & (test_w["masked"] == 0)
    rmse = float(np.sqrt(np.mean(te_s["lit_err"][normal] ** 2)) * args._lit_std)

    # 1. Pure ML model only (no invariant rule override)
    thr_raw = ev.pick_threshold(va_s["prob"], real_va["label"], real_va["masked"], real_va["attack_id"],
                                sd.SAMPLE_SECONDS, args.threshold_rule, args.max_false_alarms)
    rep_raw = ev.full_report(te_s["prob"], test_w["label"], test_w["masked"], test_w["attack_id"],
                             thr_raw, sd.SAMPLE_SECONDS)
    rep_raw["lit101_forecast_rmse_mm"] = rmse
    run_raw = dict(seed=seed, metrics=rep_raw, history=hist, test_prob=te_s["prob"])

    # 2. Hybrid: ML model + deterministic invariant rules
    va_hyb = np.maximum(va_s["prob"], invariant_score(real_va))
    te_hyb = np.maximum(te_s["prob"], invariant_score(test_w))
    thr_hyb = ev.pick_threshold(va_hyb, real_va["label"], real_va["masked"], real_va["attack_id"],
                                sd.SAMPLE_SECONDS, args.threshold_rule, args.max_false_alarms)
    rep_hyb = ev.full_report(te_hyb, test_w["label"], test_w["masked"], test_w["attack_id"],
                             thr_hyb, sd.SAMPLE_SECONDS)
    rep_hyb["lit101_forecast_rmse_mm"] = rmse
    run_hyb = dict(seed=seed, metrics=rep_hyb, history=hist, test_prob=te_hyb)

    return run_raw, run_hyb


def windows_for(segs, stride=1):
    w = sd.windows(segs, stride=stride)
    # attack ids aligned to windows (needed for event metrics)
    ids = np.concatenate([s.df["attack_id"].to_numpy() for s in segs])
    offsets = np.cumsum([0] + [len(s.df) for s in segs])[:-1]
    w["attack_id"] = ids[offsets[w["seg"]] + w["t"]]
    lockouts = np.concatenate([p1_lockout_for_df(s.df) for s in segs])
    w["p1_lockout"] = lockouts[offsets[w["seg"]] + w["t"]]
    return w


def aggregate(runs: list[dict]) -> tuple[dict, dict]:
    keys = ["attacks_detected", "attacks_total", "mean_delay_s", "false_alarms_per_day",
            "pointwise_precision", "pointwise_recall", "pointwise_f1", "pointwise_accuracy", "point_adjusted_f1",
            "lit101_forecast_rmse_mm"]
    mean, std = {}, {}
    for k in keys:
        vals = [r["metrics"][k] for r in runs if r["metrics"].get(k) is not None]
        mean[k] = float(np.mean(vals)) if vals else None
        std[k] = float(np.std(vals)) if len(vals) > 1 else None
    return mean, std


def replay_payload(test_seg: sd.Segment, w: dict, prob: np.ndarray, thr: float, every: int = 3) -> dict:
    df = test_seg.df.reset_index(drop=True)
    s = np.zeros(len(df))
    s[w["t"]] = prob
    s = pd.Series(s).rolling(every, min_periods=1).max().to_numpy()  # keep short spikes when thinning
    keep = np.arange(0, len(df), every)
    r = lambda a, d=2: [round(float(v), d) for v in a]
    return dict(
        threshold=round(float(thr), 4),
        t=[str(x) for x in df["Timestamp"].iloc[keep]],
        FIT101=r(df["FIT101"].iloc[keep], 3), LIT101=r(df["LIT101"].iloc[keep], 1),
        MV101=[int(x) for x in df["MV101"].iloc[keep]], P101=[int(x) for x in df["P101"].iloc[keep]],
        P102=[int(x) for x in df["P102"].iloc[keep]],
        label=[int(x) for x in df["label"].iloc[keep]], masked=[int(x) for x in df["masked"].iloc[keep]],
        attack_id=[int(x) for x in df["attack_id"].iloc[keep]], score=r(s[keep], 4),
    )


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--fractions", nargs="+", type=float, default=[1.0],
                    help="share of real TRAINING data to use (e.g. 0.25 0.5 1.0)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-stride", type=int, default=2, help="use every Nth training window")
    ap.add_argument("--sim-hours", type=float, default=72)
    ap.add_argument("--sim-module", help="python file with simulate(n_seconds, seed, inject_anomalies)")
    ap.add_argument("--sim-csv", help="simulator output CSV WITH attacks (instead of --sim-module)")
    ap.add_argument("--sim-csv-noanom", help="simulator output CSV WITHOUT attacks")
    ap.add_argument("--no-freeze", action="store_true", help="fine-tune all layers instead of freezing the backbone")
    ap.add_argument("--threshold-rule", choices=["false_alarms", "f1"], default="false_alarms",
                    help="how to set the alert threshold on validation data")
    ap.add_argument("--max-false-alarms", type=float, default=2.0,
                    help="false alarms per day allowed on validation (rule=false_alarms)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to check the code works")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.sim_hours = 1, 3
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = dict(generated_at=dt.datetime.now().isoformat(timespec="seconds"),
                   real_data="SWaT A1 (Dec 2015, iTrust SUTD), stage 1 only",
                   smoke_test=bool(args.smoke), config={k: v for k, v in vars(args).items()},
                   experiments=[], replay={})
    sim = None
    for frac in args.fractions:
        print(f"\n=== real training fraction {frac} ===", flush=True)
        segs, prep, summary = sd.build_real_splits(args.data, fraction=frac)
        args._lit_std = prep.std["LIT101"]
        if frac == 1.0 or "data" not in results:
            results["data"] = summary
        real_tr = windows_for(segs["train"], stride=args.train_stride)
        real_va = windows_for(segs["val"])
        test_w = windows_for(segs["test"])
        if args.smoke:
            real_tr, real_va = subsample(real_tr, 3000, 0), subsample(real_va, 3000, 0)
        if sim is None and any(e not in ("real_only", "physics_only") for e in args.experiments):
            raw, source = make_sim_data(args, prep)
            sim = {k: (windows_for(tr, stride=args.train_stride), windows_for(va)) for k, (tr, va) in raw.items()}
            if args.smoke:
                sim = {k: (subsample(a, 3000, 0), subsample(b, 3000, 0)) for k, (a, b) in sim.items()}
            results["simulator"] = source
        for name in args.experiments:
            runs_raw = []
            runs_hyb = []
            for seed in (args.seeds[:1] if name == "physics_only" else args.seeds):
                res = run_one(name, seed, real_tr, real_va, test_w, sim or {}, args)
                if res:
                    r_raw, r_hyb = res
                    runs_raw.append(r_raw)
                    runs_hyb.append(r_hyb)
            if not runs_raw:
                continue

            variants = [
                ("_raw", " (Model only)" if name != "physics_only" else " (Mass balance only)", runs_raw),
                ("", " (+ rules)", runs_hyb),
            ]
            for suffix, label_suffix, v_runs in variants:
                var_name = name + suffix if suffix else name
                var_label = EXPERIMENTS[name] + label_suffix
                mean, std = aggregate(v_runs)
                results["experiments"].append(dict(
                    name=var_name, label=var_label, fraction=frac,
                    real_train_attacks=results["data"]["splits"]["train"]["attack_ids"] if frac == 1.0 else
                    summary["splits"]["train"]["attack_ids"],
                    mean=mean, std=std,
                    runs=[dict(seed=r["seed"], metrics=r["metrics"], history=r["history"]) for r in v_runs]))
                m = v_runs[0]["metrics"]
                print(f"--> {var_label} (fraction {frac}, seed {v_runs[0]['seed']}): caught {m['attacks_detected']}/{m['attacks_total']} "
                      f"test attacks, {m['false_alarms_per_day']:.1f} false alarms/day | precision {m['pointwise_precision']:.3f}, "
                      f"recall {m['pointwise_recall']:.3f}, F1 {m['pointwise_f1']:.3f}, accuracy {m.get('pointwise_accuracy', 0):.3f}", flush=True)
                if frac == max(args.fractions):
                    results["replay"][var_name] = replay_payload(segs["test"][0], test_w, v_runs[0]["test_prob"],
                                                                 v_runs[0]["metrics"]["threshold"])

    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    write_summary(results, out / "summary.md")
    print(f"\nWrote {out / 'results.json'} and {out / 'summary.md'}")


def write_summary(results: dict, path: Path) -> None:
    lines = ["# Results (held-out test attacks: " +
             ", ".join(f"#{i}" for i in results["data"]["splits"]["test"]["attack_ids"]) + ")", ""]
    if results.get("smoke_test"):
        lines += ["**SMOKE TEST: these numbers are meaningless. Run without --smoke for real results.**", ""]
    lines += ["| Experiment | Real data | Attacks caught | Mean delay (s) | False alarms/day | Precision | Recall | F1 | Accuracy† | Point-adjusted F1* |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    fmt = lambda v, d=2: "–" if v is None else f"{v:.{d}f}"
    for e in results["experiments"]:
        m, s = e["mean"], e["std"]
        pm = lambda k, d=2: fmt(m[k], d) + (f" ± {s[k]:.{d}f}" if s.get(k) else "")
        lines.append(f"| {e['label']} | {int(e['fraction'] * 100)}% | {pm('attacks_detected', 1)} / {int(m['attacks_total'])} | "
                     f"{pm('mean_delay_s', 0)} | {pm('false_alarms_per_day', 1)} | {pm('pointwise_precision', 3)} | {pm('pointwise_recall', 3)} | "
                     f"{pm('pointwise_f1', 3)} | {pm('pointwise_accuracy', 3) if 'pointwise_accuracy' in m else '–'} | {pm('point_adjusted_f1', 3)} |")
    lines += ["", "*Point-adjusted F1 credits a whole attack if any single step is flagged; it inflates results "
              "and is shown only for comparison with published SWaT papers.",
              "†Accuracy is dominated by normal time (most test rows are normal), so a detector that never alarms still scores high. "
              "Precision, recall and F1 are per 5-second step on the held-out test days.", "",
              f"Simulator: {results.get('simulator', 'not used')}. Generated {results['generated_at']}."]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()