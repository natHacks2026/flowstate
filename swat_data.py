"""Loading, cleaning and splitting SWaT stage 1, plus the shared preprocessing.

Everything here is used for BOTH the real SWaT data and the simulator output, so the
model always sees data prepared the same way.

Key decisions (see README for the reasoning):
  * Stage 1 only: FIT101, LIT101, MV101, P101, P102.
  * Only the 10 attacks that target stage 1 count as attacks. SWaT labels every attack
    in the plant, so rows from other-stage attacks are MASKED (left out of training
    and scoring) instead of being called "normal" or "attack".
  * Splits are by attack episode and by time, so test attacks are never seen in
    training or threshold tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STAGE1 = ["FIT101", "LIT101", "MV101", "P101", "P102"]
SAMPLE_SECONDS = 5          # resample 1 s -> 5 s
WINDOW_STEPS = 60           # 60 x 5 s = 5 minutes of history per prediction
DROP_STARTUP_HOURS = 5.0    # normal file starts with the tank filling from empty

# From List_of_attacks_Final.xlsx: every attack whose attack point is in stage 1.
# (#26 and #30 also touch other stages but directly manipulate P-101 / LIT-101.)
STAGE1_ATTACKS = [
    dict(id=1,  start="2015-12-28 10:29:14", end="2015-12-28 10:44:53", point="MV-101",                 kind="valve",  what="Open MV-101 (tank overflow)"),
    dict(id=2,  start="2015-12-28 10:51:08", end="2015-12-28 10:58:30", point="P-102",                  kind="pump",   what="Turn on P-102 (pipe burst)"),
    dict(id=3,  start="2015-12-28 11:22:00", end="2015-12-28 11:28:22", point="LIT-101",                kind="spoof",  what="LIT-101 reading +1 mm every second"),
    dict(id=21, start="2015-12-29 18:30:00", end="2015-12-29 18:42:00", point="MV-101, LIT-101",        kind="valve",  what="Keep MV-101 open; spoof LIT-101"),
    dict(id=26, start="2015-12-30 17:04:56", end="2015-12-30 17:29:00", point="P-101, LIT-301",         kind="pump",   what="Keep P-101 on; spoof LIT-301"),
    dict(id=30, start="2015-12-31 15:47:40", end="2015-12-31 16:07:10", point="LIT-101, P-101, MV-201", kind="pump",   what="Keep P-101 on; spoof LIT-101; MV-201"),
    dict(id=33, start="2016-01-01 14:21:12", end="2016-01-01 14:28:35", point="LIT-101",                kind="spoof",  what="Set LIT-101 above H"),
    dict(id=34, start="2016-01-01 17:12:40", end="2016-01-01 17:14:20", point="P-101",                  kind="pump",   what="Turn P-101 off"),
    dict(id=35, start="2016-01-01 17:18:56", end="2016-01-01 17:26:56", point="P-101, P-102",           kind="pump",   what="Turn P-101 off; keep P-102 off"),
    dict(id=36, start="2016-01-01 22:16:01", end="2016-01-01 22:25:00", point="LIT-101",                kind="spoof",  what="Set LIT-101 below LL"),
]
# Chronological split of the stage-1 attacks.
SPLIT_ATTACKS = {"train": [1, 2, 3, 21], "val": [26, 30], "test": [33, 34, 35, 36]}


# ----------------------------------------------------------------------------- loading
def load_real(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the CSVs written by convert_swat.py (1-second rows)."""
    here = Path(__file__).resolve().parent
    tries = [Path(data_dir), here / data_dir, here / "data", here.parent / "data", here]

    def find(stem: str) -> Path:
        for d in tries:
            for name in (stem + ".csv.gz", stem + ".csv", stem + ".csv.gz.gz", stem + ".csv.csv"):
                if (d / name).is_file():
                    return d / name
        looked = "\n  ".join(str(d.resolve()) for d in tries)
        raise FileNotFoundError(f"Could not find {stem}.csv.gz. Looked in:\n  {looked}\n"
                                f"Put the file in one of these, or pass --data \"C:\\full\\path\\to\\folder\"")

    paths = {k: find(f"swat_{k}_stage1") for k in ("normal", "attack")}
    print(f"Data: {paths['normal'].parent}", flush=True)
    read = lambda p: pd.read_csv(p, parse_dates=["Timestamp"],
                                 compression="gzip" if p.read_bytes()[:2] == b"\x1f\x8b" else None)
    return read(paths["normal"]), read(paths["attack"])


def stage1_masks(attack_df: pd.DataFrame, tolerance_s: int = 0, recovery_s: int = 900) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (stage1_label, masked, attack_id) for each row of the 1 s attack file.

    stage1_label = 1 inside a stage-1 attack (official start/end times, kept only where
    SWaT's own label also says Attack). masked = 1 inside any OTHER attack plus a trailing
    recovery period to account for downstream hydraulic settling.
    """
    ts = attack_df["Timestamp"].to_numpy()
    plant_label = attack_df["label"].to_numpy().astype(bool)
    in_s1 = np.zeros(len(attack_df), dtype=bool)
    attack_id = np.zeros(len(attack_df), dtype=np.int16)
    tol = np.timedelta64(tolerance_s, "s")
    for a in STAGE1_ATTACKS:
        w = (ts >= np.datetime64(a["start"]) - tol) & (ts <= np.datetime64(a["end"]) + tol)
        in_s1 |= w
        attack_id[w] = a["id"]
    stage1_label = in_s1 & plant_label
    # rows SWaT labels as attack but that are not stage-1 attacks -> masked
    other_attacks = plant_label & ~in_s1
    if recovery_s > 0:
        # Extend the other-stage attack mask forward in time to cover the physical settling aftermath
        extended = pd.Series(other_attacks).rolling(int(recovery_s), min_periods=1).max().to_numpy().astype(bool)
        masked = extended & ~in_s1
    else:
        masked = other_attacks
    attack_id[~stage1_label] = 0
    return stage1_label.astype(np.int8), masked.astype(np.int8), attack_id


# ----------------------------------------------------------------------------- resampling + encoding
def resample(df: pd.DataFrame, seconds: int = SAMPLE_SECONDS) -> pd.DataFrame:
    """1 s -> `seconds` s. Mean for flows/levels, last value for actuators, max for flags.

    Works for real data and simulator output alike. Expects a Timestamp column
    (simulators without one get a synthetic clock).
    """
    df = df.copy()
    if "Timestamp" not in df:
        df["Timestamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(df)), unit="s")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    for col in ("MV101", "P101", "P102"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round()
    agg = {"FIT101": "mean", "LIT101": "mean", "MV101": "last", "P101": "last", "P102": "last"}
    for flag in ("label", "masked"):
        if flag in df:
            agg[flag] = "max"
    if "attack_id" in df:
        agg["attack_id"] = "max"
    out = df.set_index("Timestamp").resample(f"{seconds}s").agg(agg).dropna(subset=["LIT101"])
    return out.reset_index()


def encode_actuators(df: pd.DataFrame) -> pd.DataFrame:
    """SWaT codes: MV101 0=moving, 1=closed, 2=open; pumps 1=off, 2=on -> 0..1 values."""
    df = df.copy()
    df["MV_open"] = df["MV101"].map({0: 0.5, 1: 0.0, 2: 1.0}).fillna(0.5)
    df["P101_on"] = (df["P101"] == 2).astype(float)
    df["P102_on"] = (df["P102"] == 2).astype(float)
    return df


# ----------------------------------------------------------------------------- physics prior
@dataclass
class MassBalance:
    """Tank mass balance fitted on real normal data:  dLIT = a*FIT101 - b*pump_on + c  (per step)."""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0

    @staticmethod
    def fit(df: pd.DataFrame) -> "MassBalance":
        d = encode_actuators(df)
        dl = d["LIT101"].diff().shift(-1)  # change over the next step
        pump = np.clip(d["P101_on"] + d["P102_on"], 0, 1)
        X = np.column_stack([d["FIT101"], -pump, np.ones(len(d))])
        ok = dl.notna().to_numpy()
        coef, *_ = np.linalg.lstsq(X[ok], dl.to_numpy()[ok], rcond=None)
        return MassBalance(*map(float, coef))

    def predict_next(self, d: pd.DataFrame) -> np.ndarray:
        pump = np.clip(d["P101_on"] + d["P102_on"], 0, 1)
        return (d["LIT101"] + self.a * d["FIT101"] - self.b * pump + self.c).to_numpy()


# ----------------------------------------------------------------------------- features
CONT = ["FIT101", "LIT101", "mb_resid"]        # z-scored
FEATURES = ["FIT101_z", "LIT101_z", "MV_open", "P101_on", "P102_on", "mb_resid_z"]


@dataclass
class Prep:
    """Everything fitted on REAL normal training data and reused for all data."""
    mass_balance: MassBalance
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = encode_actuators(df)
        phys_next = self.mass_balance.predict_next(d)
        # residual: how much the level moved vs. what the mass balance expected
        d["mb_resid"] = d["LIT101"].to_numpy() - np.concatenate([[np.nan], phys_next[:-1]])
        d["mb_resid"] = d["mb_resid"].fillna(0.0)
        d["phys_next_LIT"] = phys_next
        for c in CONT:
            d[f"{c}_z"] = (d[c] - self.mean[c]) / self.std[c]
        d["phys_next_LIT_z"] = (d["phys_next_LIT"] - self.mean["LIT101"]) / self.std["LIT101"]
        for flag in ("label", "masked", "attack_id"):
            if flag not in d:
                d[flag] = 0
        return d

    @staticmethod
    def fit(normal_train: pd.DataFrame) -> "Prep":
        mb = MassBalance.fit(normal_train)
        p = Prep(mass_balance=mb, mean={c: 0.0 for c in CONT}, std={c: 1.0 for c in CONT})
        d = p.transform(normal_train)  # first pass to get mb_resid
        p.mean = {c: float(d[c].mean()) for c in CONT}
        p.std = {c: float(d[c].std() + 1e-6) for c in CONT}
        return p


# ----------------------------------------------------------------------------- splits
@dataclass
class Segment:
    """A continuous stretch of time (windows never cross segment boundaries)."""
    name: str
    df: pd.DataFrame   # resampled + transformed


def _cut_between(attack_rs: pd.DataFrame, end_of: int, start_of: int) -> pd.Timestamp:
    """A cut time halfway between two stage-1 attacks, moved out of any other attack."""
    e = pd.Timestamp(next(a["end"] for a in STAGE1_ATTACKS if a["id"] == end_of))
    s = pd.Timestamp(next(a["start"] for a in STAGE1_ATTACKS if a["id"] == start_of))
    cut = e + (s - e) / 2
    busy = attack_rs.set_index("Timestamp")[["label", "masked"]].max(axis=1)
    while cut < s and busy.asof(cut) > 0:
        cut += pd.Timedelta(minutes=5)
    return cut


def build_real_splits(data_dir: str | Path, fraction: float = 1.0) -> tuple[dict, Prep, dict]:
    """Return ({'train': [...], 'val': [...], 'test': [...]} segments, Prep, summary).

    fraction < 1 keeps only the first part of each TRAINING segment (to study how much
    labeled real data is needed). Validation and test never change.
    """
    normal, attack = load_real(data_dir)
    label, masked, aid = stage1_masks(attack)
    attack = attack.assign(label=label, masked=masked, attack_id=aid)
    normal = normal.assign(label=0, masked=0, attack_id=0)  # normal file has no attacks

    normal = normal[normal["Timestamp"] >= normal["Timestamp"].min() + pd.Timedelta(hours=DROP_STARTUP_HOURS)]
    n_rs = resample(normal)
    a_rs = resample(attack)

    cut_n = int(len(n_rs) * 0.8)
    normal_train, normal_val = n_rs.iloc[:cut_n], n_rs.iloc[cut_n:]
    c1 = _cut_between(a_rs, SPLIT_ATTACKS["train"][-1], SPLIT_ATTACKS["val"][0])
    c2 = _cut_between(a_rs, SPLIT_ATTACKS["val"][-1], SPLIT_ATTACKS["test"][0])
    a_train = a_rs[a_rs["Timestamp"] < c1]
    a_val = a_rs[(a_rs["Timestamp"] >= c1) & (a_rs["Timestamp"] < c2)]
    a_test = a_rs[a_rs["Timestamp"] >= c2]

    prep = Prep.fit(normal_train)  # fitted on the FULL normal train, independent of fraction

    def head(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[: max(WINDOW_STEPS + 2, int(len(df) * fraction))]

    segs = {
        "train": [Segment("normal_train", prep.transform(head(normal_train))),
                  Segment("attack_train", prep.transform(head(a_train)))],
        "val": [Segment("normal_val", prep.transform(normal_val)),
                Segment("attack_val", prep.transform(a_val))],
        "test": [Segment("attack_test", prep.transform(a_test))],
    }
    summary = {
        "sample_seconds": SAMPLE_SECONDS,
        "window_steps": WINDOW_STEPS,
        "dropped_startup_hours": DROP_STARTUP_HOURS,
        "cut_times": [str(c1), str(c2)],
        "mass_balance": prep.mass_balance.__dict__,
        "splits": {
            k: {
                "segments": [{"name": s.name, "start": str(s.df["Timestamp"].iloc[0]),
                              "end": str(s.df["Timestamp"].iloc[-1]), "rows": int(len(s.df)),
                              "attack_rows": int(s.df["label"].sum()),
                              "masked_rows": int(s.df["masked"].sum())} for s in v],
                "attack_ids": sorted({int(i) for s in v for i in s.df["attack_id"].unique() if i > 0}),
            } for k, v in segs.items()
        },
        "attacks": STAGE1_ATTACKS,
    }
    return segs, prep, summary


def windows(segments: list[Segment], stride: int = 1) -> dict[str, np.ndarray]:
    """Turn segments into model inputs.

    For each end step t: x = features[t-W+1 .. t], label = attack at t, plus
    next-step targets for the forecast head. Windows never cross segments.
    """
    xs, lab, msk, nxt, phys, nxt_normal, seg_idx, t_idx = [], [], [], [], [], [], [], []
    for si, s in enumerate(segments):
        F = s.df[FEATURES].to_numpy(np.float32)
        L = s.df["label"].to_numpy(np.float32)
        M = s.df["masked"].to_numpy(np.float32)
        Y = s.df[["FIT101_z", "LIT101_z"]].to_numpy(np.float32)
        P = s.df["phys_next_LIT_z"].to_numpy(np.float32)
        ends = np.arange(WINDOW_STEPS - 1, len(s.df) - 1, stride)
        if len(ends) == 0:
            continue
        idx = ends[:, None] - np.arange(WINDOW_STEPS - 1, -1, -1)[None, :]
        xs.append(F[idx])
        lab.append(L[ends]); msk.append(M[ends])
        nxt.append(Y[ends + 1]); phys.append(P[ends])
        nxt_normal.append(((L[ends + 1] == 0) & (M[ends + 1] == 0) & (M[ends] == 0)).astype(np.float32))
        seg_idx.append(np.full(len(ends), si)); t_idx.append(ends)
    return dict(x=np.concatenate(xs), label=np.concatenate(lab), masked=np.concatenate(msk),
                y_next=np.concatenate(nxt), phys_next=np.concatenate(phys),
                forecast_ok=np.concatenate(nxt_normal),
                seg=np.concatenate(seg_idx), t=np.concatenate(t_idx))