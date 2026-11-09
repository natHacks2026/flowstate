"""Step 0: convert the SWaT Excel files to small, fast CSVs (run once).

Reading the .xlsx files takes a few minutes each, so we do it once and keep only
what the pipeline needs: the timestamp, the five stage-1 columns and the label.

Usage:
    python convert_swat.py --swat-dir "path/to/Physical" --out data
"""
import argparse
import time
from pathlib import Path

import openpyxl
import pandas as pd

STAGE1 = ["FIT101", "LIT101", "MV101", "P101", "P102"]
FILES = {
    "normal": "SWaT_Dataset_Normal_v1.xlsx",  # v1 = v0 minus the 30-minute tank drain
    "attack": "SWaT_Dataset_Attack_v0.xlsx",
}


def read_swat_xlsx(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    header, idx, rows = None, None, []
    for row in ws.iter_rows(values_only=True):
        if header is None:
            # The real header row starts with " Timestamp" (row 1 or 2 depending on file)
            if row and isinstance(row[0], str) and row[0].strip() == "Timestamp":
                header = [str(c).strip() if c is not None else "" for c in row]
                want = ["Timestamp"] + STAGE1 + ["Normal/Attack"]
                idx = [header.index(c) for c in want]
            continue
        if row[idx[0]] is None:
            continue
        rows.append([row[i] for i in idx])
    wb.close()
    df = pd.DataFrame(rows, columns=["Timestamp"] + STAGE1 + ["label_raw"])
    # Timestamps look like " 28/12/2015 10:00:00 AM" (day first)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(str).str.strip(),
                                     format="%d/%m/%Y %I:%M:%S %p")
    # Labels include the typo "A ttack"
    df["label"] = df["label_raw"].astype(str).str.replace(" ", "").eq("Attack").astype("int8")
    df = df.drop(columns="label_raw")
    for c in STAGE1:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swat-dir", required=True, help="folder with the SWaT .xlsx files")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fname in FILES.items():
        t0 = time.time()
        df = read_swat_xlsx(Path(a.swat_dir) / fname)
        df.to_csv(out / f"swat_{name}_stage1.csv.gz", index=False)
        print(f"{name}: {len(df):,} rows, {df['label'].sum():,} attack rows, "
              f"{df['Timestamp'].min()} to {df['Timestamp'].max()} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
