"""Print precision, recall, F1, accuracy and per-attack results from results/results.json.

    python show_results.py                 # default results/results.json
    python show_results.py path/to/results.json
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("results") / "results.json")
R = json.loads(path.read_text())
if R.get("smoke_test"):
    print("SMOKE TEST: these numbers are meaningless. Run `python run_experiments.py` for real results.\n")

test_ids = R["data"]["splits"]["test"]["attack_ids"]
kinds = {a["id"]: a["kind"] for a in R["data"]["attacks"]}
f = lambda v, d=3: "  -  " if v is None else f"{v:.{d}f}"

names = ", ".join("#%d (%s)" % (i, kinds.get(i, "?")) for i in test_ids)
print(f"Held-out test attacks: {names}\n")
print(f"{'Detector':48s} {'data':>5s} {'Precision':>9s} {'Recall':>7s} {'F1':>6s} {'Accuracy':>8s} {'Caught':>7s} {'FA/day':>7s}")
for e in R["experiments"]:
    m = e["mean"]
    print(f"{e['label'][:48]:48s} {int(e['fraction']*100):4d}% {f(m['pointwise_precision']):>9s} {f(m['pointwise_recall']):>7s} "
          f"{f(m['pointwise_f1']):>6s} {f(m.get('pointwise_accuracy')):>8s} "
          f"{m['attacks_detected']:.1f}/{int(m['attacks_total']):<3d} {f(m['false_alarms_per_day'], 1):>7s}")

print("\nPer attack (seed of the first run; delay = seconds from attack start to first alarm):")
for e in R["experiments"]:
    per = {p["attack_id"]: p for p in e["runs"][0]["metrics"].get("per_attack", [])}
    cells = [f"#{i} " + (f"{per[i]['delay_s']:.0f}s" if per.get(i, {}).get("detected") else "missed") for i in test_ids]
    print(f"  {e['label'][:48]:48s} {int(e['fraction']*100):3d}%  " + "   ".join(cells))

print("\nPrecision: of the moments flagged, how many were really attacks.  Recall: of the attack moments, how many were flagged.")
print("Accuracy is mostly normal time, so it looks high even for weak detectors; quote precision/recall/F1 and attacks caught.")
