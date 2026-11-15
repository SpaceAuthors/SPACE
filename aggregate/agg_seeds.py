#!/usr/bin/env python3
import argparse, glob, json, os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--variant", default="full")
ap.add_argument("--out", required=True)
a = ap.parse_args()

dirs = sorted({d for pat in a.dirs for d in glob.glob(pat) if os.path.isdir(d)})
if not dirs:
    raise SystemExit(f"No directories found: {a.dirs}")

rows, seen = [], set()
for d in dirs:
    p = os.path.join(d, "retrain_results.json")
    if not os.path.exists(p):
        print(f"  Skipped (no retrain_results.json): {d}"); continue
    for r in json.load(open(p)).get(a.variant, []):
        k = (int(r["seed"]), int(r["fold"]))
        if k in seen:
            raise SystemExit(f"Duplicate (seed, fold) {k}: {d}")
        seen.add(k); r["_dir"] = d; rows.append(r)

best = {}
for r in rows:
    if r["val_best"] is None: continue
    s = int(r["seed"])
    if s not in best or r["val_best"] > best[s]["val_best"]: best[s] = r
if not best:
    raise SystemExit("No usable runs (val_best is empty in every run)")


def per_class(r):
    z = np.load(os.path.join(r["_dir"], f"proba_{a.variant}_f{r['fold']}_s{r['seed']}.npz"))
    y = z["y_true"]; p = z["proba"].argmax(1); t = y > 0
    return precision_recall_fscore_support(y[t], p[t], labels=[1, 2], zero_division=0)[:3]


per_seed = []
for s in sorted(best):
    r = best[s]; P, R, F = per_class(r)
    per_seed.append(dict(seed=s, fold=int(r["fold"]), val_best=r["val_best"], toxic_f1=r["toxic_only"],
                         P=float(P.mean()), R=float(R.mean()),
                         p1=float(P[0]), r1=float(R[0]), f1=float(F[0]),
                         p2=float(P[1]), r2=float(R[1]), f2=float(F[1]),
                         false_alarm=r["false_alarm"], n_benign=r["n_benign"],
                         pareto_C=r.get("pareto_C"), n_active=r.get("n_active")))

KEYS = ["toxic_f1", "P", "R", "p1", "r1", "f1", "p2", "r2", "f2", "false_alarm"]
v = lambda k: np.array([x[k] for x in per_seed], float)
ms = lambda k: dict(mean=float(v(k).mean()), sd=(float(v(k).std(ddof=1)) if len(per_seed) > 1 else None))
summary = {k: ms(k) for k in KEYS}
C = [x["pareto_C"] for x in per_seed if x["pareto_C"] is not None]
if C:
    summary["pareto_C"] = dict(median=float(np.median(C)), min=int(min(C)), max=int(max(C)))

n = len(per_seed)
f = lambda k: f"{summary[k]['mean']:.4f}" + (f" ± {summary[k]['sd']:.4f}" if summary[k]["sd"] is not None else "")
print(f"{len(dirs)} directories, {len(rows)} runs, {n} seeds (per seed, the fold with the highest val_best)")
print(f"{'seed':>6} {'fold':>4} {'val_best':>9} {'toxic F1':>9} {'P':>7} {'R':>7} {'FA':>5} {'|S|':>7}")
for x in per_seed:
    print(f"{x['seed']:>6} {x['fold']:>4} {x['val_best']:>9.4f} {x['toxic_f1']:>9.4f} {x['P']:>7.4f} {x['R']:>7.4f} "
          f"{x['false_alarm']:>5} {x['pareto_C'] if x['pareto_C'] is not None else '-':>7}")
print(f"\nMacro F1 {f('toxic_f1')}   Precision {f('P')}   Recall {f('R')}   (n = {n})")
print(f"ED   P {f('p1')}  R {f('r1')}  F1 {f('f1')}")
print(f"BDA  P {f('p2')}  R {f('r2')}  F1 {f('f2')}")
if C:
    print(f"|S| median {summary['pareto_C']['median']}, range {summary['pareto_C']['min']}-{summary['pareto_C']['max']}")

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(dirs=dirs, variant=a.variant, n_runs=len(rows), n_seeds=n, summary=summary, per_seed=per_seed),
          open(a.out, "w"), indent=1, default=float)
print(f"\nWrote {a.out}")
