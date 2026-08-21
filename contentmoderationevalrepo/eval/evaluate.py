"""Run the classifier over the golden set and score it against LOCKED targets.

Reports, per category:
  - precision, recall, F1 at the current threshold
  - PASS/FAIL vs the locked recall target and precision floor
  - confusion counts (TP/FP/FN/TN) so failures are inspectable

Then it sweeps thresholds to show the recall/precision tradeoff and prints the
lowest threshold that still meets the precision floor for each category — the
starting point for tuning toward the recall targets.

Usage: python -m eval.evaluate            (uses DEFAULT_THRESHOLDS)
       python -m eval.evaluate --sweep    (also print threshold sweep)
"""
import sys
import os
import pandas as pd
from eval.config import (
    CATEGORIES, HIGH_SEVERITY, RECALL_TARGETS, PRECISION_FLOOR, DEFAULT_THRESHOLDS,
)
from eval.classify import classify

GOLDEN = os.path.join("data", "golden_set.csv")


def _counts(y_true, y_score, thr):
    tp = fp = fn = tn = 0
    for t, s in zip(y_true, y_score):
        pred = 1 if s >= thr else 0
        if t == 1 and pred == 1: tp += 1
        elif t == 0 and pred == 1: fp += 1
        elif t == 1 and pred == 0: fn += 1
        else: tn += 1
    return tp, fp, fn, tn


def _pr(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def run(sweep=False):
    df = pd.read_csv(GOLDEN)
    print(f"Scoring {len(df)} comments with the classifier (cached after first run)...\n")

    # score every comment once
    scores = {c: [] for c in CATEGORIES}
    for _, row in df.iterrows():
        out = classify(row["id"], row["comment_text"])
        for c in CATEGORIES:
            scores[c].append(out[c])

    print(f"{'category':>14} {'thr':>5} {'prec':>6} {'rec':>6} {'F1':>6}  target  result")
    print("-" * 60)
    all_pass = True
    for c in CATEGORIES:
        thr = DEFAULT_THRESHOLDS[c]
        tp, fp, fn, tn = _counts(df[c].tolist(), scores[c], thr)
        prec, rec, f1 = _pr(tp, fp, fn)
        rec_ok = rec >= RECALL_TARGETS[c]
        prec_ok = prec >= PRECISION_FLOOR
        ok = rec_ok and prec_ok
        all_pass &= ok
        tag = "HIGH-SEV" if c in HIGH_SEVERITY else ""
        print(f"{c:>14} {thr:>5.2f} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f}  "
              f"r>={RECALL_TARGETS[c]:.2f}  {'PASS' if ok else 'FAIL'} {tag}")
        if not ok:
            why = []
            if not rec_ok: why.append(f"recall {rec:.2f} < {RECALL_TARGETS[c]:.2f} (missed {fn} bad comments)")
            if not prec_ok: why.append(f"precision {prec:.2f} < {PRECISION_FLOOR:.2f} ({fp} false flags)")
            print(f"{'':>14}   -> {'; '.join(why)}")

    print("-" * 60)
    print("OVERALL:", "PASS" if all_pass else "FAIL (tune thresholds / prompt, then re-run)")

    if sweep:
        print("\nThreshold sweep (lowest thr meeting precision floor):")
        for c in CATEGORIES:
            best = None
            for thr in [i / 20 for i in range(1, 20)]:
                tp, fp, fn, tn = _counts(df[c].tolist(), scores[c], thr)
                prec, rec, f1 = _pr(tp, fp, fn)
                if prec >= PRECISION_FLOOR and (best is None or rec > best[1]):
                    best = (thr, rec, prec)
            if best:
                print(f"  {c:>14}: thr={best[0]:.2f} -> recall {best[1]:.2f}, precision {best[2]:.2f}")
            else:
                print(f"  {c:>14}: no threshold meets precision floor {PRECISION_FLOOR}")


if __name__ == "__main__":
    run(sweep="--sweep" in sys.argv)
