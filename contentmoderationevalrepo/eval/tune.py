"""Step 8-9: per-category threshold tuning (free — reuses cached scores).

Policy: for each category, pick the LOWEST flag-threshold whose precision still
meets the floor (>=0.50). Lower threshold = flag more aggressively = catch more
(higher recall). This is the "be less cautious" knob, applied per category, and
it costs no new API calls. Categories that can't reach their recall target even
at their best threshold are flagged for ESCALATION to a stronger model.

Usage: MOD_MODEL=... python -m eval.tune
"""
import os
import pandas as pd
from eval.config import CATEGORIES, HIGH_SEVERITY, RECALL_TARGETS, PRECISION_FLOOR
from eval.classify import classify
from eval.evaluate import _counts, _pr

GOLDEN = os.path.join("data", "golden_set.csv")


def run():
    df = pd.read_csv(GOLDEN)
    scores = {c: [] for c in CATEGORIES}
    for _, row in df.iterrows():
        out = classify(row["id"], row["comment_text"])
        for c in CATEGORIES:
            scores[c].append(out[c])

    print(f"{'category':>14} {'best thr':>8} {'prec':>6} {'rec':>6}  target  result")
    print("-" * 62)
    tuned = {}
    escalate = []
    for c in CATEGORIES:
        y = df[c].tolist()
        # lowest threshold meeting precision floor -> maximizes recall
        best = None
        for thr in [i / 100 for i in range(1, 100)]:
            tp, fp, fn, tn = _counts(y, scores[c], thr)
            prec, rec, f1 = _pr(tp, fp, fn)
            if prec >= PRECISION_FLOOR:
                if best is None or rec > best["rec"]:
                    best = {"thr": thr, "prec": prec, "rec": rec}
        if best is None:  # never meets precision floor
            best = {"thr": 0.99, "prec": 0.0, "rec": 0.0}
        tuned[c] = best["thr"]
        ok = best["rec"] >= RECALL_TARGETS[c] and best["prec"] >= PRECISION_FLOOR
        tag = "HIGH-SEV" if c in HIGH_SEVERITY else ""
        if not ok:
            escalate.append(c)
        print(f"{c:>14} {best['thr']:>8.2f} {best['prec']:>6.2f} {best['rec']:>6.2f}  "
              f"r>={RECALL_TARGETS[c]:.2f}  {'PASS' if ok else 'FAIL -> escalate'} {tag}")

    print("-" * 62)
    print("Tuned thresholds:", {c: round(tuned[c], 2) for c in CATEGORIES})
    if escalate:
        print("Still failing after tuning (need stronger model/prompt):", escalate)
    else:
        print("All categories pass after tuning.")


if __name__ == "__main__":
    run()
