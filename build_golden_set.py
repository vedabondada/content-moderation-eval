"""Build a STRATIFIED golden set from the raw Jigsaw train.csv.

Why stratified and not random:
  Jigsaw is wildly imbalanced (threat ~0.3% of rows). A random 2k-row sample
  yields ~6 threats — you cannot measure 95% recall on 6 examples. We instead
  deliberately over-pull rare high-severity categories so every category has
  enough positives to measure recall, plus a block of clean comments so we can
  measure precision (false-positive rate).

Input : data/train.csv  (Kaggle "Jigsaw Toxic Comment Classification" train.csv)
Output: data/golden_set.csv  (columns: id, comment_text, + 6 binary label cols)

Usage : python -m eval.build_golden_set
"""
import os
import pandas as pd
from eval.config import CATEGORIES, GOLDEN_SET_TARGETS, GOLDEN_SET_CLEAN

RAW = os.path.join("data", "train.csv")
OUT = os.path.join("data", "golden_set.csv")
SEED = 42  # fixed seed => reproducible golden set (an eval must be reproducible)


def build():
    df = pd.read_csv(RAW)
    missing = {"id", "comment_text", *CATEGORIES} - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing expected columns: {sorted(missing)}")

    picked_ids = set()
    parts = []

    # 1) Pull positives per category, rarest first (so scarce rows aren't
    #    consumed by a more common category's quota via multi-label overlap).
    order = sorted(CATEGORIES, key=lambda c: int(df[c].sum()))
    for cat in order:
        pool = df[(df[cat] == 1) & (~df["id"].isin(picked_ids))]
        n = min(GOLDEN_SET_TARGETS[cat], len(pool))
        take = pool.sample(n=n, random_state=SEED)
        picked_ids.update(take["id"])
        parts.append(take)
        print(f"{cat:>14}: pool={len(pool):>6}  took={n:>4}")

    # 2) Add clean (all-zero) comments to measure precision / false positives.
    clean_pool = df[(df[CATEGORIES].sum(axis=1) == 0) & (~df["id"].isin(picked_ids))]
    n_clean = min(GOLDEN_SET_CLEAN, len(clean_pool))
    clean = clean_pool.sample(n=n_clean, random_state=SEED)
    parts.append(clean)
    print(f"{'clean':>14}: pool={len(clean_pool):>6}  took={n_clean:>4}")

    golden = pd.concat(parts).drop_duplicates(subset="id").reset_index(drop=True)
    golden = golden[["id", "comment_text", *CATEGORIES]]
    golden.to_csv(OUT, index=False)

    print(f"\nGolden set: {len(golden)} comments -> {OUT}")
    print("Positives per category in the golden set:")
    for c in CATEGORIES:
        print(f"  {c:>14}: {int(golden[c].sum())}")


if __name__ == "__main__":
    build()
