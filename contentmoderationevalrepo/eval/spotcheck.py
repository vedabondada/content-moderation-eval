"""Spot check: does a STRONGER model close the severe_toxic gap?

Design (kept small + fair so it's cheap and defensible):
  - Balanced 60-comment sample: 30 genuine severe_toxic, 30 not-severe
    (mix of clean + toxic-but-not-severe, the exact confusion case).
  - Same comments, same prompt, two models. Only the MODEL varies.
  - Reports severe_toxic recall + precision at best threshold for each model.

This targets ONLY severe_toxic — the one category we recommend escalating.
Cost ~10-15 cents total. Requires ANTHROPIC_API_KEY with credits.

Usage: python -m eval.spotcheck
"""
import os
import json
import hashlib
import pandas as pd
from anthropic import Anthropic
from eval.config import CATEGORIES, PRECISION_FLOOR

CHEAP = "claude-haiku-4-5-20251001"
STRONG = os.environ.get("STRONG_MODEL", "claude-sonnet-4-5-20250929")
SAMPLE = os.path.join("data", "spotcheck_sample.csv")
CACHE = os.path.join("data", "cache_spotcheck")
os.makedirs(CACHE, exist_ok=True)
SEED = 7

# v3 prompt: keeps the threat broadening, gives severe_toxic a BALANCED definition
# (not the over-suppressed v2). Same prompt for both models -> fair test.
SYSTEM = (
    "You are a content-moderation classifier for an online community platform. "
    "For each category, output a calibrated confidence 0.0-1.0 that the comment "
    "belongs to it:\n  toxic, severe_toxic, obscene, threat, insult, identity_hate.\n"
    "Definitions:\n"
    "- toxic = rude, disrespectful, or unreasonable.\n"
    "- severe_toxic = a subset of toxic that is clearly EXTREME: explicit hatred, "
    "aggression, slurs, or dehumanization, distinctly worse than ordinary rudeness. "
    "Judge severity honestly on its merits.\n"
    "- obscene = profane or sexual.\n"
    "- threat = intent to harm a person, OR wishing serious harm/illness/death on "
    "them, OR encouraging violence against them.\n"
    "- insult = demeaning a person.\n"
    "- identity_hate = hate or slurs targeting a protected identity.\n"
    "A comment may belong to several categories or none. Respond with ONLY a JSON "
    "object mapping each category to its confidence."
)


def build_sample():
    if os.path.exists(SAMPLE):
        return pd.read_csv(SAMPLE)
    df = pd.read_csv(os.path.join("data", "golden_set.csv"))
    pos = df[df["severe_toxic"] == 1].sample(n=30, random_state=SEED)
    # negatives: the hard confusion case = toxic but NOT severe, plus some clean
    tox_not_sev = df[(df["toxic"] == 1) & (df["severe_toxic"] == 0)].sample(n=20, random_state=SEED)
    clean = df[df[CATEGORIES].sum(axis=1) == 0].sample(n=10, random_state=SEED)
    s = pd.concat([pos, tox_not_sev, clean]).drop_duplicates("id").reset_index(drop=True)
    s.to_csv(SAMPLE, index=False)
    return s


def classify(client, model, cid, text):
    key = hashlib.md5(f"{model}:v3:{cid}".encode()).hexdigest()
    cp = os.path.join(CACHE, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    m = client.messages.create(model=model, max_tokens=200, system=SYSTEM,
                               messages=[{"role": "user", "content": text[:8000]}])
    raw = m.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    d = json.loads(raw)
    out = {c: float(d.get(c, 0.0)) for c in CATEGORIES}
    json.dump(out, open(cp, "w"))
    return out


def score(scores, truth):
    best = None
    for i in range(1, 100):
        thr = i / 100
        tp = fp = fn = 0
        for s, t in zip(scores, truth):
            p = 1 if s >= thr else 0
            if t == 1 and p == 1: tp += 1
            elif t == 0 and p == 1: fp += 1
            elif t == 1 and p == 0: fn += 1
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        if prec >= PRECISION_FLOOR and (best is None or rec > best[1]):
            best = (thr, rec, prec)
    return best or (0.99, 0.0, 0.0)


def run():
    s = build_sample()
    client = Anthropic()
    truth = s["severe_toxic"].tolist()
    print(f"Spot check on {len(s)} comments ({sum(truth)} severe_toxic, {len(s)-sum(truth)} not).")
    print(f"{'model':>34}  {'sev_toxic recall':>16} {'precision':>10}")
    print("-" * 66)
    for label, model in [("cheap  " + CHEAP, CHEAP), ("STRONG " + STRONG, STRONG)]:
        sc = [classify(client, model, r["id"], r["comment_text"])["severe_toxic"] for _, r in s.iterrows()]
        thr, rec, prec = score(sc, truth)
        print(f"{label:>34}  {rec:>16.2f} {prec:>10.2f}")
    print("\nSame comments, same prompt, only the model differs -> difference = model effect.")


if __name__ == "__main__":
    run()
