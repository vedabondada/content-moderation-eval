"""Prefill the classifier cache over the whole golden set, in parallel.

Sequential classification of ~4.4k comments takes ~90 min. This runs many
concurrently (results are cached to disk by classify()), so a re-run or the
evaluate.py pass is basically free afterward. Rate-limit/overload errors are
retried with backoff.

Usage: ANTHROPIC_API_KEY=... MOD_MODEL=... python -m eval.run_baseline
"""
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from eval.classify import classify

GOLDEN = os.path.join("data", "golden_set.csv")
WORKERS = int(os.environ.get("MOD_WORKERS", "20"))
MAX_RETRY = 5


def _one(cid, text):
    for attempt in range(MAX_RETRY):
        try:
            classify(cid, text)
            return True
        except Exception as e:
            name = type(e).__name__
            transient = any(k in name for k in ("RateLimit", "Overloaded", "APIStatus", "Connection", "Timeout"))
            if transient and attempt < MAX_RETRY - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  FAIL {cid}: {name} {str(e)[:100]}")
            return False
    return False


def run():
    df = pd.read_csv(GOLDEN)
    total = len(df)
    print(f"Classifying {total} comments with {WORKERS} workers...")
    t0 = time.time()
    done = ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_one, r["id"], r["comment_text"]): r["id"] for _, r in df.iterrows()}
        for f in as_completed(futs):
            done += 1
            ok += 1 if f.result() else 0
            if done % 250 == 0 or done == total:
                dt = time.time() - t0
                print(f"  {done}/{total}  ok={ok}  {dt:.0f}s  ({done/dt:.1f}/s)")
    print(f"Done: {ok}/{total} classified in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
