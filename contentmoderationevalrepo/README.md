# Content Moderation Classifier + Evaluation Harness

A per-category content-moderation classifier and the Python evaluation harness used to
measure and improve it. Claude scores each comment on six toxicity categories and flags
the likely-harmful ones for human review — it does not remove anything automatically. The
system runs on the lowest-cost model.

Built as Project 1 of a Forward-Deployed PM portfolio. Client framing: a community
platform (~2M monthly users) with a 48-hour moderation backlog and legal exposure from
slow response to threats and hate speech.

## What this repo is

The point of the project is not the classifier — it is the **evaluation harness**: the
part that measures whether the classifier is good enough to trust, and shows where it
fails. The harness covers stratified sampling, per-category precision/recall against
targets, threshold tuning, and a cheap-vs-stronger model comparison.

## Approach

- **Flag for review, not auto-remove.** A human clears false alarms downstream, so the
  system is tuned to catch harmful content rather than to minimize flags.
- **Metrics tie to the business goal.** Missing harmful content is worse than a false
  alarm, so the primary metric is recall (catch rate), with precision as a guardrail so
  reviewers are not overwhelmed. Targets: 95% recall on high-severity categories (threat,
  severe-toxic, identity-hate), 85% on the rest.
- **Stratified evaluation set.** Threats are ~0.3% of the Jigsaw data. A random sample
  would contain too few to measure, so the golden set takes all 478 threats, over-samples
  the other rare categories, and adds clean comments to measure false alarms
  (4,378 comments total).
- **Cheapest model first.** Establish cost on the lowest-cost model before considering
  anything more expensive; escalate only if measurement justifies it.

## Results (lowest-cost model, Claude Haiku)

| Category | Baseline | Final | Target | Result |
|---|---|---|---|---|
| toxic | 94% | 100% | 85% | met |
| obscene | 78% | 98% | 85% | met |
| insult | 95% | 100% | 85% | met |
| threat | 92% | 98% | 95% | met |
| identity-hate | 79% | 95% | 95% | met |
| severe-toxic | 79% | — | 95% | covered via toxic (see note) |

The path from baseline to final: threshold tuning (free) fixed obscene and identity-hate;
a prompt revision raised threat to target; a 60-comment cheap-vs-stronger model comparison
showed a stronger model was not better, so it was not used.

**Severe-toxic note.** Severe-toxic falls short as a standalone metric. But every
severe-toxic comment in the dataset is also labeled toxic (1,595 of 1,595), and the
classifier catches 100% of toxic comments — so every severe-toxic comment is already
flagged for review under the toxic label. The business requirement (no harmful comment
reaches users unreviewed) is met. The remaining limitation is that the severe-toxic signal
is too noisy to reliably rank comments by severity.

## Run it

```bash
pip install anthropic pandas scikit-learn
export ANTHROPIC_API_KEY=sk-...

# 1. put the Kaggle "Jigsaw Toxic Comment Classification" train.csv in data/
# 2. build the stratified evaluation set
python -m eval.build_golden_set
# 3. classify the full set (parallel, cached)
python -m eval.run_baseline
# 4. score against targets (add --sweep for the threshold sweep)
python -m eval.evaluate --sweep
# 5. per-category threshold tuning
python -m eval.tune
# 6. optional: cheap-vs-stronger model comparison on severe-toxic
python -m eval.spotcheck
```

## Files

- `eval/config.py` — categories, locked targets, stratified-sampling plan (single source of truth)
- `eval/build_golden_set.py` — stratified sampler → `data/golden_set.csv`
- `eval/classify.py` — Claude per-category classifier; on-disk cache versioned by prompt
- `eval/run_baseline.py` — parallel classification of the full set, with retry
- `eval/evaluate.py` — precision/recall vs targets, pass/fail, threshold sweep
- `eval/tune.py` — per-category threshold tuning
- `eval/spotcheck.py` — cheap-vs-stronger model comparison on a balanced sample
- `docs/case-study.html` — written case study of the project

## Notes

Data and cache files are gitignored; the Jigsaw dataset must be downloaded separately from
Kaggle. Total model spend to reach the results above was a few dollars.
