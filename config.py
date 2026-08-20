"""Locked scope + success metrics for Project 1 (MidSize Co content moderation).

These are the LOCKED targets from the scoping doc. The eval measures against them.
Targets are goals to validate, not promises — baseline first, then tune thresholds.
"""

# The 6 Jigsaw toxicity categories (multi-label: a comment can be several at once)
CATEGORIES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# High-severity categories carry legal/safety urgency — missing one is a disaster.
HIGH_SEVERITY = ["threat", "severe_toxic", "identity_hate"]
STANDARD = ["toxic", "obscene", "insult"]

# LOCKED recall targets (primary metric — we optimize to catch bad content).
RECALL_TARGETS = {
    "threat": 0.95,
    "severe_toxic": 0.95,
    "identity_hate": 0.95,
    "toxic": 0.85,
    "obscene": 0.85,
    "insult": 0.85,
}

# LOCKED precision guardrail (keep the human review queue workable).
PRECISION_FLOOR = 0.50  # applies to all categories

# Per-category confidence threshold: the knob that trades recall for precision.
# Start neutral; tune AFTER measuring baseline on the golden set.
DEFAULT_THRESHOLDS = {c: 0.5 for c in CATEGORIES}

# Stratified golden-set target counts (positives to pull per category).
# Rationale: rare high-severity categories need enough positives to measure recall.
# "threat" only has ~478 in the whole dataset — take (almost) all of them.
GOLDEN_SET_TARGETS = {
    "threat": 478,          # take all — too rare to sample
    "identity_hate": 700,
    "severe_toxic": 700,
    "obscene": 500,
    "insult": 500,
    "toxic": 500,
}
# Plus clean (all-zero) comments to measure false positives / precision:
GOLDEN_SET_CLEAN = 1000
