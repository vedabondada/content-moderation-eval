"""Classifier: Claude scores a comment 0..1 on each of the 6 toxicity categories.

Design notes (these are the FDPM-relevant choices):
  - Output is PER-CATEGORY confidence, not a single flag. Threats route to legal
    urgency; obscene routes to community standards. Per-category preserves that.
  - We ask for calibrated 0..1 scores so the eval can sweep thresholds later
    (the recall/precision knob) instead of baking one threshold into the model.
  - Results are cached by comment id so re-running the eval is free and
    deterministic-ish (a real eval must be cheap to re-run).

Requires: ANTHROPIC_API_KEY in the environment.  pip install anthropic
"""
import json
import os
import hashlib
from eval.config import CATEGORIES

MODEL = os.environ.get("MOD_MODEL", "claude-sonnet-4-5")  # cheap/fast for classification
CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_client = None


def _client_lazy():
    global _client
    if _client is None:
        from anthropic import Anthropic  # imported lazily so eval math is testable without the SDK
        _client = Anthropic()
    return _client


# PROMPT_VERSION: v3 (2026-08-20) — keeps v2's broadened threat; gives severe_toxic a
# BALANCED definition (v2's "prefer toxic" over-suppressed it). This is the version the
# spot check validated on a balanced sample.
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


def _cache_path(comment_id, model):
    # include a hash of the system prompt so changing the prompt busts the cache
    # (v1 and v2 results are stored side by side, never conflated).
    pver = hashlib.md5(SYSTEM.encode()).hexdigest()[:8]
    key = hashlib.md5(f"{model}:{pver}:{comment_id}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


def classify(comment_id, text, model=MODEL):
    cp = _cache_path(comment_id, model)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)

    msg = _client_lazy().messages.create(
        model=model,
        max_tokens=200,
        system=SYSTEM,
        messages=[{"role": "user", "content": text[:8000]}],
    )
    raw = msg.content[0].text.strip()
    # tolerate ```json fences
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    scores = json.loads(raw)
    scores = {c: float(scores.get(c, 0.0)) for c in CATEGORIES}

    with open(cp, "w") as f:
        json.dump(scores, f)
    return scores
