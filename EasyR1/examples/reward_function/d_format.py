"""Canonical trace format for idea D — single source of truth.

Imported by D1 (trace assembly), D2 (filter), D3 (audit), D5 (RL reward) and the
P2 probe. Never copy these regexes elsewhere.

Tags are PLAIN BPE TEXT by contract — never registered as tokenizer special
tokens (EasyR1's reward manager and mm-eval decode with
skip_special_tokens=True, which silently strips special=True tokens).
"""

import re

CHECK_OPEN, CHECK_CLOSE = "<check>", "</check>"
VERD_OPEN, VERD_CLOSE = "<verdict>", "</verdict>"
ANS_OPEN, ANS_CLOSE = "<answer>", "</answer>"

# Multi-token tags DRIFT ("< check >", "</ verdict>") — EasyR1 carries the same
# workaround for <think> (examples/reward_function/math.py:40); Search-R1 needs
# six whitespace variants of "</search>" as stop strings. Normalise BEFORE
# matching.
TAG_WS_RE = re.compile(r"\s*(<|>|/)\s*")
# 2026-08-13 (arm 10, CoVe ordering): the gap between </check> and <verdict>
# may hold an evidence sentence — any text NOT containing another tag. The
# check-text group carries the same tag guard so lazy backtracking can never
# bridge a pair across a stray tag (a latent quirk of the old `(.*?)` form,
# previously masked by structural_ok's tag-count balance). v1 traces (zero
# gap, tag-free check text) parse byte-identically; capture groups unchanged.
_NO_TAG = r"(?:(?!</?(?:check|verdict|answer)>).)"
PAIR_RE = re.compile(
    rf"<check>({_NO_TAG}*?)</check>\s*{_NO_TAG}*?<verdict>\s*(yes|no)\s*</verdict>",
    re.S | re.I,
)
ANS_RE = re.compile(r"<answer>\s*([A-E])\s*</answer>", re.I)
# 2026-08-14 (relpos rows): their answers are short option PHRASES ("left of",
# "such a sign is not found"), not letters. Phrase parsing is a row-aware
# OPT-IN (allow_phrase) so the MCQ contract stays strict — "B. blue" in the
# answer slot still fails on MCQ rows. Phrases normalize to lowercase; relpos
# extra_info.denial_letter carries the lowercase denial phrase, so the
# letter == denial_letter equality used by monitors works verbatim.
ANS_PHRASE_RE = re.compile(r"<answer>[ \t]*([^<>\n]{1,60}?)[ \t]*</answer>")

# Loose variants used only for DIAGNOSTICS (how often did the model open a tag
# but break the strict grammar?) — never for reward or filtering.
CHECK_OPEN_LOOSE_RE = re.compile(r"<\s*check\s*>", re.I)
VERD_ANY_RE = re.compile(r"<\s*verdict\s*>(.*?)<\s*/\s*verdict\s*>", re.S | re.I)
ANS_ANY_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.S | re.I)

BANNED_SUBSTRINGS = ("<think>", "<box>", "<tool_call>", "<sq>", "Sub-questions:", "Sub-answer")

MAX_CHECKS = 8


def normalize(response: str) -> str:
    return TAG_WS_RE.sub(r"\1", response)


def parse_trace(response: str, allow_phrase: bool = False):
    """Return (pairs, letter, normalized_text).

    pairs: list of (question, verdict) with verdict in {"yes", "no"}.
    letter: "A".."E", or (allow_phrase only) a lowercase short phrase, or None.
    """
    s = normalize(response)
    pairs = [(q.strip(), v.lower()) for q, v in PAIR_RE.findall(s)]
    m = ANS_RE.search(s)
    letter = m.group(1).upper() if m else None
    if letter is None and allow_phrase:
        m = ANS_PHRASE_RE.search(s)
        letter = m.group(1).strip().lower() if m else None
    return pairs, letter, s


def structural_ok(response: str, allow_phrase: bool = False) -> tuple[bool, list[str]]:
    """Constraints 1-5 of the contract. Returns (ok, list_of_violations)."""
    s = normalize(response)
    pairs, letter, _ = parse_trace(response, allow_phrase)
    problems = []
    if not (1 <= len(pairs) <= MAX_CHECKS):
        problems.append(f"n_checks={len(pairs)} outside [1,{MAX_CHECKS}]")
    counts = {t: s.count(t) for t in (CHECK_OPEN, CHECK_CLOSE, VERD_OPEN, VERD_CLOSE)}
    if len(set(counts.values())) != 1 or counts[CHECK_OPEN] != len(pairs):
        problems.append(f"unbalanced tags {counts} vs pairs={len(pairs)}")
    if s.count(ANS_OPEN) != 1 or s.count(ANS_CLOSE) != 1:
        problems.append(f"answer tag count {s.count(ANS_OPEN)}/{s.count(ANS_CLOSE)}")
    if letter is None:
        problems.append("no parseable answer" if allow_phrase else "no bare A-E letter in <answer>")
    tail = s.rsplit(ANS_CLOSE, 1)[-1] if ANS_CLOSE in s else ""
    if tail.strip():
        problems.append("text after </answer>")
    for b in BANNED_SUBSTRINGS:
        if b in s:
            problems.append(f"banned substring {b}")
    return (not problems), problems


def consistency_ok(pairs, letter, denial_letter: str) -> bool:
    """Constraint 6: any 'no' verdict <=> answer is the denial letter."""
    if not pairs or letter is None or not denial_letter:
        return False
    any_no = any(v == "no" for _, v in pairs)
    return (letter == denial_letter) == any_no


def consistency_deny_requires_no(pairs, letter, denial_letter: str) -> bool:
    """Direction A of constraint 6 only: answering the denial letter requires
    some 'no' verdict. A 'no' verdict does NOT force the denial answer — a
    trace may conclude a premise fails and still commit to a substantive
    option (a hedge)."""
    if not pairs or letter is None or not denial_letter:
        return False
    return letter != denial_letter or any(v == "no" for _, v in pairs)
