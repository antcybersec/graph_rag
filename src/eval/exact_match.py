"""Deterministic answer scoring against the gold answers -- no LLM, no quota.

Gold answers in eval_public.jsonl are short exact strings: a count, an event
title, or a medallist's name as written in the infobox. A row is correct when:

  * the pipeline reported a `short_answer` (event_graph does): it is a single
    value (no " | "-separated alternatives -- listing several matching events is
    an ambiguous answer, and free-text answers get no such hedge credit) and
    its normalized value equals a normalized gold answer;
  * otherwise (free-text answers from rag/graphrag/agentic): for numeric gold,
    the first number the answer states -- ignoring citations, years and
    numbers copied from the question -- equals it; for text gold, the
    normalized gold appears inside the normalized answer.

The free-text rule is a heuristic. `validate_against_judge()` reports how
often it agrees with the LLM judge on the existing benchmark rows.
"""
import re

from src.common.infobox import norm_key

_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_NUMBER_WORDS.update({w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())})


def _words_to_digits(text: str) -> str:
    def compound(m):
        tens, unit = m.group(1).lower(), (m.group(2) or "").lower()
        return str(_NUMBER_WORDS[tens] + (_NUMBER_WORDS.get(unit, 0) if unit else 0))
    text = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\b",
                  compound, text, flags=re.I)
    return re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda m: str(_NUMBER_WORDS[m.group(1).lower()]), text, flags=re.I)


# "24 nations", "2 biathlon events" -- a number attached to the thing being counted.
_COUNT_RE = re.compile(r"\b(\d+)\s+(?:[^\W\d]+\s+){0,4}?(?:nations|countries|events)\b", re.I)
_NO_EVENTS_RE = re.compile(r"\bno\b(?=(?:\s+[^\W\d]+){0,4}\s+events\b)", re.I)
# Superlative gold is a full title; free-text answers usually name just the event.
_TITLE_GOLD_RE = re.compile(r"^.+ at the \d{4} (?:Summer|Winter) Olympics – (.+)$")


def _stated_number(answer: str, question: str):
    text = re.sub(r"[\[【][^\]】]*[\]】]", " ", answer)  # citations like [Q123_c0] carry digits
    text = _NO_EVENTS_RE.sub("0", _words_to_digits(text))
    # "rifle three positions" in the question must not count as a stated 3.
    from_question = set(re.findall(r"\d+", _words_to_digits(question)))
    usable = lambda n: len(n) != 4 and n not in from_question
    counted = [m.group(1) for m in _COUNT_RE.finditer(text) if usable(m.group(1))]
    if counted:
        return counted[0]
    return next((n for n in re.findall(r"\d+", text) if usable(n)), None)


def _squash(s: str) -> str:
    """Letters/digits only, "and" dropped -- infobox golds concatenate team members
    ("Dani KingLaura TrottJoanna Rowsell") where answers write "Dani King, Laura Trott, and ..."."""
    return re.sub(r"\s+", "", re.sub(r"\band\b", " ", norm_key(s)))


def is_correct(answer: str, gold: list, question: str = "", short_answer: str = None) -> bool:
    gold = [g for g in gold if g]
    if not gold:
        return False
    gold_keys = {norm_key(g) for g in gold}
    if short_answer is not None:
        parts = short_answer.split(" | ")
        if len(parts) != 1:
            return False  # listing several candidates is an ambiguous answer, not a correct one
        candidate = parts[0]
        if norm_key(candidate) in gold_keys:
            return True
        # Same formatting tolerances the free-text path already allows, so a
        # pipeline is not penalised for punctuation: infobox golds concatenate
        # team members ("Dani KingLaura TrottJoanna Rowsell") where an answer
        # writes a list, and a title gold may be given as its event part alone.
        squashed = _squash(candidate)
        for g in gold:
            if _squash(g) == squashed:
                return True
            title = _TITLE_GOLD_RE.match(g)
            if title and norm_key(candidate) == norm_key(title.group(1)):
                return True
        return False
    if all(re.fullmatch(r"\d+", g) for g in gold_keys):
        return _stated_number(answer or "", question) in gold_keys
    answer_tokens = f" {norm_key(answer)} "
    answer_squashed = _squash(answer or "")
    for g in gold:
        title = _TITLE_GOLD_RE.match(g)
        if title:
            # Whole tokens, not squashed: "women's marathon" must not match "men's marathon".
            if f" {norm_key(title.group(1))} " in answer_tokens:
                return True
        elif _squash(g) in answer_squashed:
            return True
    return False


def validate_against_judge(results_path: str = "data/results/benchmark_results.jsonl"):
    """Agreement between is_correct() and judge accuracy >= 4 on existing rows."""
    import json
    rows = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and (r.get("judge") or {}).get("accuracy") is not None:
                rows[(r["qid"], r["pipeline"])] = r
    agree, disagreements = 0, []
    for r in rows.values():
        em = is_correct(r["answer"], r["reference_answer"].split("; "), r["question"], r.get("short_answer"))
        judged = r["judge"]["accuracy"] >= 4
        agree += em == judged
        if em != judged:
            disagreements.append((r["qid"], r["pipeline"], em, r["judge"]["accuracy"], r["reference_answer"], r["answer"][:160]))
    print(f"exact-match vs judge agreement: {agree}/{len(rows)}")
    return disagreements


if __name__ == "__main__":
    for d in validate_against_judge():
        print(d)
