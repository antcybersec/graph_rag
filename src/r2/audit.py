"""Audit an answer file with no model, no database and no quota.

Why this exists. The hidden set ships without gold answers, so there is no
score to read and nothing to tell us which of the fifty are wrong. Round 1 hit
this exactly: the hidden answers written two minutes before a fix carried the
bug, and five of them were factually wrong -- three answering "12", which was
the evidence cap rather than the count. Nothing in the file said so. An
independent check found it, and an independent check is the only thing that
can.

Round 2's questions are not templates, so a full oracle is not available the
way it was in Round 1. What *is* available is a consistency check between parts
of the record that were produced by different means:

  1. **Numbers.** A figure in the answer should come from the arithmetic the
     query computed. Where the record carries a `computed` block, every number
     in the answer text is held against it. A number that appears nowhere is
     not proof of error -- it may be a year, or a figure read from prose -- but
     it is the shortlist worth reading by hand, and it is the exact shape of
     Round 1's "12".

     Two things are reconciled before a figure is called unexplained. A number
     written with a unit after it ("Greco-Roman 60 kg") is part of a name. And
     a listing whose figures add up to a tally's `total`, in as many terms as
     the tally has `entries`, is that tally's ranking written out: the
     per-entry figures are not in the record, so they cannot be matched one by
     one, but they can be checked as a sum -- which is the stronger test of
     the two, because a listing that does not add up is a finding.

  2. **Ties.** If the arithmetic found a tie at the top and the answer names
     one holder without the word "tie" or the other holders, the answer is
     wrong rather than short. This is checkable and so it is checked.

  3. **Route.** An answer whose route found nothing, or which fell through to
     full-text search, is worth reading whatever its status says.

  4. **Honesty.** `not_in_corpus` with citations, or `answered` whose text
     reads like a refusal, are contradictions between the status field and the
     answer field.

  5. **Citations.** How many documents each answer leans on, so over-citation
     is visible: citing everything says nothing about which document carries
     the answer, and citations are 10% of the score.

  6. **Superlatives.** Where the plan asked for an extreme and the code ranked
     the candidates, the answer must name the holder the ranking put at the top
     or quote its figure. An answer that names neither took its superlative
     from somewhere other than the arithmetic, which is Round 1's single most
     expensive failure: 1 of 21 superlatives right before the ranking was
     computed in code, 21 of 21 after.

Nothing here decides anything. It prints a worklist, ordered so the answers
most likely to be wrong are at the top.

    python -m src.r2.audit data/results/r2/answers_hidden.jsonl
    python -m src.r2.audit data/results/r2/answers_public.jsonl --quiet
"""
import argparse
import json
import re
import sys
from collections import Counter

# Four or more digits with no separators is usually a year or an id, and years
# are everywhere in this corpus; they are matched anyway and filtered below by
# whether the record's own figures explain them.
# The minus sign is only a minus sign at the start of a token: in "top-25" and
# "Greco-Roman 60" the hyphen joins words, and reading it as a negative put a
# phantom "-25" on the worklist.
_NUMBER = re.compile(r"(?<![0-9A-Za-z])-?\d+(?:,\d{3})*(?:\.\d+)?")

_REFUSAL = ("not in the corpus", "does not say", "no source", "cannot be determined",
            "not stated", "does not state", "no document", "not established",
            "corpus does not")


def _numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text or "")]


def _as_float(s: str):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _names(holders: list) -> list[str]:
    """Holder keys as they would appear in prose.

    A graph key is `athletics | men s 100 metres`; the answer writes "Men's
    100 metres". The leading segment, truncated, is what survives both.
    Anything shorter than three characters is dropped: a holder called "1"
    matches every answer and would make the check useless.
    """
    out = []
    for h in holders or []:
        head = str(h).split(" | ")[0].strip()[:24]
        if len(head) >= 3:
            out.append(head.lower())
    return out


# A number written with a unit after it is part of a name, not arithmetic:
# "Men's Greco-Roman 60 kg", "women's 800 metres", "the 76 kg final".
_MEASURED = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*"
                       r"(?:kg|kilograms?|m\b|metres?|meters?|km|lb|lbs|ft)", re.I)


def _subset_sums_to(values: list[float], target: float, size: int) -> bool:
    """Can `size` of these numbers add up to `target`?

    This is what tells a correct per-edition listing apart from an invented
    figure. An `event_series` answer legitimately writes out every edition's
    count -- "14 in 2000, 16 in 2004, 16 in 2008 ..." -- and those per-entry
    figures live in the tally's `ranked` list, which the record leaves out as
    too long. So the audit cannot match them one by one. What it can do is
    check the listing against the two figures the record does carry: the
    tally's `total` and how many `entries` went into it. Eight numbers from
    the answer that add up to the computed total are the computed ranking,
    near enough -- and eight that do not are worth reading, which is the
    stronger statement of the two.

    Exact arithmetic on a small set: at most a few dozen numbers and a
    cardinality bound, so the reachable (count, sum) pairs are enumerated
    directly rather than approximated.
    """
    if size <= 0 or size > len(values):
        return False
    # Work in hundredths to keep the sums integral, since the figures can be
    # ratios as well as counts.
    scaled = [int(round(v * 100)) for v in values]
    goal = int(round(target * 100))
    reach = {(0, 0)}
    for v in scaled:
        reach |= {(c + 1, t + v) for c, t in reach if c < size and t + v <= goal}
    return (size, goal) in reach


def _asked_about(computed: list, plan: dict) -> list:
    """The tallies the question actually asked for.

    One question computes several tallies. An `event_series` question gets
    both competitors-per-edition and nations-per-edition; a medal question
    gets one tally per colour, because a colour question needs its own. Only
    one of them is usually what was asked, and holding the answer against the
    others produces findings that are not defects -- r2-pub-010 answered the
    nations question exactly right, row for row, and was flagged because the
    *competitors* tally happened to have a tie at the same value of 17.

    The plan already records what was asked: `metric` ("nations") and `medal`
    ("gold"). Where that names a tally, that is the tally to check. Where it
    names none, all of them stay in -- a check that silently stops running is
    worse than one that is occasionally noisy.
    """
    want = [str(plan.get(k) or "").lower() for k in ("medal", "metric")]
    want = [w for w in want if w]
    if not want:
        return computed
    picked = [t for t in computed
              if any(w in f"{t.get('unit', '')} {t.get('what', '')}".lower()
                     for w in want)]
    return picked or computed


def _computed_values(record: dict) -> set:
    """Every figure the query computed for this question, as floats."""
    out = set()
    for t in record.get("computed") or []:
        for key in ("highest", "lowest"):
            v = _as_float((t.get(key) or {}).get("value"))
            if v is not None:
                out.add(v)
        for key in ("total", "entries"):
            v = _as_float(t.get(key))
            if v is not None:
                out.add(v)
    return out


def audit_one(record: dict) -> list[str]:
    """Reasons this answer is worth reading by hand, worst first."""
    flags = []
    answer = str(record.get("answer") or "")
    status = record.get("status")
    computed = record.get("computed") or []
    steps = record.get("trace") or []

    # 2. a tie reported as a single holder
    #
    # Only for a tally the answer actually engages with. A question about gold
    # gets a tally for every colour, because a colour question needs its own
    # one; a tie in the bronze column is not a defect in an answer about gold,
    # and flagging it buried the four real flags under four irrelevant ones on
    # the first twenty answers this ran against.
    in_answer = {_as_float(x) for x in _numbers(answer)}
    asked = _asked_about(computed, record.get("plan") or {})
    for t in asked:
        box = t.get("highest") or {}
        holders = box.get("holders") or []
        if len(holders) <= 1:
            continue
        value = _as_float(box.get("value"))
        named = sum(1 for h in holders
                    if str(h).split(" | ")[0][:24].lower() in answer.lower())
        engaged = named > 0 or (value is not None and value in in_answer)
        if engaged and named < len(holders) and "tie" not in answer.lower():
            flags.append(f"TIE NOT REPORTED: {len(holders)} hold the top "
                         f"{box.get('value')} in {t.get('what')!r}; the answer "
                         f"names {named} and does not say they are level")

    # 1. a number in the answer that the computed figures do not explain
    if computed and status in ("answered", "partial"):
        known = _computed_values(record)
        # Event names are full of numbers -- "Men's 100 metres", "Women's
        # 10,000 metres" -- and flagging those buries the flag that matters.
        # Any number that appears inside a name the arithmetic itself carries
        # is accounted for.
        in_names = " ".join(h for t in computed for key in ("highest", "lowest")
                            for h in ((t.get(key) or {}).get("holders") or []))
        # Holders carry figures of their own in the text ("USA=14"), and years
        # are not arithmetic; neither is a doc id, which is not a number here.
        measured = set(_MEASURED.findall(answer))
        unexplained = []
        for raw in _numbers(answer):
            v = _as_float(raw)
            if v is None or v in known:
                continue
            if 1800 <= v <= 2100 and float(v).is_integer():
                continue                     # a year
            if raw in in_names:
                continue                     # part of an event or film name
            if raw in measured:
                continue                     # a weight class or a distance
            unexplained.append(raw)
        # Before calling them unexplained: a listing that adds up to a total
        # the query computed *is* the computed ranking, written out. This is
        # the common case on `event_series`, where the answer gives every
        # edition's count and the per-edition figures are not in the record.
        adds_up = None
        if unexplained:
            nums = [x for x in (_as_float(r) for r in _numbers(answer))
                    if x is not None and not (1800 <= x <= 2100)]
            for t in computed:
                total, entries = _as_float(t.get("total")), t.get("entries")
                if total is None or not isinstance(entries, int):
                    continue
                if _subset_sums_to(nums, total, entries):
                    adds_up = (t.get("what"), entries, total)
                    break
        if unexplained and adds_up:
            pass  # the listing reconciles with the arithmetic; nothing to read
        elif unexplained:
            # A question the planner marked `wants_prose` is answered from
            # documents, so its figures are read rather than computed and the
            # check cannot settle them. Still worth listing -- r2-pub-024's
            # 9.76 and 9.87 were right, and the way to know was to read the
            # cited document -- but say which kind of flag it is, or the
            # reader has to work that out for each one.
            prose = (record.get("plan") or {}).get("wants_prose")
            why = (" -- the plan asked for prose, so these were probably read "
                   "from the cited document rather than computed; check them "
                   "against it" if prose else "")
            flags.append(f"FIGURE NOT IN THE ARITHMETIC: {unexplained[:6]} appear in the "
                         f"answer but not among the figures the query computed "
                         f"({sorted(known)[:8]}){why}")

    # 4. status against the text
    #
    # Note what is NOT flagged here. `not_in_corpus` with citations attached is
    # correct, not a defect: the public set's own key gives gold documents for
    # two of its four not_in_corpus questions (r2-pub-046, r2-pub-049), because
    # the documents are what establish that the corpus does not settle it. An
    # earlier version of this check called that a problem, which would have
    # pushed the system toward citing nothing on exactly the questions where
    # citations are worth most.
    if status == "answered" and any(p in answer.lower() for p in _REFUSAL):
        flags.append("status is answered but the answer reads like a refusal")
    if status in ("answered", "partial") and not answer.strip():
        flags.append("status claims an answer and there is no answer text")

    # 3. the route
    tools = [s.get("tool", "") for s in steps]
    if (status in ("answered", "partial")
            and any("search_chunks" in t or t in ("chunks_containing",
                                                  "vector_search_chunks")
                    for t in tools)
            and record.get("route") not in ("text", "evidence")):
        # Only when it then answered. A route that found nothing, widened and
        # reported not_in_corpus did exactly what it is built to do, and
        # flagging that is how a worklist becomes something nobody reads.
        flags.append(f"route {record.get('route')!r} answered from full-text search "
                     f"after its own queries found nothing")
    if not any(s.get("tigergraph") for s in steps):
        flags.append("no TigerGraph step at all")
    if sum(1 for s in steps if s.get("tigergraph") and s.get("rows", 0) > 0) == 0:
        flags.append("every TigerGraph query returned zero rows")

    # 6. a superlative the arithmetic settled and the answer did not use
    #
    # This is Round 1's failure class stated as a check. The planner says the
    # question asked for an extreme; the code then ranked the candidates and
    # wrote down who holds the top. If the answer names neither that holder
    # nor that figure, the model answered a superlative from somewhere other
    # than the ranking that was handed to it, and that is the single most
    # expensive way to be wrong here.
    #
    # The test is deliberately generous: a question can ask for the extreme
    # and want the number rather than the name ("how many did the top nation
    # win"), and several tallies can be computed for one question when only
    # one of them is the one asked about. So it flags only when *no* tally's
    # top holder and *no* tally's top figure appears anywhere in the answer.
    extreme = (record.get("plan") or {}).get("extreme") or ""
    if extreme in ("most", "fewest") and computed and status in ("answered", "partial"):
        key = "highest" if extreme == "most" else "lowest"
        low = answer.lower()
        used, boxes = False, []
        for t in asked:
            box = t.get(key) or {}
            holders = _names(box.get("holders"))
            if not holders:
                continue
            boxes.append((t.get("what"), box.get("value"), box.get("holders")))
            value = _as_float(box.get("value"))
            if any(h in low for h in holders) or (value is not None and value in in_answer):
                used = True
                break
        if boxes and not used:
            what, value, holders = boxes[0]
            flags.append(f"SUPERLATIVE NOT TAKEN FROM THE ARITHMETIC: the plan asked "
                         f"for the {extreme}, the ranking puts {holders} at "
                         f"{value} in {what!r}, and the answer names neither that "
                         f"holder nor that figure")

    # 5. over-citation
    n = len(record.get("citations") or [])
    if n > 25:
        flags.append(f"{n} citations -- more documents than an answer usually rests on")
    return flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("answers", help="an answers_*.jsonl written by src.r2.run")
    ap.add_argument("--quiet", action="store_true", help="counts only, no worklist")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in open(args.answers, encoding="utf-8")
               if line.strip()]
    flagged = [(r, audit_one(r)) for r in records]
    worklist = sorted(((r, f) for r, f in flagged if f),
                      key=lambda rf: -len(rf[1]))

    print(f"\n{len(records)} answers in {args.answers}")
    print(f"  by status : " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(r.get("status") for r in records).items())))
    print(f"  by route  : " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(r.get("route") for r in records).items())))
    cites = [len(r.get("citations") or []) for r in records]
    if cites:
        print(f"  citations : {sum(cites)} in all, median "
              f"{sorted(cites)[len(cites) // 2]}, most {max(cites)}")
    withmath = sum(1 for r in records if r.get("computed"))
    print(f"  arithmetic: {withmath} of {len(records)} answers rest on a computed figure")

    # Efficiency is 10% of the score and nothing else in the repository shows
    # it per question. A single answer that cost ten times the median is
    # usually a route that widened to full-text search, so this doubles as a
    # pointer at the same answers the route flag catches.
    def _dist(label, values, unit):
        if not values:
            return
        v = sorted(values)
        print(f"  {label}: total {sum(v):,}{unit}  median {v[len(v) // 2]:,}{unit}  "
              f"worst {v[-1]:,}{unit}")
    _dist("tokens    ", [r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in records], "")
    _dist("latency   ", [r.get("latency_ms", 0) for r in records], "ms")
    _dist("model call", [sum(1 for s in (r.get("trace") or []) if not s.get("tigergraph"))
                         for r in records], "")
    worst = sorted(records, key=lambda r: -(r.get("input_tokens", 0) + r.get("output_tokens", 0)))[:3]
    if worst:
        print("  dearest   : " + ", ".join(
            f"{r.get('qid')} ({r.get('input_tokens', 0) + r.get('output_tokens', 0):,} tok, "
            f"route={r.get('route')})" for r in worst))
    print(f"\n{len(worklist)} of {len(records)} worth reading by hand.")

    if not args.quiet:
        for r, fl in worklist:
            print(f"\n--- {r.get('qid')}  [{r.get('status')}, route={r.get('route')}]")
            print(f"    {str(r.get('answer'))[:220]}")
            for f in fl:
                print(f"    ! {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
