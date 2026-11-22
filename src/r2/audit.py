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
_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

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
    for t in computed:
        holders = (t.get("highest") or {}).get("holders") or []
        if len(holders) > 1:
            named = sum(1 for h in holders if h.split(" | ")[0][:24].lower() in answer.lower())
            if named < len(holders) and "tie" not in answer.lower():
                flags.append(f"TIE NOT REPORTED: {len(holders)} hold the top "
                             f"{t.get('highest', {}).get('value')} in "
                             f"{t.get('what')!r}; the answer names {named} and does "
                             f"not say they are level")

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
        unexplained = []
        for raw in _numbers(answer):
            v = _as_float(raw)
            if v is None or v in known:
                continue
            if 1800 <= v <= 2100 and float(v).is_integer():
                continue                     # a year
            if raw in in_names:
                continue                     # part of an event or film name
            unexplained.append(raw)
        if unexplained:
            flags.append(f"FIGURE NOT IN THE ARITHMETIC: {unexplained[:6]} appear in the "
                         f"answer but not among the figures the query computed "
                         f"({sorted(known)[:8]})")

    # 4. status against the text
    if status == "not_in_corpus" and record.get("citations"):
        flags.append("not_in_corpus with citations attached")
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
