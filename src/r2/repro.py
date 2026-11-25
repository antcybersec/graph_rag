"""Re-answer a submitted question and report whether the answer reproduces.

    python -m src.r2.repro questions_public.jsonl data/results/r2/answers_public.jsonl --qid r2-pub-003
    python -m src.r2.repro questions_hidden.jsonl data/results/r2/answers_hidden.jsonl --sample 3

The brief disqualifies "answers that your system cannot reproduce", and
reproduction is the one claim that cannot be checked by reading the file: it
needs the question put through the agent a second time and the two records
held against each other. This does that, and says which of three things
happened rather than printing a diff and leaving the reader to judge.

What counts as reproduced, and why not byte equality
----------------------------------------------------
The composed sentence comes from a language model, so two runs phrase the same
fact differently and byte equality would fail on a system that is in fact
perfectly repeatable. What must not move is everything the score is computed
from:

  status      the honesty criterion reads this field directly
  figures     every number in the answer -- a count, a year, a medal tally.
              This is the check that matters: the arithmetic is done in code
              from graph rows, so if a figure moves, something non-deterministic
              is feeding it and the answer was luck.
  citations   the document ids, as a set
  retrieval   the route and the sequence of TigerGraph queries

MATCH      status, figures and citations all identical
CLOSE      status and figures identical; citations or wording moved
DIFFERENT  status or a figure moved -- this is the one that would be a finding

Exits non-zero if anything is DIFFERENT, so it can be run before submitting
and believed.

One caveat worth stating plainly: a reproduction run has to use the same
provider and model as the run that produced the file. Re-answering with a
local model and getting the same figures is evidence that the arithmetic is
deterministic, which is worth having, but it is not evidence that the
submitted file reproduces. The verdict line says which provider it used.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from collections import Counter
import random
import re
import sys
from pathlib import Path

_NUM = re.compile(r"\d+(?:\.\d+)?")

# Capitalised runs, which in these answers are the nations, athletes, films and
# offices the answer names. The leading-word blocklist is what keeps "The",
# "No" and "According" out of the set: every sentence starts capitalised, so
# without it every reworded opening would read as a changed entity.
_NAME = re.compile(r"\b[A-Z][\w'\u2019-]*(?:\s+[A-Z][\w'\u2019-]*)*")
_NOT_A_NAME = {
    "A", "An", "The", "This", "That", "These", "Those", "There", "It", "Its",
    "No", "None", "Not", "Nothing", "According", "Based", "However", "While",
    "When", "Where", "Which", "Who", "What", "Why", "How", "If", "In", "On",
    "At", "For", "From", "Of", "To", "By", "With", "And", "Or", "But", "As",
    "Both", "Neither", "Either", "Only", "All", "Two", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine", "Ten", "First", "Second", "Third",
    "Corpus", "Document", "Documents", "Source", "Sources", "Olympics",
    "Olympic", "Games", "Summer", "Winter", "Men", "Women", "Mens", "Womens",
}


def names(text: object) -> set:
    """The proper names an answer states, as a set.

    `figures` compares only digits, so an argmax that moved from one holder to
    another with the same count reproduced as a clean MATCH -- "France won 13
    silver medals" against "Italy won 13 silver medals". The number is the
    easy half of a superlative; which entity holds it is the answer.
    """
    blob = json.dumps(text, ensure_ascii=False) if not isinstance(text, str) else text
    out = set()
    for raw in _NAME.findall(blob):
        parts = [w for w in raw.split() if w not in _NOT_A_NAME]
        if parts:
            cleaned = " ".join(parts)
            if len(cleaned) > 2:
                out.add(cleaned)
    return out


def figures(text: object) -> list[str]:
    """Every number in an answer, in order, as written.

    Returned in order, but `compare` judges them as a multiset, because the
    composed sentence is model-written and restates a varying amount of the
    question. Measured on a rehearsal, an ordered comparison called 2 of 3
    correct reproductions DIFFERENT.

    The cost of that, stated plainly: "13 silver and 28 in all" against "28
    silver and 13 in all" holds the same multiset, so it comes out CLOSE
    rather than DIFFERENT. Both ordered lists are printed, so the reordering
    is visible to whoever is reading -- but a gate keyed on DIFFERENT alone
    will not stop it. Pairing each figure with the label beside it would, and
    is the fix if a real swap is ever observed; it has not been, while the
    false alarms were measured at two in three.
    """
    return _NUM.findall(json.dumps(text, ensure_ascii=False) if not isinstance(text, str) else text)


def retrieval(record: dict) -> list[str]:
    """The route plus the ordered TigerGraph steps, as a comparable list."""
    out = [f"route={record.get('route', '')}"]
    for step in record.get("trace", []) or []:
        if step.get("tigergraph"):
            out.append(str(step.get("tool", "?")))
    return out


def compare(old: dict, new: dict) -> dict:
    """Hold two records for the same question against each other."""
    f_old, f_new = figures(old.get("answer")), figures(new.get("answer"))
    c_old = sorted(old.get("citations") or [])
    c_new = sorted(new.get("citations") or [])
    n_old, n_new = names(old.get("answer")), names(new.get("answer"))
    same_status = old.get("status") == new.get("status")
    same_cites = c_old == c_new
    r_old, r_new = retrieval(old), retrieval(new)

    # Figures by the same substitution-not-addition rule as names, and as a
    # multiset rather than an ordered list.
    #
    # Measured on a rehearsal: 2 of 3 correct reproductions were reported
    # DIFFERENT by an ordered-list comparison, both because the composed
    # sentence is model-written and restates a different amount of the
    # question. r2-pub-021 went ['63.5','2028','40'] -> ['40'] by not
    # repeating "the men's 63.5 kg at the 2028 Games"; its answer -- suspended
    # two hours, gusts over 40 km/h -- was identical. r2-pub-010 gave the same
    # seven per-edition counts and the same most and fewest, with the
    # restatement at the end phrased once instead of twice.
    #
    # A gate that fires on two of three good reproductions is worse than no
    # gate, because the judging call reads it as "this does not reproduce".
    # So: each side holding a figure the other lacks is a substitution and
    # stays DIFFERENT; one side being a subset of the other said more or less
    # of the same thing; an equal multiset in a different order is a
    # reordering. Both of those are CLOSE, printed, and left to a human --
    # which keeps the "13 silver and 28 in all" / "28 silver and 13 in all"
    # swap visible without calling a rewording a failure.
    k_old, k_new = Counter(f_old), Counter(f_new)
    figures_swapped = bool((k_old - k_new) and (k_new - k_old))
    same_figures = f_old == f_new
    figures_comparable = not figures_swapped

    # A substitution, not an addition. One answer saying strictly more than the
    # other is a rewording ("Japan" against "Japan (JPN)") and is not a changed
    # answer; each naming something the other does not is one entity standing
    # where another stood, which is the failure a superlative hides behind an
    # unchanged number.
    swapped = sorted(n_old - n_new) if (n_old - n_new) and (n_new - n_old) else []
    same_names = not swapped

    if not same_status or figures_swapped or not same_names:
        verdict = "DIFFERENT"
    elif same_cites and r_old == r_new and same_figures:
        verdict = "MATCH"
    else:
        # The docstring promised four things must not move and the verdict
        # used two: a route change and a halved citation set both passed as
        # MATCH, while "returns no DIFFERENT" is the submission's gate.
        verdict = "CLOSE"

    text_old = str(old.get("answer") or "")
    text_new = str(new.get("answer") or "")
    return {
        "qid": old.get("qid"),
        "verdict": verdict,
        "status": [old.get("status"), new.get("status")],
        "same_status": same_status,
        "figures": [f_old, f_new],
        "same_figures": same_figures,
        "figures_comparable": figures_comparable,
        "figures_only_in_submitted": sorted((k_old - k_new).elements()),
        "figures_only_in_rerun": sorted((k_new - k_old).elements()),
        "names": [sorted(n_old), sorted(n_new)],
        "same_names": same_names,
        "names_swapped": swapped,
        "citations": [len(c_old), len(c_new)],
        "same_citations": same_cites,
        "citations_lost": [c for c in c_old if c not in c_new],
        "citations_gained": [c for c in c_new if c not in c_old],
        "retrieval": [r_old, r_new],
        "same_retrieval": r_old == r_new,
        "text_similarity": round(difflib.SequenceMatcher(None, text_old, text_new).ratio(), 3),
        "answers": [text_old, text_new],
    }


def report(c: dict, verbose: bool) -> None:
    mark = {"MATCH": "ok  ", "CLOSE": "~~  ", "DIFFERENT": "!!  "}[c["verdict"]]
    print(f"\n{mark}{c['qid']}  {c['verdict']}")
    print(f"      status      {c['status'][0]} -> {c['status'][1]}"
          f"{'' if c['same_status'] else '   MOVED'}")
    if c["same_figures"]:
        note = ""
    elif c["figures_comparable"]:
        gone = c["figures_only_in_submitted"]
        extra = c["figures_only_in_rerun"]
        note = ("   reordered, same figures" if not gone and not extra
                else f"   restated differently (only in submitted: {gone}, "
                     f"only in re-run: {extra})")
    else:
        note = (f"   SUBSTITUTED (submitted had {c['figures_only_in_submitted']}, "
                f"re-run has {c['figures_only_in_rerun']})")
    print(f"      figures     {c['figures'][0]} -> {c['figures'][1]}{note}")
    print(f"      citations   {c['citations'][0]} -> {c['citations'][1]}"
          f"{'' if c['same_citations'] else '   ' + str(len(c['citations_lost'])) + ' lost, ' + str(len(c['citations_gained'])) + ' gained'}")
    print(f"      retrieval   {'identical' if c['same_retrieval'] else 'MOVED'}"
          f"  ({len(c['retrieval'][0])} steps -> {len(c['retrieval'][1])})")
    print(f"      wording     {c['text_similarity']:.0%} similar")
    if verbose or c["verdict"] == "DIFFERENT":
        print(f"      submitted   {c['answers'][0][:300]!r}")
        print(f"      re-run      {c['answers'][1][:300]!r}")
        if not c["same_retrieval"]:
            print(f"      queries     {c['retrieval'][0]}\n              -> {c['retrieval'][1]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("questions")
    ap.add_argument("answers")
    ap.add_argument("--qid", default="", help="comma-separated; default: a random sample")
    ap.add_argument("--sample", type=int, default=2, help="how many to re-answer if no --qid")
    ap.add_argument("--seed", type=int, default=0, help="0 means genuinely random")
    ap.add_argument("--log", default="data/results/r2/repro_querylog.jsonl")
    ap.add_argument("--out", default="", help="write the comparison as JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="print both answers in full")
    args = ap.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from src.r2.agent import Agent
    from src.r2.conn import describe, get_connection
    from src.r2.trace import QueryLog

    questions = {q["qid"]: q["question"]
                 for q in (json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip())}
    submitted = {a["qid"]: a
                 for a in (json.loads(l) for l in open(args.answers, encoding="utf-8") if l.strip())}

    if args.qid:
        qids = [q.strip() for q in args.qid.split(",") if q.strip()]
        missing = [q for q in qids if q not in submitted]
        if missing:
            print(f"not in {args.answers}: {', '.join(missing)}", file=sys.stderr)
            return 2
    else:
        # Only questions that actually got an answer are worth re-running.
        pool = sorted(q for q, a in submitted.items() if a.get("trace"))
        if not pool:
            print(f"{args.answers} has no answered questions to reproduce", file=sys.stderr)
            return 2
        rng = random.Random(args.seed or None)
        qids = rng.sample(pool, min(args.sample, len(pool)))
        print(f"re-answering {len(qids)} of {len(pool)} answered questions, chosen at random")

    # Which provider and model produced the submitted file is recorded in the
    # file, on every model step -- so read it from there rather than naming it
    # here. A hardcoded name was wrong ("google", where every trace in both
    # submitted files says "gemini"), which meant this tool printed "these are
    # not the answers' own provider" on every single run, including the one
    # configuration where it was false. A disclaimer that fires always says
    # nothing, and in a judging call it says the opposite of the truth.
    submitted_pairs = {(str(step.get("provider")), str(step.get("model")))
                       for a in submitted.values()
                       for step in (a.get("trace") or [])
                       if isinstance(step, dict) and step.get("provider")}
    provider = os.environ.get("GEN_PROVIDER", "gemini")
    model = os.environ.get("GEN_MODEL", "(default)")
    print(f"connecting: {describe()}")
    print(f"answering with: provider={provider} model={model}")
    if submitted_pairs:
        shown = ", ".join(f"{p}/{m}" for p, m in sorted(submitted_pairs))
        print(f"submitted file was produced with: {shown}")
        if not any(p == provider for p, _ in submitted_pairs):
            print("  NOTE: that is not the provider answering now, so a match here is\n"
                  "        evidence that the arithmetic is deterministic, not that the\n"
                  "        submitted answers reproduce.")
        elif not any((p, m) == (provider, model) for p, m in submitted_pairs):
            print("  NOTE: same provider, different model -- a match is strong evidence\n"
                  "        but not a like-for-like reproduction.")
    else:
        print("  NOTE: the submitted file records no provider, so this run cannot confirm\n"
              "        it was produced with the same one.")

    conn = get_connection()
    agent = Agent(conn, query_log=QueryLog(args.log))

    comparisons = []
    for qid in qids:
        fresh = agent.answer(qid, questions[qid])
        c = compare(submitted[qid], fresh)
        comparisons.append(c)
        report(c, args.verbose)

    tally = {v: sum(1 for c in comparisons if c["verdict"] == v)
             for v in ("MATCH", "CLOSE", "DIFFERENT")}
    print(f"\n{len(comparisons)} re-answered: {tally['MATCH']} MATCH, "
          f"{tally['CLOSE']} CLOSE, {tally['DIFFERENT']} DIFFERENT")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"provider": provider, "model": model, "tally": tally,
                        "comparisons": comparisons}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        print(f"-> {args.out}")
    if tally["DIFFERENT"]:
        print("a status or a figure moved between runs -- that is the finding to chase")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
