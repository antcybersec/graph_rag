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
import random
import re
import sys
from pathlib import Path

_NUM = re.compile(r"\d+(?:\.\d+)?")


def figures(text: object) -> list[str]:
    """Every number in an answer, in order, as written.

    Compared as a list rather than a set: "13 silver and 28 in all" differs
    from "28 silver and 13 in all", and that difference matters.
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
    same_status = old.get("status") == new.get("status")
    same_figures = f_old == f_new
    same_cites = c_old == c_new
    r_old, r_new = retrieval(old), retrieval(new)

    if not same_status or not same_figures:
        verdict = "DIFFERENT"
    elif same_cites:
        verdict = "MATCH"
    else:
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
    print(f"      figures     {c['figures'][0]} -> {c['figures'][1]}"
          f"{'' if c['same_figures'] else '   MOVED'}")
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

    provider = os.environ.get("GEN_PROVIDER", "google")
    model = os.environ.get("GEN_MODEL", "(default)")
    print(f"connecting: {describe()}")
    print(f"answering with: provider={provider} model={model}")
    if provider != "google":
        print("  NOTE: this is not the provider the submitted file was produced with, so a\n"
              "        match here is evidence that the arithmetic is deterministic, not that\n"
              "        the submitted answers reproduce.")

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
