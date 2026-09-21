"""Deterministic ground truth for the eval templates -- no LLM, no database.

Every question in this eval set is a template over `[Infobox Olympic event]`
fields, so the correct answer can be computed directly from the corpus. That
makes this the cheapest available check on any pipeline's answers, and the only
one that works for the hidden set, which ships without gold answers.

It exists because a stale artifact nearly shipped: the hidden answers generated
on 2026-09-18 17:46 were produced two minutes before the fix at 3ab06cd, and
carried the capped-evidence bug (counting questions answered "12", the cap).
Comparing against this oracle catches that class of error for free.

This is a checker, not a pipeline: it hard-codes the five question shapes and
would answer nothing else. The pipelines are what generalise.

    python -m src.eval.oracle data/results/hidden_answers_jev.jsonl [...]
"""
import json
import re
import sys
from collections import defaultdict

from src.common.infobox import OLYMPIC_CALENDAR, date_key, norm_key, parse_event_doc

CORPUS_PATH = "data/raw_dataset/corpus/corpus.jsonl"


def load_events(path: str = CORPUS_PATH) -> list:
    with open(path) as f:
        return [e for e in (parse_event_doc(json.loads(line)) for line in f) if e]


def _same_sport(a: str, b: str) -> bool:
    return norm_key(a).rstrip("s") == norm_key(b).rstrip("s")


def answer(question: str, events: list):
    """The computed answer, or None when the question is not one of the templates."""
    if m := re.match(r"According to the provided corpus, how many (.+?) events at the (\d{4}) (Summer|Winter) Olympics had more than (\d+) competitors\?", question):
        sport, year, season, threshold = m[1], int(m[2]), m[3], int(m[4])
        pool = [e for e in events if _same_sport(sport, e["sport"]) and e["games_year"] == year and e["season"] == season]
        return str(sum(1 for e in pool if e["competitors"] >= 0 and e["competitors"] > threshold))

    if m := re.match(r"According to the provided corpus, which (.+?) event at the (\d{4}) (Summer|Winter) Olympics had the highest number of competitors\?", question):
        sport, year, season = m[1], int(m[2]), m[3]
        pool = [e for e in events if _same_sport(sport, e["sport"]) and e["games_year"] == year
                and e["season"] == season and e["competitors"] >= 0]
        return max(pool, key=lambda e: e["competitors"])["title"] if pool else None

    if m := re.match(r"How many nations competed in (.+?)\?$", question):
        hit = [e for e in events if norm_key(e["title"]) == norm_key(m[1])]
        return str(hit[0]["nations"]) if hit and hit[0]["nations"] >= 0 else None

    if m := re.match(r"Who won the gold medal in the (.+) event at the (Summer|Winter) Olympics held immediately before (\d{4})\?", question):
        described, season, year = m[1], m[2], int(m[3])
        calendar = OLYMPIC_CALENDAR[season]
        if year not in calendar:
            return None
        previous = calendar[calendar.index(year) - 1]
        # The question writes "<event name> <sport>", e.g. "men's pole vault athletics".
        hits = [e for e in events if e["games_year"] == previous and e["season"] == season
                and norm_key(described) == norm_key(f"{e['event_name']} {e['sport']}")]
        return hits[0]["gold"] if len(hits) == 1 else None

    if m := re.match(r"Who won the gold medal in the event held at (.+?) on (.+?)\?$", question):
        venue, date = m[1], m[2]
        # The Games may be named after the date ("... on 3 to 4 August at the 2012 Summer Olympics").
        games = re.search(r"at the (\d{4}) (Summer|Winter) Olympics$", date)
        if games:
            date = date[: games.start()].strip()
        hits = [e for e in events
                if norm_key(e["venue"]) == norm_key(venue) and e["date_key"] == date_key(date)
                and (not games or (e["games_year"] == int(games[1]) and e["season"] == games[2]))]
        golds = {e["gold"] for e in hits if e["gold"]}
        # Two events can share a venue and a date; then the question is ambiguous
        # and there is no single ground truth to check against.
        return hits[0]["gold"] if len(golds) == 1 else None

    return None


def score(paths: list):
    events = load_events()
    for path in paths:
        rows = [r for r in map(json.loads, open(path)) if "error" not in r]
        agreed, checked, misses = 0, 0, []
        for row in rows:
            truth = answer(row["question"], events)
            if truth is None:
                continue
            checked += 1
            given = row.get("short_answer") or ""
            # A multi-value answer is a hedge, not a match.
            if norm_key(given) == norm_key(truth):
                agreed += 1
            else:
                misses.append((row["qid"], row.get("qtype"), given, truth))
        print(f"{path}: agrees with the oracle on {agreed}/{checked} checkable rows ({len(rows)} rows total)")
        for qid, qtype, given, truth in misses:
            print(f"   {qid} {str(qtype):11s} answered={given!r}  oracle={truth!r}")


if __name__ == "__main__":
    score(sys.argv[1:] or ["data/results/hidden_answers_jev.jsonl"])
