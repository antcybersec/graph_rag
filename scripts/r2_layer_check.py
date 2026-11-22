"""DEV PROBE -- not part of the answering system. Never imported by run.sh.

Asks one question: can the parsed record layer reach the published answers at
all? If a family of questions is unreachable here, no amount of agent design
will recover it, so this runs before any graph or LLM work. Each family is
computed with plain Python over src.r2.parse records and compared to the gold
answer in questions_public.jsonl.

This file hard-codes question parameters on purpose, which is exactly what
Round 2 rule 4 forbids in a *system*. It is a measurement harness: the shipped
pipeline derives these parameters from the question with a model, and the
scorer that produces metrics.json only ever compares answers to gold.
"""
import collections
import json
import sys

sys.path.insert(0, "/Users/anantkumar/graph_rag")
from src.common.infobox import norm_key
from src.r2.parse import classify, iter_docs

CORPUS = "data/corpus/corpus_r2.jsonl"
QUESTIONS = "/Users/anantkumar/Downloads/r2/questions_public.jsonl"

R = classify(iter_docs(CORPUS))
EVENTS, FILMS, PEOPLE = R["events"], R["films"], R["people"]
GOLD = {q["qid"]: q for q in (json.loads(l) for l in open(QUESTIONS, encoding="utf-8"))}

results = []


def check(qid, got, expect=None):
    """Compare to a projection of the gold answer; expect=None means use it whole."""
    want = GOLD[qid]["answer"] if expect is None else expect
    ok = got == want
    results.append((qid, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {qid}  {GOLD[qid]['question'][:62]}")
    if not ok:
        print(f"        got  {json.dumps(got, ensure_ascii=False)[:230]}")
        print(f"        want {json.dumps(want, ensure_ascii=False)[:230]}")


def events_in(sport=None, year=None, season="Summer"):
    out = EVENTS
    if sport:
        out = [e for e in out if e["sport_key"] == norm_key(sport)]
    if year:
        out = [e for e in out if e["games_year"] == year and e["season"] == season]
    return out


# --- repeat champions in a sport: same athlete, same event, 2+ editions -------
def repeat_champions(sport, season="Summer"):
    by = collections.defaultdict(set)
    for e in events_in(sport=sport):
        if e["season"] != season:
            continue
        for m in e["medals"]:
            if m["medal"] == "gold" and not m["multi"]:
                by[(m["recipient"], e["event_name"])].add(e["games_year"])
    return sorted(
        ({"athlete": a, "event": ev, "years": sorted(y)} for (a, ev), y in by.items() if len(y) > 1),
        key=lambda r: r["athlete"],
    )


print("\n=== repeat champions ===")
check("r2-pub-001", repeat_champions("Judo"))
check("r2-pub-002", repeat_champions("Fencing"))
check("r2-pub-032", repeat_champions("Snowboarding", season="Winter"))

# --- golds by nation ---------------------------------------------------------
print("\n=== golds by nation ===")


def golds_by_nation(sport, year, season="Summer"):
    c = collections.Counter()
    for e in events_in(sport=sport, year=year, season=season):
        for m in e["medals"]:
            if m["medal"] == "gold" and m["noc"]:
                c[m["noc"]] += 1
    return c


def top_nation(sport, year, season="Summer"):
    c = golds_by_nation(sport, year, season)
    noc, n = c.most_common(1)[0]
    return {"noc": noc, "golds": n}


check("r2-pub-006", top_nation("Wrestling", 2024))
check("r2-pub-007", top_nation("Judo", 2020))
check("r2-pub-033", top_nation("Rowing", 2024))
c = golds_by_nation("Canoeing", 2024)
check("r2-pub-008", {"count": len(c), "nations": sorted(c)})
check("r2-pub-003", sorted(
    ({"event": e["event_name"], "gold": m["recipient"]}
     for e in events_in(sport="Wrestling", year=2024) for m in e["medals"]
     if m["medal"] == "gold" and m["noc"] == "JPN"),
    key=lambda r: r["event"]))

# --- trend of nations across editions of one event series --------------------
print("\n=== series trend ===")


def series_trend(sport, event_name, season="Summer"):
    rows = sorted(
        ({"year": e["games_year"], "nations": e["nations"]}
         for e in EVENTS
         if e["series_key"] == f"{norm_key(sport)}|{norm_key(event_name)}"
         and e["season"] == season and e["nations"] is not None),
        key=lambda r: r["year"])
    return {"rows": rows,
            "max_year": max(rows, key=lambda r: r["nations"])["year"],
            "min_year": min(rows, key=lambda r: r["nations"])["year"]}


check("r2-pub-009", series_trend("Athletics", "Women's 800 metres"))
check("r2-pub-010", series_trend("Taekwondo", "Men's 80 kg"))
check("r2-pub-011", series_trend("Cycling", "Women's cross-country"))
check("r2-pub-012", series_trend("Athletics", "Men's hammer throw"))

# --- athlete career ----------------------------------------------------------
print("\n=== athlete career ===")


def career(name, medals=("gold", "silver", "bronze"), include_team=False):
    """Medals where this athlete is the named medallist.

    Team lines name the whole squad in one string -- "Simone BilesGabby
    Douglas..." or "FRAAmandine Buchard..." -- and a name match inside one of
    those is a team medal, not an individual one. The published key counts only
    individual lines (it omits Biles's 2016 team gold and Riner's 2020 mixed
    team gold), so team medals are excluded by default and available on request
    rather than dropped from the graph.
    """
    key = norm_key(name)
    return sorted(
        ({"year": e["games_year"], "event": e["event_name"], "medal": m["medal"]}
         for e in EVENTS for m in e["medals"]
         if m["medal"] in medals and key in m["recipient_key"]
         and (include_team or not m["multi"])),
        key=lambda r: (r["year"], r["event"]))


check("r2-pub-015", career("Teddy Riner"))
check("r2-pub-014", career("Kaori Icho", medals=("gold",)))
check("r2-pub-013", career("Simone Biles"))
check("r2-pub-016", career("Usain Bolt", medals=("gold",)))
check("r2-pub-034", career("Xinyi Gu", medals=("gold",)))

# --- superlative within a sport at one Games ---------------------------------
print("\n=== superlative nations in a sport ===")


def extreme_nations(sport, year, which, season="Summer"):
    cand = [e for e in events_in(sport=sport, year=year, season=season) if e["nations"] is not None]
    pick = (max if which == "max" else min)(cand, key=lambda e: e["nations"])
    return {"event": pick["event_name"], "nations": pick["nations"]}


check("r2-pub-025", extreme_nations("Athletics", 2024, "min"))
check("r2-pub-026", extreme_nations("Cycling", 2024, "max"))
check("r2-pub-035", extreme_nations("Swimming", 2024, "max"))
check("r2-pub-027", extreme_nations("Shooting", 2024, "min"),
      expect={"event": GOLD["r2-pub-027"]["answer"]["event"],
              "nations": GOLD["r2-pub-027"]["answer"]["nations"]})

# --- single-event gold lookups ----------------------------------------------
print("\n=== event lookup ===")


def gold_of(sport, event_name, year, season="Summer"):
    for e in events_in(sport=sport, year=year, season=season):
        if e["event_key"] == norm_key(event_name):
            return next((m["recipient"] for m in e["medals"] if m["medal"] == "gold"), None)
    return None


check("r2-pub-004", [{"year": y, "gold": gold_of("Shooting", "Women's 10 metre air rifle", y)}
                     for y in (2012, 2016, 2024, 2028)])
check("r2-pub-036", gold_of("Shooting", "Women's 10 metre air rifle", 2028))
check("r2-pub-038", next(e["competitors"] for e in events_in(sport="Athletics", year=2024)
                         if e["event_key"] == norm_key("Men's decathlon")))

# pub-037 "the Games immediately after 2024" -- follow the corpus's own next edge
nxt = {e["next_year"] for e in events_in(sport="Gymnastics", year=2024) if e["next_year"]}
check("r2-pub-037", gold_of("Gymnastics", "Women's floor", max(nxt)))

# pub-029 partial: 2008 men's +100 kg judo is absent from the corpus
found = [{"year": y, "gold": gold_of("Judo", "Men's +100 kg", y)} for y in (2004, 2008, 2012, 2016)]
check("r2-pub-029", {"found": [f for f in found if f["gold"]],
                     "not_in_corpus": [f["year"] for f in found if not f["gold"]]})

# --- films -------------------------------------------------------------------
print("\n=== films ===")
check("r2-pub-005", sorted(({"film": f["name"], "year": f["year"]} for f in FILMS
                            if "Ron Howard" in f["directors"] and (f["year"] or 9999) < 1995),
                           key=lambda r: r["year"]))
spielberg = [f for f in FILMS if "Steven Spielberg" in f["directors"]
             and f["budget"] and f["gross"] and f["budget_currency"] == f["gross_currency"]]
check("r2-pub-017", next((f["name"] for f in spielberg if f["gross"] < 2 * f["budget"]), None))

# --- politicians -------------------------------------------------------------
print("\n=== politicians ===")
king = next(p for p in PEOPLE if p["name"] == "Angus King")
gov = [o for o in king["offices"] if o["successor"] and o["slot"] == 1]
check("r2-pub-019", [gov[0]["successor"], gov[0]["predecessor"]])
blunt = next(p for p in PEOPLE if p["name"] == "Matt Blunt")
pred = blunt["offices"][0]["predecessor"]
holden = next(p for p in PEOPLE if p["name"] == pred)
check("r2-pub-018", [pred, holden["offices"][0]["predecessor"]])

n = sum(1 for _, ok in results if ok)
print(f"\n{n}/{len(results)} families reachable from the parsed record layer")
