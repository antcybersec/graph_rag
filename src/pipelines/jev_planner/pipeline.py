"""Pipeline 6: selection planning with TypeSafe System One -- no Gemini on the answer path.

The investigator plans with a generative model: it writes out a query plan as
free text and we parse it. That works, but it costs a Gemini call per question,
and Gemini's free tier is 500 calls a day -- one benchmark run. Every
measurement in this project has died on that wall, and a 429 during the Round 2
live demo would be unrecoverable.

The planner does not actually need generation. Every value it emits already
exists as a row in the graph: 47 sports, 20 Games, 316 venues, 475 event names.
Choosing among known candidates is a *selection* problem, which is what a System
One model does natively -- and the numbers fit Choice's 255-option limit once
candidates are narrowed by what stage 1 established:

    stage 1 (one request) : operation (5) + sport (47) + Games (20), asked as
                            three independent Choices over the question text
    code                  : thresholds, dates and venue strings by regex --
                            exact lookups stay in code, per TypeSafe's guidance
    stage 2 (one request) : the event name (<= 43 after narrowing) or the venue
                            (<= 37 within one Games), only when stage 1 needs it
    execute               : the same GSQL and the same templated answer the
                            event_graph pipeline already uses -- zero LLM calls

So: two System One calls per question, no Gemini, and the answer is composed by
code from graph rows rather than written by a model.

Confidence is a first-class part of the design, not decoration. Every selection
carries a probability distribution; anything below CONFIDENCE_FLOOR (TypeSafe's
documented universal floor) hands the question to the investigator instead of
guessing. `route` records which path answered, so the benchmark can report how
often the cheap path was enough.
"""
import re
from typing import Optional

from src.common.infobox import OLYMPIC_CALENDAR, games_id, norm_key
from src.common.tg_conn import get_connection
from src.common.typesafe import ask, is_enabled
from src.pipelines.event_graph.pipeline import QueryPlan, _venue_candidates, execute_plan
from src.pipelines.investigator import pipeline as investigator_pipeline

# TypeSafe's documented universal floor: below this, route away rather than act.
CONFIDENCE_FLOOR = 0.6
# The operation decides which query runs at all, so a wrong pick is costlier
# than a wrong sport (which usually just returns nothing and falls back).
OPERATION_FLOOR = 0.7
MAX_CHOICE_OPTIONS = 255  # Choice accepts up to 255 options

OPERATIONS = {
    "count_events_over": "How many events of one sport at one Games had more than N competitors",
    "top_event_by_competitors": "Which event of one sport at one Games had the most competitors",
    "event_field": "A single field (nations, competitors, gold, venue, date) of one named event",
    "gold_at_venue_date": "Who won gold in the event held at a named venue on a named date",
    "gold_previous_games": "Who won gold in a named event at the Games immediately before a given year",
    "other": "None of the above fits this question",
}

_catalog: dict = {}


def _build_catalog(conn) -> dict:
    """Candidate lists, read once from the graph -- the graph is the source of
    truth for what can be selected, so the model can never pick a value the
    database does not hold."""
    if _catalog:
        return _catalog
    events = [v["attributes"] for v in conn.getVertices("OlympicEvent", limit=5000)]
    _catalog["sports"] = sorted({e["sport"] for e in events})
    _catalog["games"] = sorted({f"{e['games_year']} {e['season']}" for e in events})
    _catalog["events_by_sport_games"] = {}
    _catalog["venues_by_games"] = {}
    for e in events:
        game = f"{e['games_year']} {e['season']}"
        _catalog["events_by_sport_games"].setdefault((e["sport"], game), set()).add(e["event_name"])
        if e.get("venue"):
            _catalog["venues_by_games"].setdefault(game, set()).add(e["venue"])
    _catalog["venue_keys"] = {norm_key(e["venue"]) for e in events if e.get("venue")}
    return _catalog


def _extract(question: str) -> dict:
    """Exact values the model should never be asked to reproduce: numbers, dates
    and the verbatim venue string. Regex is exact and free; selection is not."""
    found = {}
    if m := re.search(r"(?:more than|larger than|over|at least|above)\s+(\d+)", question, re.I):
        found["min_competitors"] = int(m.group(1))
    if m := re.search(r"immediately (?:before|preceding)\s+(\d{4})", question, re.I):
        found["before_year"] = int(m.group(1))
    if m := re.search(r"held at (.+?) on (.+?)(?:\s+at the \d{4}\s+(?:Summer|Winter) Olympics)?\?*$", question, re.I):
        found["venue"] = m.group(1).strip()
        found["date"] = m.group(2).strip()
    if m := re.search(r"\b(\d{4})\s+(Summer|Winter)\b", question, re.I):
        found["year"], found["season"] = int(m.group(1)), m.group(2).capitalize()
    return found


def _choice(answers: dict, key: str, floor: float) -> tuple:
    """(value, confidence, ok) -- `ok` is False when the model is unsure or
    picked the explicit no-match option."""
    answer = answers.get(key) or {}
    value, confidence = answer.get("choice"), answer.get("confidence", 0.0)
    ok = bool(value) and value not in ("other", "none") and confidence >= floor
    return value, confidence, ok


def _plan(conn, question: str, tracker, question_id, trace: list) -> Optional[QueryPlan]:
    catalog = _build_catalog(conn)
    extracted = _extract(question)

    # --- stage 1: three independent selections over one state, in one request ---
    answers = ask(
        {"question": question},
        {
            "operation": {"type": "choice",
                          "instructions": "Which kind of question is `question` asking?",
                          "criteria": OPERATIONS},
            "sport": {"type": "choice",
                      "instructions": "Which Olympic sport is `question` about? Choose none if it names no sport.",
                      "criteria": {**{s: None for s in catalog["sports"][:MAX_CHOICE_OPTIONS - 1]}, "none": "The question names no sport"}},
            "games": {"type": "choice",
                      "instructions": ("Which Olympic Games is `question` asking about, by year and season? "
                                       "A host city implies its Games. Choose none if it names no Games, or if it "
                                       "only refers to the Games *before* some year."),
                      "criteria": {**{g: None for g in catalog["games"][:MAX_CHOICE_OPTIONS - 1]}, "none": "The question names no Games"}},
        },
        tracker=tracker, question_id=question_id, pipeline="jev_planner",
    )
    operation, op_conf, op_ok = _choice(answers, "operation", OPERATION_FLOOR)
    sport, sport_conf, sport_ok = _choice(answers, "sport", CONFIDENCE_FLOOR)
    game, game_conf, game_ok = _choice(answers, "games", CONFIDENCE_FLOOR)
    trace.append({"stage": "select_operation_sport_games", "operation": operation, "operation_confidence": op_conf,
                  "sport": sport, "sport_confidence": sport_conf, "games": game, "games_confidence": game_conf,
                  "extracted": extracted})
    if not op_ok:
        return None

    year, season = None, None
    if game_ok:
        year, season = int(game.split()[0]), game.split()[1]
    elif extracted.get("year"):
        year, season = extracted["year"], extracted.get("season")

    if operation == "gold_at_venue_date":
        # The venue string is in the question verbatim; only fall back to
        # selection when the literal text matches nothing in the graph.
        venue = extracted.get("venue")
        if venue and _venue_candidates(venue, catalog["venue_keys"]):
            return QueryPlan(operation=operation, venue=venue, date=extracted.get("date"),
                             year=year, season=season)
        options = sorted(catalog["venues_by_games"].get(game, set())) if game_ok else []
        if not options:
            return None
        picked = ask(
            {"question": question, "venue_text": venue or ""},
            {"venue": {"type": "choice",
                       "instructions": "Which venue does `question` refer to? `venue_text` is the phrase it used.",
                       "criteria": {**{v: None for v in options[:MAX_CHOICE_OPTIONS - 1]}, "none": "No listed venue matches"}}},
            tracker=tracker, question_id=question_id, pipeline="jev_planner",
        )
        venue_pick, venue_conf, venue_ok = _choice(picked, "venue", CONFIDENCE_FLOOR)
        trace.append({"stage": "select_venue", "venue": venue_pick, "confidence": venue_conf, "options": len(options)})
        if not venue_ok:
            return None
        return QueryPlan(operation=operation, venue=venue_pick, date=extracted.get("date"), year=year, season=season)

    if operation in ("count_events_over", "top_event_by_competitors"):
        if not (sport_ok and year and season):
            return None
        return QueryPlan(operation=operation, sport=sport, year=year, season=season,
                         min_competitors=extracted.get("min_competitors"))

    # event_field and gold_previous_games both need the event name. Narrow the
    # candidates to the sport and Games that stage 1 established -- for
    # "immediately before YEAR" that is the PREVIOUS Games in the real calendar.
    if not sport_ok:
        return None
    if operation == "gold_previous_games":
        target_year = extracted.get("before_year") or year
        target_season = season or extracted.get("season")
        if not (target_year and target_season) or target_year not in OLYMPIC_CALENDAR[target_season]:
            return None
        calendar = OLYMPIC_CALENDAR[target_season]
        lookup_game = games_id(calendar[calendar.index(target_year) - 1], target_season)
        year, season = target_year, target_season  # execute_plan itself steps back a Games
    else:
        if not (year and season):
            return None
        lookup_game = games_id(year, season)

    options = sorted(catalog["events_by_sport_games"].get((sport, lookup_game), set()))
    if not options:
        return None
    picked = ask(
        {"question": question, "sport": sport, "games": lookup_game},
        {"event": {"type": "choice",
                   "instructions": f"Which {sport} event at the {lookup_game} Olympics is `question` about?",
                   "criteria": {**{e: None for e in options[:MAX_CHOICE_OPTIONS - 1]}, "none": "No listed event matches"}}},
        tracker=tracker, question_id=question_id, pipeline="jev_planner",
    )
    event_name, event_conf, event_ok = _choice(picked, "event", CONFIDENCE_FLOOR)
    trace.append({"stage": "select_event", "event": event_name, "confidence": event_conf, "options": len(options)})
    if not event_ok:
        return None

    field = "nations" if re.search(r"how many nations", question, re.I) else None
    return QueryPlan(operation=operation, sport=sport, year=year, season=season,
                     event_name=event_name, field=field)


def answer_question(conn, question: str, tracker=None, question_id=None) -> dict:
    trace: list = []
    plan = None
    if is_enabled():
        try:
            plan = _plan(conn, question, tracker, question_id, trace)
        except Exception as exc:
            # Any planning failure -- service error, malformed answer, missing
            # candidates -- means fall back to the agent, never answer badly.
            trace.append({"stage": "error", "error": f"{type(exc).__name__}: {exc}"[:200]})

    result = None
    if plan is not None:
        try:
            result = execute_plan(conn, plan, question)
        except Exception as exc:
            trace.append({"stage": "execute_error", "error": f"{type(exc).__name__}: {exc}"[:200]})

    if result is None:
        # Low confidence, an unsupported shape, or no matching rows: hand the
        # question to the generative agent rather than answer badly.
        fallback = investigator_pipeline.answer_question(conn, question, tracker=tracker, question_id=question_id)
        return fallback | {"route": "fallback_investigator", "trace": trace,
                           "plan": plan.model_dump() if plan else None}

    evidence = result["evidence"]
    return {
        "answer": result["answer"],
        "short_answer": result["short_answer"],
        "route": "jev_planner",
        "plan": plan.model_dump(),
        "trace": trace,
        "retrieved_doc_ids": sorted({e["event_id"] for e in evidence}),
        "context_text": "\n".join(
            f"[{e['event_id']}] {e['title']}: competitors={e['competitors']}, nations={e['nations']}, "
            f"venue={e['venue']}, date={e['date_text']}, gold={e['gold']}" for e in evidence[:12]),
    }


if __name__ == "__main__":
    conn = get_connection()
    for q in [
        "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
        "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
        "Who won the gold medal in the event held at Richmond Olympic Oval on 14 February 2010?",
        "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?",
        "Which sailing event at Sydney 2000 had the biggest field?",
    ]:
        r = answer_question(conn, q)
        print(f"\nQ: {q}\n  -> {r['short_answer']!r} via {r['route']}")
        for step in r["trace"]:
            print("    ", {k: v for k, v in step.items() if k != "extracted"})
