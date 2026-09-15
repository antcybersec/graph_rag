"""Pipeline 4: Event-graph GraphRAG. One LLM call plans, the graph answers.

  question --(1 LLM call)--> typed QueryPlan (operation + sport/Games/venue/date/...)
           --> GSQL over the structured event layer (src/ingestion/build_event_graph.py)
           --> filter / count / argmax over OlympicEvent infobox fields
           --> templated answer citing the source documents

Falls back to plain RAG when the plan is `unsupported` or the graph finds no
match, so a question outside the event layer still gets an answer (`route`
says which path answered).

Why (docs/architecture.md, "Structured event layer"): aggregation and
superlative questions span 8-43 documents, which no top-k retrieval can
count; the other question types key on exact infobox values (venue + date,
the previous Games) that similarity search blurs. Planning is the only step
that needs language understanding, so it's the only LLM call. That also
keeps this pipeline at 1 call/question against the free-tier quota.
"""
import difflib
import hashlib
import json
import os
import re
from typing import Literal, Optional

from pydantic import BaseModel

from src.common.infobox import OLYMPIC_CALENDAR, date_key, games_id, norm_key
from src.common.llm import generate
from src.common.tg_conn import get_connection
from src.pipelines.rag import pipeline as rag_pipeline

# Re-running the benchmark while iterating shouldn't re-spend quota on plans
# that were already made. Off by default so benchmark token/call numbers
# reflect a cold run; rows record `plan_cached` either way.
PLAN_CACHE_PATH = "data/results/event_plan_cache.jsonl"
PLAN_CACHE_ENABLED = os.environ.get("EVENT_PLAN_CACHE", "false").lower() == "true"

FUZZY_CUTOFF = 0.85
MAX_EVIDENCE_EVENTS = 60

PLANNER_SYSTEM_INSTRUCTION = """You translate a question into a query plan over a graph of Olympic
events. Each event node is linked to its Games (year + Summer/Winter), its Sport and its Venue, and
carries that event's infobox fields: competitors, nations, venue, date, gold/silver/bronze.

Operations:
  - count_events_over: how many events of `sport` at the `year`/`season` Games had more than
    `min_competitors` competitors.
  - top_event_by_competitors: which event of `sport` at the `year`/`season` Games had the most
    competitors.
  - event_field: one `field` of a single event, identified by `sport`, `year`, `season`, `event_name`.
  - gold_at_venue_date: who won gold in the event held at `venue` on `date`. Copy venue and date text
    exactly as written in the question. If the question names the Games separately from the date,
    put that year/season in `year`/`season`.
  - gold_previous_games: who won gold in `event_name` of `sport` at the Games held immediately before
    `year`. `year` is the year named in the question, not the earlier Games.
  - unsupported: anything the operations above can't answer.

`sport` must be copied exactly from this list: {sports}
`event_name` is the event alone, without sport or Games, keeping any "Men's"/"Women's"/"Mixed" the
question gives, e.g. "Men's 20 kilometres walk", "Women's RS:X", "Light flyweight"."""

# Cached plans are only reusable under the prompt that produced them.
_PROMPT_VERSION = hashlib.md5(PLANNER_SYSTEM_INSTRUCTION.encode()).hexdigest()[:8]

# Near-identical names that differ in one of these are different events or venues
# ("Men's K-1 500 metres" vs "Women's K-1 500 metres", "Pavilion 4" vs "Pavilion 6"),
# so fuzzy matching must never cross them.
_DISCRIMINATOR_RE = re.compile(r"\d+|\+|\b(?:men|women|mixed)\b")


class QueryPlan(BaseModel):
    operation: Literal[
        "count_events_over", "top_event_by_competitors", "event_field",
        "gold_at_venue_date", "gold_previous_games", "unsupported",
    ]
    sport: Optional[str] = None
    year: Optional[int] = None
    season: Optional[Literal["Summer", "Winter"]] = None
    event_name: Optional[str] = None
    field: Optional[Literal["nations", "competitors", "gold", "silver", "bronze", "venue", "date_text"]] = None
    min_competitors: Optional[int] = None
    venue: Optional[str] = None
    date: Optional[str] = None


_catalog: dict = {}


def _get_catalog(conn) -> dict:
    """Sport and Venue keys from the graph, fetched once per process."""
    if not _catalog:
        _catalog["sports"] = {v["v_id"]: v["attributes"]["name"] for v in conn.getVertices("Sport")}
        _catalog["venues"] = {v["v_id"]: v["attributes"]["name"] for v in conn.getVertices("Venue")}
    return _catalog


def _fuzzy_unique(key: str, candidates) -> Optional[str]:
    """The one close candidate that agrees with `key` on numbers, `+` and gender.
    Ambiguity returns None: falling back beats a confident wrong match."""
    close = difflib.get_close_matches(key, list(candidates), n=3, cutoff=FUZZY_CUTOFF)
    close = [c for c in close if _DISCRIMINATOR_RE.findall(c) == _DISCRIMINATOR_RE.findall(key)]
    return close[0] if len(close) == 1 else None


def _snap(value: Optional[str], keys) -> Optional[str]:
    """Exact normalized key if present, else a unique safe fuzzy match."""
    k = norm_key(value)
    if not k:
        return None
    return k if k in keys else _fuzzy_unique(k, keys)


def _load_cached_plan(question: str, model: str) -> Optional[QueryPlan]:
    if not os.path.exists(PLAN_CACHE_PATH):
        return None
    with open(PLAN_CACHE_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["question"] == question and row["model"] == model and row.get("prompt") == _PROMPT_VERSION:
                return QueryPlan.model_validate(row["plan"])
    return None


def _save_plan(question: str, model: str, plan: QueryPlan):
    os.makedirs(os.path.dirname(PLAN_CACHE_PATH), exist_ok=True)
    with open(PLAN_CACHE_PATH, "a") as f:
        f.write(json.dumps({"question": question, "model": model, "prompt": _PROMPT_VERSION, "plan": plan.model_dump()}) + "\n")


def plan_question(conn, question: str, tracker=None, question_id=None) -> tuple:
    """Returns (plan, was_cached)."""
    model = os.environ.get("GEN_MODEL", "")
    if PLAN_CACHE_ENABLED and (cached := _load_cached_plan(question, model)):
        return cached, True
    sports = sorted(_get_catalog(conn)["sports"].values())
    plan, _rec = generate(
        f"Question: {question}",
        system_instruction=PLANNER_SYSTEM_INSTRUCTION.format(sports=", ".join(sports)),
        response_schema=QueryPlan,
        thinking_budget=0,  # a structured parse, not open-ended reasoning
        pipeline="event_graph",
        call_type="planner",
        tracker=tracker,
        question_id=question_id,
    )
    _save_plan(question, model, plan)
    return plan, False


def _vertices(result: list, name: str) -> list:
    for block in result:
        if name in block:
            return [v["attributes"] | {"event_id": v["v_id"]} for v in block[name]]
    return []


def _pick_event_by_name(pool: list, event_name: Optional[str], question: str = "") -> list:
    key = norm_key(event_name)
    by_key = {e["event_key"]: e for e in pool}
    if key in by_key:
        return [by_key[key]]
    # Planner dropped the gender ("pole vault" for "men's pole vault"): restore it
    # from the question itself rather than guessing, which would match both events.
    question_key = f" {norm_key(question)} "
    if not re.search(r"\b(?:men|women|mixed)\b", key):
        for gender in ("women", "men", "mixed"):
            if f" {gender} " in question_key:
                for variant in (f"{gender} s {key}", f"{gender} {key}"):
                    if variant in by_key:
                        return [by_key[variant]]
    match = _fuzzy_unique(key, by_key) if key else None
    return [by_key[match]] if match else []


def _date_tokens(s: str) -> set:
    return set(date_key(s).split())


def _match_date(events: list, date: Optional[str], year: Optional[int], season: Optional[str]) -> list:
    """Tiered: exact normalized date, then same Games with overlapping date tokens
    that share a day number, then -- only when the question gave no date -- the
    only event at this venue in those Games."""
    if year:
        in_games = [e for e in events if e["games_year"] == year and (not season or e["season"] == season)]
    else:
        in_games = events
    exact = [e for e in in_games if date and e["date_key"] == date_key(date)]
    if exact:
        return exact
    want = _date_tokens(date or "")
    want_days = {t for t in want if t.isdigit() and len(t) <= 2}

    def overlaps(e):
        # An event with no infobox date has an empty token set, which is a subset
        # of everything -- it must never count as a match.
        have = _date_tokens(e["date_text"])
        return bool(have and want_days & have and (want <= have or have <= want))

    overlap = [e for e in in_games if want and overlaps(e)]
    if overlap:
        return overlap
    return in_games if not date and year and len(in_games) == 1 else []


def _cite(e: dict) -> str:
    return f"[{e['event_id']}]"


def _evidence_line(e: dict) -> str:
    return (f"{_cite(e)} {e['title']}: competitors={e['competitors']}, nations={e['nations']}, "
            f"venue={e['venue']}, date={e['date_text']}, gold={e['gold']}")


def execute_plan(conn, plan: QueryPlan, question: str = "") -> Optional[dict]:
    """Run the plan against the event layer. None means "the graph can't answer this"."""
    catalog = _get_catalog(conn)
    if plan.season is None:
        # The planner sometimes leaves season null even when the question says
        # "Winter Olympics" (pub-049); the question text is unambiguous about it.
        named = re.findall(r"\b(Summer|Winter)\b", question)
        if len(set(named)) == 1:
            plan = plan.model_copy(update={"season": named[0]})
    sport_key = _snap(plan.sport, catalog["sports"])
    # A Games id outside the calendar isn't a vertex; querying it would raise.
    valid_games = plan.year and plan.season and plan.year in OLYMPIC_CALENDAR[plan.season]
    games = games_id(plan.year, plan.season) if valid_games else None

    if plan.operation in ("count_events_over", "top_event_by_competitors", "event_field"):
        if not (sport_key and games):
            return None
        # VERTEX<T> params go as 1-tuples; plain values make pyTigerGraph fail the
        # POST and retry over GET, doubling every query's round trips.
        params = {"games": (games,), "sport": (sport_key,)}
        if plan.operation == "count_events_over":
            if plan.min_competitors is None:
                return None
            params["min_competitors"] = plan.min_competitors
        result = conn.runInstalledQuery("events_by_games_sport", params=params)
        pool = _vertices(result, "Pool")
        if not pool:
            return None

        if plan.operation == "count_events_over":
            over = sorted(_vertices(result, "Over"), key=lambda e: -e["competitors"])
            listing = "; ".join(f"{e['title']} ({e['competitors']} competitors) {_cite(e)}" for e in over) or "none"
            return {
                "short_answer": str(len(over)),
                "answer": (f"{len(over)} of the {len(pool)} {plan.sport} events at the {games} Olympics in the corpus "
                           f"had more than {plan.min_competitors} competitors: {listing}."),
                "evidence": pool,
            }

        if plan.operation == "top_event_by_competitors":
            sized = [e for e in pool if e["competitors"] >= 0]
            if not sized:
                return None
            top = max(e["competitors"] for e in sized)
            winners = [e for e in sized if e["competitors"] == top]
            return {
                "short_answer": " | ".join(e["title"] for e in winners),
                "answer": " and ".join(f"{e['title']} {_cite(e)}" for e in winners)
                          + f" had the most competitors ({top}) of the {len(sized)} {plan.sport} events at the {games} Olympics in the corpus.",
                "evidence": pool,
            }

        if plan.field is None:
            return None
        field = plan.field
        hits = _pick_event_by_name(pool, plan.event_name, question)
        values = [str(e[field]) for e in hits if e[field] not in ("", -1)]
        if not values:
            return None
        return {
            "short_answer": " | ".join(values),
            "answer": "; ".join(f"{e['title']}: {field} = {e[field]} {_cite(e)}" for e in hits),
            "evidence": hits,
        }

    if plan.operation == "gold_at_venue_date":
        venue_key = _snap(plan.venue, catalog["venues"])
        if not venue_key:
            return None
        at_venue = _vertices(conn.runInstalledQuery("events_at_venue", params={"venue": (venue_key,)}), "AtVenue")
        hits = [e for e in _match_date(at_venue, plan.date, plan.year, plan.season) if e["gold"]]
        if not hits:
            return None
        return {
            "short_answer": " | ".join(e["gold"] for e in hits),
            "answer": "; ".join(f"{e['gold']} won gold in {e['title']} ({e['venue']}, {e['date_text']}) {_cite(e)}" for e in hits)
                      + ("" if len(hits) == 1 else f". Note: {len(hits)} events match this venue and date."),
            "evidence": hits,
        }

    if plan.operation == "gold_previous_games":
        if not (sport_key and games and plan.event_name):
            return None
        result = conn.runInstalledQuery("events_in_previous_games", params={"games": (games,), "sport": (sport_key,)})
        hits = [e for e in _pick_event_by_name(_vertices(result, "Pool"), plan.event_name, question) if e["gold"]]
        if not hits:
            return None
        return {
            "short_answer": " | ".join(e["gold"] for e in hits),
            "answer": "; ".join(f"{e['gold']} won gold in {e['title']} {_cite(e)}" for e in hits),
            "evidence": hits,
        }

    return None


def answer_question(conn, question: str, tracker=None, question_id=None) -> dict:
    plan, plan_cached = plan_question(conn, question, tracker=tracker, question_id=question_id)
    graph_result = None
    if plan.operation != "unsupported":
        try:
            graph_result = execute_plan(conn, plan, question)
        except Exception:
            # A malformed plan (e.g. a vertex id the graph doesn't have) is the
            # same outcome as "no match": answer via RAG instead of an error row.
            graph_result = None

    if graph_result is None:
        fallback = rag_pipeline.answer_question(conn, question, tracker=tracker, question_id=question_id)
        return fallback | {"route": "fallback_rag", "plan": plan.model_dump(), "plan_cached": plan_cached, "short_answer": None}

    evidence = graph_result["evidence"][:MAX_EVIDENCE_EVENTS]
    return {
        "answer": graph_result["answer"],
        "short_answer": graph_result["short_answer"],
        "route": "event_graph",
        "plan": plan.model_dump(),
        "plan_cached": plan_cached,
        "retrieved_doc_ids": sorted({e["event_id"] for e in graph_result["evidence"]}),
        "context_text": "\n".join(_evidence_line(e) for e in evidence),
    }


if __name__ == "__main__":
    conn = get_connection()
    for q in [
        "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
        "Who won the gold medal in the men's 80 kg taekwondo event at the Summer Olympics held immediately before 2016?",
    ]:
        r = answer_question(conn, q)
        print(r["route"], "|", r["short_answer"], "|", r["answer"][:300])
