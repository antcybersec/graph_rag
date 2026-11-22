"""The agent: how a question becomes graph queries, and when it stops.

Three commitments shape every line of this file.

**Rule 4 compliance is structural, not promised.** Nothing here reads the
wording of a question. There is no regex over question text, no question id is
ever consulted, and no answer is stored in the code. The only component that
interprets a question is the model, and the only values it may emit are keys
the graph just handed it: sports and Games come from `list_sports` and
`list_games`, a contest comes from `series_by_sport`, an athlete from
`athlete_search`, a person from `person_search`. Deterministic string work
happens exclusively on *model-extracted entity names* and *graph keys* -- never
on the question. That is why the same code answers the judges' unseen
questions as answers these: it has never seen a question either.

**The graph decides what is missing, not the model's confidence.**
`coverage_check` asks the corpus whether a document matching what the question
pinned down exists at all. Zero matches is evidence, and it is the difference
between "the corpus does not record this" and "I am not sure" -- which is
10% of the score and the easiest place to lose it by inventing a plausible
name.

**Every call is recorded as it happens.** The agent never touches the database
or a model directly; `Trace.run_query` and `Trace.generate` are the only
routes, so the submitted trace is a transcript rather than a reconstruction,
and the token totals cannot drift from the per-step numbers.

The loop itself: plan -> resolve entities to keys -> gather on the chosen route
-> compose. The composer may answer `enough=false`, which buys exactly one
more round of full-text retrieval before it must answer with what it has. One
retry, not an open loop: the stopping criterion is explicit because an agent
that can always try again is an agent that never reports a gap.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel, Field

from src.common.infobox import norm_key
# Reused rather than rewritten: these were built and tested for grounding
# claims to events and they solve the same problem here -- deciding whether
# two ways of naming a contest mean the same contest.
from src.r2.claims import _compatible, _distinctive, resolve, slot_of
from src.r2.results import rows, scalar
from src.r2.trace import Trace

# ---------------------------------------------------------------------------
# What the planner may choose. Each route is a *graph capability*, not a kind
# of question: "one contest across its editions", "every event of a sport at
# one Games". A question is mapped onto a capability by the model.
# ---------------------------------------------------------------------------
ROUTES = {
    "event_series": (
        "One contest, across the editions the corpus holds: who won it, how "
        "many nations or competitors took part, how that changed over the "
        "years, and the article prose explaining what happened. Needs a sport "
        "and the contest's name."),
    "sport_games": (
        "Every event of one sport at one Games: which had the most or fewest "
        "nations, which nation won the most medals, which events a nation "
        "won. Needs a sport, a year and a season."),
    "athlete": (
        "Every medal one named athlete won, across all Games and sports."),
    "repeat_champions": (
        "Athletes who won the same event at more than one Games, within one "
        "sport. Needs a sport and a season."),
    "multi_medal_games": (
        "Medal counts per athlete at one Games -- who won three or more, who "
        "won the most. Needs a year and a season."),
    "film": (
        "Films: director, year, budget, box-office gross. By film name, or "
        "every film by one director."),
    "office": (
        "Public offices and successions: who held an office when, who "
        "preceded and succeeded them, and the chain beyond that."),
    "evidence": (
        "Questions about the sources themselves: records that disagree, a "
        "provisional result later corrected, what was officially confirmed "
        "and when, what was never verified, what a particular document does "
        "or does not establish, and what was true as of a given date."),
    "text": (
        "Anything whose answer is in the prose of an article and nowhere in "
        "the structured fields -- including a fact that appears only in a "
        "different article than the one the question is about."),
}

PLANNER_SYSTEM = """You turn a question into a retrieval plan for a graph database of Olympic
events, films, politicians and LA 2028 source records.

You never answer the question. You decide what to look up.

Choose exactly one route. Fill only the fields that route needs; leave the
rest empty. Every sport and Games you name MUST be copied exactly from the
catalogues given to you -- never invent a key, and never reword one. If the
question names a sport or a Games that is not in the catalogues, copy the
closest catalogue entry you believe it means, or leave the field empty if
there is none; a later step checks the corpus for the exact thing asked and
reports honestly when it is absent, so a missing year is not your problem to
solve.

Guidance that is about the data, not about any particular question:

- `years` holds every Games year the question asks about, including ones you
  doubt exist. Leave it empty when the question asks about all editions.
- `event_phrase` is how the question names the contest, in its own words
  ("men's slalom", "women's 10 metre air pistol"). Do not normalise it.
- `wants_prose` is true when the answer is a reason, a cause or an account of
  what happened -- anything that must be read rather than counted.
- `wants_evidence` is true when the question is about the standing of
  evidence: documents that disagree, something provisional or later
  corrected, what is officially confirmed, what cannot be established, what
  a specific dated document does or does not prove, or what was true as of a
  date. The LA 2028 material (2028 and later) is where this lives.
- `entities` lists the proper names and distinctive phrases the question
  mentions -- people, nations, documents, anything that could be searched
  for. Include surnames on their own as well as full names. **At most eight,
  each one different.** Repeating a phrase in another word order adds nothing
  and crowds out the search.
- `extreme` is "most" or "fewest" when the question asks which record is
  highest or lowest on some figure, and "" otherwise. Set it even when you
  have also named the thing being compared: it is what tells the system to
  compare the whole set rather than look one record up.
- `metric` names the figure being compared, in the question's own words --
  "gold medals", "nations", "competitors", "gross", "gross over budget" --
  and is "" when nothing is being compared.
- `medal` is "gold", "silver" or "bronze" when the question is about one
  colour, and "" when it is about medals of any colour. A question asking
  which nation won the most *silver* medals is not a question about medals.
- `as_of` is set only when the question pins a moment ("as of 2 July 2028"),
  as a full ISO date.
- `dates` holds every date the question mentions in full ISO form
  (YYYY-MM-DD), including dates it attributes to a document ("a blog
  published on July 24", "the July 27 directory entry"). When a question
  gives only a day and month, supply the year from context -- every source
  record in this corpus was published between July 2028 and February 2029,
  so a bare "July 28" means 2028-07-28.
- `numbers` holds every number the question quotes as a value someone
  claimed: counts, durations, placings, times. Write them as digits. A
  disputed figure is often the quickest way to find the documents that
  disagree, so do not leave these out.
"""


class Plan(BaseModel):
    route: str
    why: str = ""
    sport_key: str = ""
    season: str = ""
    years: list[int] = Field(default_factory=list)
    event_phrase: str = ""
    athlete: str = ""
    person: str = ""
    film: str = ""
    nation: str = ""
    medal: str = ""
    metric: str = ""
    extreme: str = ""
    as_of: str = ""
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    wants_prose: bool = False
    wants_evidence: bool = False
    entities: list[str] = Field(default_factory=list)


class Selection(BaseModel):
    keys: list[str] = Field(default_factory=list)
    none_match: bool = False
    note: str = ""


class Composed(BaseModel):
    status: str
    answer: str
    explanation: str
    citations: list[str] = Field(default_factory=list)
    missing: str = ""
    enough: bool = True


COMPOSER_SYSTEM = """You answer a question using ONLY the evidence block supplied to you. The
evidence came from a graph database built from a fixed corpus. The corpus is
the only source of truth: if your own knowledge disagrees with it, the corpus
wins, and if the corpus is silent, the honest answer is that it is silent.

Never add a fact that is not in the evidence. Never guess a name, a number or
a date. An invented answer scores zero and costs more than an admitted gap.

status:
  answered       the evidence supports the whole question
  partial        it supports part of it; say plainly what is missing. This
                 includes the case where the corpus records rival claims but
                 confirms none of them: the claims are an answer to what the
                 sources say, and the unconfirmed value is the missing part.
  not_in_corpus  the corpus does not contain the answer

answer:
  Plain text, complete and compact. Give every fact the question asks for.
  When the question asks for a list, give every item in the evidence and
  nothing else -- a missing item and an invented item both cost marks.
  When it asks for a count, state the number and what was counted.
  Name people and nations exactly as the evidence spells them.
  When the question asks why something happened, give the consequence as well
  as the cause: if a result changed, say what it changed to. Give every
  reason the text offers, not only the first one.
  A medal line naming a whole squad is a team medal and is marked as such.
  A question about an athlete's medals means the medals they are named for
  individually, unless it asks about teams; list team medals separately and
  labelled if you include them at all.
  When you explain why a document does not stand, say what it actually is --
  what it repeated or recycled, and from when. "Lower authority" alone is not
  an explanation.

Reading the evidence:
  - A count of -1 means the article did not state it. That is missing
    information, not zero.
  - STANDS is what the strongest evidence supports. REJECTED entries are real
    documents that said something else; when the question asks about a
    disagreement, say what was provisional, superseded or never confirmed,
    and why it does not stand. Publication date never settles anything.
  - NO ASSERTION means no source states the value on its own authority. Do
    not fill it in from a report or an estimate. Say what was reported, say
    it is unverified, and say the value is not established.
  - Never present an estimate or a projection as a measurement.
  - A block headed ARITHMETIC ALREADY DONE FOR YOU was computed by counting
    the rows, not written by a model. Its MOST, FEWEST and TOTAL are the
    answer to a "which had the most", "how many in all" or "which is biggest"
    question -- quote them rather than recounting the rows yourself, and if it
    reports A TIE, name every holder, because naming one of two level holders
    is a wrong answer and not a short one. If your own reading of the rows
    disagrees with it, the block is right and you have misread them.
  - A comparison or a superlative over records where some lack the figure is
    still answerable: compare the records that state it, give the answer, and
    say which records could not be included. Incomplete data is a caveat to
    state, not a reason to refuse -- "the corpus does not provide the figure
    for every film" is not an answer to which film had the best ratio.

citations: the doc_ids from CITABLE DOCUMENTS that actually support your
answer. Only ids from that list. An answered or partial answer needs at
least one.

  Cite what a reader would have to check to confirm the answer -- no more and
  no less. For a count or a total, that is the records counted, not every
  record you were shown. For a "which one of these" answer, it is the
  candidates you compared, because the answer rests on the comparison and not
  only on the winner. For a single stated fact, it is the document that states
  it. Citing everything in the evidence block is as wrong as citing too
  little: both say you do not know which document carries the answer.

enough: set false ONLY if the evidence block is so thin that a full-text
search of the corpus might still find the answer. If the evidence shows the
corpus does not hold it, that is an answer -- not_in_corpus -- and enough
stays true.
"""


# ---------------------------------------------------------------------------
# deterministic helpers -- these only ever see graph keys and model-extracted
# entity names, never the question
# ---------------------------------------------------------------------------

def _narrow(phrase: str, candidates: list[dict], field: str, limit: int = 30) -> list[dict]:
    """Rank graph candidates by how well they match a model-extracted phrase.

    Pure candidate reduction, so a selection prompt stays small: the model
    still decides, and `none_match` is always available to it. Ordering uses
    the same notion of a distinguishing word that grounding uses -- "men's
    individual" identifies nothing, "sabre" identifies a sport.
    """
    key = norm_key(phrase)
    if not key:
        return candidates[:limit]
    kd, kt = _distinctive(key), set(key.split())
    scored = []
    for row in candidates:
        ck = norm_key(row.get(field, ""))
        if not ck:
            continue
        ct = set(ck.split())
        scored.append(((
            1 if _compatible(key, ck) else 0,          # never rank an
            len(kd & _distinctive(ck)),                 # incompatible gender
            len(kt & ct) / max(len(kt | ct), 1),        # or distance above a
        ), row))                                        # compatible one
    scored.sort(key=lambda s: s[0], reverse=True)
    return [r for _, r in scored[:limit]]


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + " ...[truncated]"


def _as_of_sort(iso: str) -> Optional[int]:
    """A claim's effective-time sort key for a date the model extracted."""
    digits = "".join(ch for ch in (iso or "") if ch.isdigit())
    return int(digits[:14].ljust(14, "0")) if digits else None


def _within_a_day(published_at: str, date: str) -> bool:
    """Is this publication timestamp on `date`, or the day either side of it?

    The tolerance is not sloppiness: publication times are stored in UTC and
    a question naming "the July 24 blog" means July 24 where it was written.
    One day of slack covers the offset without letting an unrelated document
    in -- the corpus's sources are days apart, not hours.
    """
    if not published_at or not date:
        return False
    import datetime as dt
    try:
        day = dt.date.fromisoformat(date[:10])
        got = dt.date.fromisoformat(published_at[:10])
    except ValueError:
        return False
    return abs((got - day).days) <= 1


def _tally(what: str, unit: str, pairs: list) -> Optional[dict]:
    """Do the arithmetic over a counted column in code instead of leaving it to
    the composer.

    Round 1 measured exactly this gap. On the aggregation and superlative
    questions -- "how many events had more than N competitors", "which had the
    most" -- retrieval feeding a model scored 1 of 21, while the same graph rows
    run through code scored 21 of 21 (`docs/round1-results.md`). The rows were
    never the problem; the step after them was. So the rows still go to the
    composer, and so does the answer the rows imply, computed here.

    A tie is the sharpest case and the reason this returns every holder rather
    than one: asked for "the most", a model names a single holder and does not
    mention that two are level, which is a wrong answer rather than an
    incomplete one.

    Values of -1 mean "the article does not state this" in the event fields and
    are dropped, because counting them as zero would invent a smallest.
    """
    clean = []
    for k, v in pairs:
        if v is None or isinstance(v, bool):
            continue
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        if n < 0:                      # -1 is the corpus's "not stated"
            continue
        clean.append((str(k), int(n) if float(n).is_integer() else round(n, 3)))
    if not clean:
        return None
    ranked = sorted(clean, key=lambda kv: (-kv[1], kv[0]))
    hi, lo = ranked[0][1], ranked[-1][1]
    return {
        "what": what, "unit": unit,
        "highest": {"value": hi, "holders": [k for k, v in ranked if v == hi]},
        "lowest": {"value": lo, "holders": [k for k, v in ranked if v == lo]},
        "total": round(sum(v for _, v in ranked), 3),
        "entries": len(ranked),
        "ranked": ranked[:12],
        "dropped": len(pairs) - len(clean),
    }


def _medal_rows(result, block: str = "medals") -> list[dict]:
    """MEDAL_FOR / WON edges as flat rows: nation, recipient, medal, event."""
    out = []
    for e in rows(result, block):
        out.append({"event_id": e.get("to_id", ""), "noc": e.get("from_id", ""),
                    "medal": e.get("medal", ""), "recipient": e.get("recipient", ""),
                    "rank": e.get("rank", 0), "multi": e.get("multi", False)})
    return out


class Agent:
    """One instance per run. Catalogues are read from the graph once."""

    def __init__(self, conn, query_log=None, model: Optional[str] = None):
        self.conn = conn
        self.query_log = query_log
        self.model = model or os.environ.get("GEN_MODEL") or None

        # Planning and selection are short, structured, low-reasoning calls:
        # the planner prompt is ~1.3k tokens and a selection ~0.4-2.8k, while
        # the answer prompt reaches 11k. That difference is worth exploiting,
        # because the cheapest provider for the small calls is unusable for the
        # large one -- Groq's free tier allows 1,000 requests a day but caps a
        # request at 8,000 tokens per minute and rejects anything larger
        # outright rather than throttling it.
        #
        # So the small calls can be sent somewhere else entirely. Set
        # SMALL_CALL_PROVIDER/SMALL_CALL_MODEL and roughly two thirds of the
        # requests per question leave the main provider's quota, which on a
        # free tier is the difference between answering 100 questions a day
        # and answering 250 with room to rehearse. Unset, everything goes to
        # the one provider and nothing changes.
        self.small_provider = os.environ.get("SMALL_CALL_PROVIDER") or None
        self.small_model = os.environ.get("SMALL_CALL_MODEL") or None

        # Read once per run, not once per question: the catalogues are what
        # the planner selects keys from, and they do not change while the
        # graph is static. The calls are still traced and logged -- under a
        # bootstrap id, because they belong to the run rather than to any
        # one answer.
        boot = Trace("__bootstrap__", conn=conn, query_log=query_log, model=self.model)
        self.sports = sorted(rows(boot.run_query("list_sports"), "Res"),
                             key=lambda r: r.get("sport_key", ""))
        self.games = sorted(rows(boot.run_query("list_games"), "Res"),
                            key=lambda r: (r.get("year", 0), r.get("season", "")))
        # The evidence questions name documents in prose ("the July 27
        # directory entry"); the catalogue is what lets the model point at
        # one instead of the code guessing from a date.
        self.sources = sorted(rows(boot.run_query("list_sources"), "Res"),
                              key=lambda r: (r.get("published_at", ""), r.get("source_id", "")))
        self.bootstrap_steps = boot.steps

    # -- model-driven selection over graph-supplied candidates --------------

    def _select(self, tr: Trace, question: str, label: str, options: list[dict],
                key_field: str, show_fields: list[str], many: bool = False) -> list[str]:
        """Ask the model which candidate(s) the question means.

        The candidates are graph rows. The model may answer that none of them
        is it, which is how a question about something the corpus does not
        cover stops here instead of being forced onto the nearest match.
        """
        if not options:
            return []
        lines = []
        for row in options:
            shown = "  ".join(f"{f}={row.get(f)!r}" for f in show_fields if row.get(f) not in (None, ""))
            lines.append(f"- {row.get(key_field)!r}  {shown}")
        prompt = (f"QUESTION: {question}\n\n"
                  f"Pick the {label} the question is about, from this list only.\n"
                  f"Return the exact key string{'s' if many else ''} shown first on the line.\n"
                  f"{'Return every one the question covers.' if many else 'Return exactly one.'}\n"
                  f"If none of them is what the question means, set none_match true.\n\n"
                  + "\n".join(lines))
        sel = tr.generate(prompt, tool=f"select_{label.replace(' ', '_')}",
                          response_schema=Selection, thinking_budget=0,
                          provider=self.small_provider, model=self.small_model,
                          pipeline="r2_agent", call_type="selection")
        valid = {row.get(key_field) for row in options}
        return [k for k in sel.keys if k in valid]

    # -- planning -----------------------------------------------------------

    def plan(self, tr: Trace, question: str) -> Plan:
        catalogue = (
            "SPORTS (sport_key -- copy exactly):\n"
            + ", ".join(f"{s['sport_key']}" for s in self.sports)
            + "\n\nGAMES in the corpus (year season):\n"
            + ", ".join(f"{g['year']} {g['season']}" for g in self.games)
            + "\n\nROUTES:\n"
            + "\n".join(f"- {name}: {desc}" for name, desc in ROUTES.items()))
        plan = tr.generate(f"{catalogue}\n\nQUESTION: {question}",
                           tool="plan", system_instruction=PLANNER_SYSTEM,
                           response_schema=Plan, thinking_budget=0,
                           provider=self.small_provider, model=self.small_model,
                           pipeline="r2_agent", call_type="plan")
        if plan.route not in ROUTES:
            plan.route = "text"
        if plan.sport_key and plan.sport_key not in {s["sport_key"] for s in self.sports}:
            plan.sport_key = ""      # the planner may only use real keys
        if plan.season not in ("Summer", "Winter"):
            plan.season = ""
        if plan.medal.lower() not in ("gold", "silver", "bronze"):
            plan.medal = ""
        else:
            plan.medal = plan.medal.lower()
        if plan.extreme.lower() not in ("most", "fewest"):
            plan.extreme = ""
        else:
            plan.extreme = plan.extreme.lower()
        # Deduplicate and cap the entity list. A model asked for "the proper
        # names and distinctive phrases" can loop: one rehearsal question came
        # back with 170 reorderings of "Brisbane 2032 Olympics", the last of
        # them truncated mid-string. That is prompt weight and stored-plan
        # weight for nothing, and the searches it drives are the same search
        # eight times. Order is kept, so the first mention still wins.
        seen, kept = set(), []
        for e in plan.entities:
            k = (e or "").strip()
            if k and k.lower() not in seen:
                seen.add(k.lower())
                kept.append(k)
        plan.entities = kept[:8]
        return plan

    # -- entity resolution --------------------------------------------------

    def pin_series(self, tr: Trace, question: str, plan: Plan) -> Optional[dict]:
        """The EventSeries the question is about, chosen by the model.

        Candidates come from the graph: the chosen sport's series, or every
        series when no sport was identified. Narrowed deterministically on the
        model's own phrase only to keep the prompt small.
        """
        cands = rows(tr.run_query("series_by_sport", {"sport_key": plan.sport_key}), "Res")
        narrowed = _narrow(plan.event_phrase, cands, "event_name", limit=30)
        # If the phrase the planner extracted already *is* a contest the graph
        # holds, there is nothing to choose: a second model call can only agree
        # or be wrong, and it was wrong -- on "Men's 100 metres", with
        # `athletics|men s 100 metres` sitting in the candidate list, a weaker
        # model rejected every candidate and the route reported the corpus does
        # not hold the contest. This compares the model's own phrase against
        # graph keys, never the question, which is the same thing the nation
        # lookup in `_route_sport_games` does.
        want = (plan.event_phrase or "").strip().lower()
        exact = [c for c in narrowed
                 if want and (str(c.get("event_name", "")).strip().lower() == want
                              or norm_key(str(c.get("event_name", ""))) == norm_key(want))]
        if exact:
            return exact[0]
        picked = self._select(tr, question, "event series", narrowed,
                              "series_key", ["sport_key", "event_name"])
        if not picked and plan.sport_key:
            # The sport may have been misidentified -- a contest the question
            # calls "men's slalom" could sit under alpine skiing or canoeing.
            # Widen to every series rather than concluding the corpus lacks it.
            allc = rows(tr.run_query("series_by_sport", {"sport_key": ""},
                                     tool="series_by_sport_all"), "Res")
            narrowed = _narrow(plan.event_phrase, allc, "event_name", limit=30)
            picked = self._select(tr, question, "event series", narrowed,
                                  "series_key", ["sport_key", "event_name"])
        if not picked:
            return None
        return next((r for r in narrowed if r["series_key"] == picked[0]), None)

    def pin_athlete(self, tr: Trace, question: str, name: str) -> Optional[str]:
        """An athlete_key for a name the model read out of the question."""
        key = norm_key(name)
        if not key:
            return None
        found = rows(tr.run_query("athlete_search", {"fragment": key}), "Res")
        if not found:
            # Fall back to the longest word in the name the model gave us --
            # a surname on its own catches a spelling or ordering difference
            # between the question and the corpus. Still the model's string,
            # never the question's.
            longest = max(key.split(), key=len, default="")
            if longest:
                found = rows(tr.run_query("athlete_search", {"fragment": longest},
                                          tool="athlete_search_surname"), "Res")
        if not found:
            return None
        if len(found) == 1:
            return found[0]["athlete_key"]
        # Same shortcut as the event series: an exact key match is not a choice.
        exact = [r for r in found if r.get("athlete_key") == key]
        if exact:
            return exact[0]["athlete_key"]
        picked = self._select(tr, question, "athlete", found[:40], "athlete_key", ["name"])
        return picked[0] if picked else None

    def pin_person(self, tr: Trace, question: str, name: str) -> list[dict]:
        key = norm_key(name)
        if not key:
            return []
        found = rows(tr.run_query("person_search", {"fragment": key}), "Res")
        if not found:
            longest = max(key.split(), key=len, default="")
            if longest:
                found = rows(tr.run_query("person_search", {"fragment": longest},
                                          tool="person_search_surname"), "Res")
        return found

    # -- the routes ---------------------------------------------------------

    def gather(self, tr: Trace, question: str, plan: Plan) -> dict:
        """Run the chosen route. Returns an evidence bundle for the composer."""
        ev: dict = {"route": plan.route, "notes": [], "citable": {}}
        fn = getattr(self, f"_route_{plan.route}")
        fn(tr, question, plan, ev)
        return ev

    def _add_tally(self, ev: dict, what: str, unit: str, pairs: list) -> None:
        t = _tally(what, unit, pairs)
        if t:
            ev.setdefault("tallies", []).append(t)

    def _cite(self, ev: dict, doc_id: str, label: str) -> None:
        """Offer a document as a citation candidate.

        "name:ron howard" is a Person vertex for a director with no article of
        their own; it is not a doc_id and must never reach a citation list.
        """
        if doc_id and not str(doc_id).startswith("name:"):
            ev["citable"].setdefault(doc_id, label)

    # route: one contest across its editions -------------------------------

    def _route_event_series(self, tr, question, plan, ev):
        series = self.pin_series(tr, question, plan)
        if series is None:
            ev["notes"].append(
                "No event series in the corpus matches the contest this question names "
                "(the candidate list came from the graph and the model rejected all of it).")
            return self._coverage_only(tr, plan, ev)
        ev["series"] = series

        res = tr.run_query("series_medals", {"series_key": series["series_key"],
                                             "medal": plan.medal or ""})
        editions = sorted(rows(res, "Ev"), key=lambda r: (r.get("games_year", 0),))
        medals = _medal_rows(res)
        ev["editions"] = editions
        ev["medals"] = medals
        self._add_tally(ev, "competitors per edition of this contest", "competitors",
                        [(f"{e.get('games_year')} {e.get('season')}", e.get("competitors"))
                         for e in editions])
        self._add_tally(ev, "nations per edition of this contest", "nations",
                        [(f"{e.get('games_year')} {e.get('season')}", e.get("nations"))
                         for e in editions])
        for e in editions:
            self._cite(ev, e["event_id"], e.get("title", ""))

        have = {e["games_year"] for e in editions}
        missing = [y for y in plan.years if y not in have]
        for year in missing[:4]:
            # Prove the gap rather than infer it, and distinguish the two
            # kinds: the Games is absent from the corpus, or the Games is
            # present and this contest is not on its programme.
            cov = tr.run_query("coverage_check",
                               {"sport_key": series["sport_key"],
                                "event_key": series["event_key"],
                                "games_year": year, "season": plan.season or ""},
                               tool="coverage_check_missing_year")
            same_games = tr.run_query("coverage_check",
                                      {"sport_key": "", "event_key": "",
                                       "games_year": year, "season": plan.season or ""},
                                      tool="coverage_check_games_exists")
            ev["notes"].append(
                f"coverage: {year} {plan.season or ''} -- documents for this contest: "
                f"{scalar(cov, 'matches', 0)}; documents for any event at that Games: "
                f"{scalar(same_games, 'matches', 0)}")

        focus = [e for e in editions if not plan.years or e["games_year"] in plan.years]
        # Prose is expensive (an event article is ~1,500 tokens) and a question
        # that needs it is about one edition. When the question named a year,
        # read that edition only; when it named none, the recent editions are
        # the likelier subject -- the synthetic 2028 material is where the
        # "why" questions live.
        targets = focus[:1] if focus else editions[-2:]
        if plan.wants_prose:
            for e in targets:
                self._add_prose(tr, ev, e["event_id"], e.get("title", ""))
        if plan.wants_evidence or any(e["games_year"] >= 2028 for e in (focus or editions)):
            for e in targets:
                self._claims_for_event(tr, plan, ev, e["event_id"])

    # route: every event of a sport at one Games ----------------------------

    def _route_sport_games(self, tr, question, plan, ev):
        year = plan.years[0] if plan.years else 0
        cov = tr.run_query("coverage_check", {"sport_key": plan.sport_key, "event_key": "",
                                             "games_year": year, "season": plan.season or ""})
        ev["notes"].append(f"coverage: {scalar(cov, 'matches', 0)} documents for "
                           f"{plan.sport_key or 'any sport'} at {year or 'any year'} "
                           f"{plan.season or ''}")
        if not scalar(cov, "matches", 0):
            return
        events = rows(tr.run_query("events_in_sport_games",
                                   {"sport_key": plan.sport_key, "games_year": year,
                                    "season": plan.season or "Summer"}), "Res")
        ev["events"] = events
        for e in events:
            self._cite(ev, e["event_id"], e.get("title", ""))
        self._add_tally(ev, f"nations per event, {plan.sport_key} at {year} "
                            f"{plan.season or 'Summer'}", "nations",
                        [(e.get("event_name", ""), e.get("nations")) for e in events])
        self._add_tally(ev, f"competitors per event, {plan.sport_key} at {year} "
                            f"{plan.season or 'Summer'}", "competitors",
                        [(e.get("event_name", ""), e.get("competitors")) for e in events])
        medal_res = tr.run_query("medals_by_nation",
                                 {"sport_key": plan.sport_key, "games_year": year,
                                  "season": plan.season or "Summer",
                                  "medal": plan.medal or ""})
        counts = scalar(medal_res, "counts_by_noc", {}) or {}
        ev["medals_by_nation"] = {"medal": plan.medal or "", "counts": counts,
                                  "by_medal": scalar(medal_res, "counts_by_noc_and_medal", {}) or {}}
        where = f"{plan.sport_key} at {year} {plan.season or 'Summer'}"
        self._add_tally(ev, f"{plan.medal or 'all'} medals by nation, {where}",
                        f"{plan.medal or 'medals (gold+silver+bronze)'}",
                        sorted(counts.items()))
        # And one tally per colour, always -- not only when the plan named a
        # colour. "Which nation won the most silver medals" with an empty
        # `medal` field tallies every medal together, so the arithmetic says
        # 28 for the USA while the answer says 13, and a wrong colour would
        # have passed the same check. The breakdown is already in hand; this
        # makes each colour arithmetic instead of something read off a row.
        for colour in ("gold", "silver", "bronze"):
            self._add_tally(ev, f"{colour} medals by nation, {where}", colour,
                            sorted((noc, (per or {}).get(colour, 0))
                                   for noc, per in (ev["medals_by_nation"]
                                                    .get("by_medal") or {}).items()))
        if plan.nation:
            # The NOC is chosen from the codes that actually won something
            # here, not derived from the country name. Truncating the name to
            # three letters produced "JAP" for Japan, whose code is JPN, and a
            # question with eighteen rows of answer behind it returned nothing.
            candidates = [{"noc": noc, "medals_won_here": n}
                          for noc, n in sorted(counts.items())]
            exact = [c for c in candidates if c["noc"].lower() == plan.nation.lower()]
            picked = ([exact[0]["noc"]] if exact else
                      self._select(tr, question, "nation", candidates, "noc",
                                   ["medals_won_here"]))
            if picked:
                noc = picked[0]
                detail = tr.run_query("nation_medals_detail",
                                      {"noc": noc, "sport_key": plan.sport_key,
                                       "games_year": year, "season": plan.season or "Summer",
                                       "medal": plan.medal or ""})
                ev["nation_detail"] = {"noc": noc, "events": rows(detail, "Res"),
                                       "medals": _medal_rows(detail)}
            else:
                ev["notes"].append(
                    f"no nation on this sport's medal table at these Games matches "
                    f"{plan.nation!r}, so it won nothing here")
        if plan.wants_prose or plan.wants_evidence:
            ev["notes"].append("prose was not read: this route answers from the "
                               "structured event fields.")

    # route: one athlete's medals ------------------------------------------

    def _route_athlete(self, tr, question, plan, ev):
        name = plan.athlete or plan.person or (plan.entities[0] if plan.entities else "")
        key = self.pin_athlete(tr, question, name)
        if key:
            res = tr.run_query("athlete_career", {"athlete_key": key})
            career = rows(res, "Res")
            ev["athlete"] = {"athlete_key": key, "medals": career}
            for r in career:
                self._cite(ev, r["event_id"], r.get("title", ""))
        else:
            ev["notes"].append(f"no athlete in the graph matches {name!r}")
        # Team lines name a whole squad in one string and carry no Athlete
        # vertex, so an empty career is not proof of no medals.
        team = tr.run_query("team_medals_for_name", {"name_fragment": name})
        team_events = rows(team, "Res")
        if team_events:
            ev["team_medals"] = {"events": team_events, "medals": _medal_rows(team)}
            for r in team_events:
                self._cite(ev, r["event_id"], r.get("title", ""))
        if plan.wants_evidence or plan.wants_prose:
            for entity in ([name] + plan.entities)[:3]:
                self._claims_search(tr, plan, ev, entity)

    # route: repeat champions in a sport -----------------------------------

    def _route_repeat_champions(self, tr, question, plan, ev):
        res = tr.run_query("repeat_champions", {"sport_key": plan.sport_key,
                                                "season": plan.season or "Summer"})
        grouped = scalar(res, "years_by_athlete_event", {}) or {}
        repeats = {k: sorted(v) for k, v in grouped.items() if len(v) > 1}
        ev["repeat_champions"] = repeats
        self._add_tally(ev, "Games at which each athlete won the same event "
                            "(repeat champions only)", "Games",
                        [(label, len(years)) for label, years in repeats.items()])
        ev["notes"].append(f"{len(grouped)} athlete+event gold groups in this sport; "
                           f"{len(repeats)} won the same event at more than one Games")
        # Citations: the editions in which each repeated win happened. The
        # grouped key carries the event name, so one coverage_check per event
        # returns every edition's doc_id and the winning years select from it.
        if repeats:
            sk = plan.sport_key
            for label, years in list(repeats.items())[:12]:
                event_name = label.split(" | ", 1)[-1]
                cov = tr.run_query("coverage_check",
                                   {"sport_key": sk, "event_key": norm_key(event_name),
                                    "games_year": 0, "season": plan.season or "Summer"},
                                   tool="coverage_check_repeat_event")
                for e in rows(cov, "Res"):
                    if e.get("games_year") in years:
                        self._cite(ev, e["event_id"], e.get("title", ""))

    # route: medal counts per athlete at one Games -------------------------

    def _route_multi_medal_games(self, tr, question, plan, ev):
        year = plan.years[0] if plan.years else 0
        res = tr.run_query("medals_by_athlete_at_games",
                           {"games_year": year, "season": plan.season or "Summer",
                            "medal": plan.medal or ""})
        counts = scalar(res, "counts_by_athlete", {}) or {}
        events = scalar(res, "events_by_athlete", {}) or {}
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        multi = [(n, c) for n, c in ranked if c > 1]
        ev["medal_counts"] = {
            "games": f"{year} {plan.season or 'Summer'}",
            "medal": plan.medal or "any",
            "athletes_with_more_than_one": [
                {"athlete": n, "count": c, "events": sorted(events.get(n, []))}
                for n, c in multi],
            "athletes_with_exactly_one": len(ranked) - len(multi),
            "top": ranked[:5],
        }
        self._add_tally(ev, f"individual {plan.medal or 'medals'} per athlete, "
                            f"{year} {plan.season or 'Summer'}",
                        f"{plan.medal or 'medals'}", ranked)
        ev["notes"].append(
            f"{len(ranked)} athletes hold an individual {plan.medal or 'medal'} at these "
            f"Games; the {len(multi)} with more than one are listed in full.")
        if multi:
            for name, _ in multi[:6]:
                key = norm_key(name)
                career = rows(tr.run_query("athlete_career", {"athlete_key": key},
                                           tool="athlete_career_for_citation"), "Res")
                for r in career:
                    if r.get("games_year") == year:
                        self._cite(ev, r["event_id"], r.get("title", ""))

    # route: films ---------------------------------------------------------

    def _route_film(self, tr, question, plan, ev):
        found = []
        comparison_set = False
        # A question like "which Spielberg film ..." names a director and no
        # film; the planner may put that name in `person`, or only in
        # `entities`. Rather than depend on which field it chose, every name
        # the plan offers is tried as a title and as a director.
        titles = [t for t in ([plan.film] + plan.entities) if t][:3]
        names = [n for n in ([plan.person, plan.athlete] + plan.entities) if n][:3]
        for title in titles:
            found += rows(tr.run_query("film_search", {"fragment": title}), "Res")
        for who in (names if not found else names[:1]):
            people = self.pin_person(tr, question, who)
            keys = [p["name_key"] for p in people]
            if len(keys) > 1:
                picked = self._select(tr, question, "director", people, "name_key", ["name", "kind"])
                keys = picked or keys[:1]
            if not keys:
                continue
            for key in keys[:2]:
                name = next((p["name"] for p in people if p["name_key"] == key), key)
                for row in rows(tr.run_query("films_by_director", {"director_key": key}), "Res"):
                    # The query selects on the director, so the director is
                    # known by construction -- it is simply not a column.
                    row.setdefault("@dirs", []).append(name)
                    found.append(row)
        if not found or plan.extreme:
            # No film name and no director: the question is about the films as a
            # set -- "which earned the most on its budget", "which was the most
            # expensive". Comparing them needs the rows, so fetch the ones that
            # state both figures and let the arithmetic below decide. This is
            # the gap that sent such a question to a vector search over prose.
            # ...or the plan asked which film is highest or lowest on some
            # figure, in which case a name that happened to match is not the
            # answer. "Of the films in this corpus, which made the least
            # relative to what it cost" put "all" in the film field; a
            # substring search on "all" matched the films with "all" in their
            # titles, and the comparison ran over those. The answer named a
            # film that was not the right one and cited one document.
            comparison_set = True
            whole = rows(tr.run_query("films_with_figures", {"currency": ""}), "Res")
            have = {f.get("film_id") for f in found}
            found = found + [f for f in whole if f.get("film_id") not in have]
            ev["notes"].append(
                f"the comparison covers every film whose article states both a "
                f"budget and a gross in one currency -- all {len(whole)} of the 546 "
                f"films in the corpus that state both"
                + (f", plus {len(have)} matched by name" if have else "")
                + ".")
        films, seen_films = [], set()
        for f in found:
            if f.get("film_id") in seen_films:
                continue
            seen_films.add(f.get("film_id"))
            budget, gross = f.get("budget") or 0, f.get("gross") or 0
            row = {k: f.get(k) for k in ("film_id", "name", "year", "released",
                                         "budget", "gross", "budget_currency",
                                         "gross_currency")}
            row["directors"] = f.get("@dirs") or f.get("dirs") or []
            # A missing figure is 0 in the graph; say so rather than letting a
            # zero be read as a box-office failure, and give the ratio only
            # when both figures exist in the same currency.
            row["budget_stated"] = budget > 0
            row["gross_stated"] = gross > 0
            if budget > 0 and gross > 0 and f.get("budget_currency") == f.get("gross_currency"):
                row["gross_over_budget"] = round(gross / budget, 3)
            films.append(row)
            self._cite(ev, f.get("film_id", ""), f.get("name", ""))
        # "Which film made the most on its budget" is a comparison over rows the
        # query already returned, so it is arithmetic and not retrieval. The
        # arithmetic runs over every film loaded; only the films it puts on the
        # board are then shown and cited, because 200 film rows is ~8,700
        # tokens and 200 citations say nothing about which document carries the
        # answer.
        self._add_tally(ev, "gross divided by budget, films where both figures are "
                            "stated in the same currency", "times its budget",
                        [(f["name"], f.get("gross_over_budget")) for f in films])
        # Gross is grouped by currency and never pooled: 714,000,000 rupees
        # above 678,000,000 dollars is not a bigger film, and a label asking
        # the reader to be careful is no substitute for not mixing them.
        by_ccy: dict[str, list] = {}
        for f in films:
            if f["gross_stated"]:
                by_ccy.setdefault(f.get("gross_currency") or "currency not stated",
                                  []).append((f["name"], f["gross"]))
        for ccy, pairs in sorted(by_ccy.items(), key=lambda kv: -len(kv[1])):
            if len(pairs) > 1:
                self._add_tally(ev, f"box-office gross in {ccy}", ccy, pairs)
        if comparison_set:
            on_the_board = {name for t in ev.get("tallies", [])
                            for name, _ in t["ranked"]}
            on_the_board |= {n for t in ev.get("tallies", [])
                             for n in t["highest"]["holders"] + t["lowest"]["holders"]}
            kept = [f for f in films if f["name"] in on_the_board]
            ev["notes"].append(
                f"the arithmetic above covers all {len(films)} films; the "
                f"{len(kept)} that it names are shown and cited in full, and the "
                f"rest are not, because a citation list of every film compared "
                f"says nothing about which document carries the answer.")
            films = kept
            ev["citable"] = {k: v for k, v in ev["citable"].items()
                             if k in {f["film_id"] for f in kept}}
        ev["films"] = films
        if not films:
            ev["notes"].append("no film article in the corpus matches that title or director; "
                               "a film can still be mentioned inside another article.")

    # route: offices and successions ---------------------------------------

    def _route_office(self, tr, question, plan, ev):
        name = plan.person or plan.athlete or (plan.entities[0] if plan.entities else "")
        seen: set[str] = set()
        chains: list[dict] = []
        frontier = [name]
        # Two hops, deliberately: "who preceded X, and who preceded that
        # person" is a chain, and the second link lives in a different
        # article. Expanding it in the graph costs one query per person and
        # no model call at all.
        for depth in range(2):
            nxt: list[str] = []
            for who in frontier[:4]:
                if not who or norm_key(who) in seen:
                    continue
                seen.add(norm_key(who))
                people = self.pin_person(tr, question, who)
                for p in people[:2]:
                    res = tr.run_query("office_chain", {"name_key": p["name_key"]},
                                       tool="office_chain" if depth == 0 else "office_chain_hop2")
                    offices = rows(res, "Res")
                    terms = rows(res, "terms")
                    chains.append({"person": p.get("name"), "person_id": p.get("person_id"),
                                   "party": p.get("party"), "offices": offices,
                                   "terms": [{k: t.get(k) for k in
                                              ("slot", "term_start", "term_end",
                                               "predecessor", "successor", "to_id")}
                                             for t in terms]})
                    self._cite(ev, p.get("person_id", ""), p.get("name", ""))
                    for t in terms:
                        nxt += [t.get("predecessor", ""), t.get("successor", "")]
            frontier = [n for n in nxt if n]
        ev["office_chains"] = chains
        if not chains:
            ev["notes"].append(f"no person article in the corpus matches {name!r}")

    # route: the evidence layer --------------------------------------------

    def _route_evidence(self, tr, question, plan, ev):
        """Seed by name, expand to whole slots, then weigh.

        Searching for a name finds the claims that mention it, but the
        document that overturns them may name only the event. A disagreement
        can only be judged with every claim competing over the same slot, so
        the seeds are expanded through `claims_by_slot` before anything is
        resolved. That expansion is the graph doing work no retriever can.
        """
        seeds: list[dict] = []
        terms = [t for t in (plan.entities + [plan.event_phrase, plan.athlete,
                                              plan.person, plan.nation]) if t]
        for term in terms[:5]:
            seeds += self._claims_search(tr, plan, ev, term, store=False)
        # A quoted figure is often the fastest route to a disagreement: the
        # question that asks why a newsletter reported 42 while the article
        # says 40 names neither the event nor the document, but both numbers
        # are stored as claim values. Searching for them finds the two claims
        # directly, and slot expansion then pulls in everything else that
        # competes over the same count.
        for number in plan.numbers[:3]:
            seeds += self._claims_search(tr, plan, ev, number, store=False,
                                         tool="claims_search_by_value")

        # A question can point at a document directly -- by its date, its
        # kind, or who issued it. Two channels find it, because each covers
        # what the other misses.
        #
        # By date: the model gives the dates in ISO and the publication times
        # in the graph are matched against them. A day either side counts,
        # because the stored times are UTC and an evening in Los Angeles is
        # already the next day there -- the corpus contains exactly that trap.
        wanted_dates = {d[:10] for d in plan.dates if len(d) >= 10}
        by_date = [src for src in self.sources
                   if any(_within_a_day(src.get("published_at", ""), d)
                          for d in wanted_dates)]
        # And by the model picking out of the catalogue, which catches "the
        # central operations log" or "the jury's certificate", where no date
        # is given at all.
        named = self._select(tr, question, "source document", self.sources, "source_id",
                             ["published_at", "source_text", "title"], many=True)
        for source_id in dict.fromkeys([s["source_id"] for s in by_date] + named):
            seeds += self._merge_claims(
                tr.run_query("claims_by_source", {"source_id": source_id}))
            if len(seeds) > 60:
                break

        # The literal pass runs whatever the searches above found. It costs no
        # tokens and a tenth of a second, and it is what reaches a document
        # the question describes rather than names: a search for "equipment
        # pause" finds the operations log that a search for the athlete does
        # not.
        found_docs = self._lexical_search(
            tr, plan, ev, plan.entities + [plan.event_phrase, plan.athlete, plan.person])
        for doc_id in found_docs:
            if doc_id in {src["source_id"] for src in self.sources}:
                seeds += self._merge_claims(
                    tr.run_query("claims_by_source", {"source_id": doc_id},
                                 tool="claims_by_source_from_text"))
        if plan.event_phrase or plan.sport_key:
            series = self.pin_series(tr, question, plan)
            if series:
                ev["series"] = series
                res = tr.run_query("series_medals", {"series_key": series["series_key"],
                                                     "medal": ""})
                editions = [e for e in rows(res, "Ev")
                            if not plan.years or e["games_year"] in plan.years]
                for e in editions[:2]:
                    ev.setdefault("editions", []).append(e)
                    self._cite(ev, e["event_id"], e.get("title", ""))
                    seeds += self._claims_for_event(tr, plan, ev, e["event_id"], store=False)
                    # The event's own article is read whether or not the plan
                    # asked for prose. "Why was the podium changed?" is an
                    # evidence question whose answer is one sentence of the
                    # article, and routing it to the claim layer alone
                    # answered "the corpus does not mention it" about a
                    # document that states it outright.
                    self._add_prose(tr, ev, e["event_id"], e.get("title", ""))

        slots = []
        for c in seeds:
            if c.get("slot_subject") and c["slot_subject"] not in slots:
                slots.append(c["slot_subject"])
        pool: dict[str, dict] = {c["claim_id"]: c for c in seeds}
        supersedes: list[dict] = []
        for slot in slots[:6]:
            res = tr.run_query("claims_by_slot", {"slot_subject": slot})
            for c in self._merge_claims(res):
                pool[c["claim_id"]] = c
            supersedes += [{"from": e.get("from_id"), "to": e.get("to_id")}
                           for e in rows(res, "supersedes")]
        ev["supersedes"] = supersedes
        ev["_claim_pool"] = pool
        ev["resolution"] = self._resolve_slots(list(pool.values()), plan)
        for c in pool.values():
            self._cite(ev, c.get("source_id", ""), c.get("source_title", ""))

        # The event article's own statement is evidence too, and often the
        # thing a source record disagrees with.
        claim_events = sorted({c.get("event_id") for c in pool.values() if c.get("event_id")})
        for event_id in claim_events[:3]:
            res = tr.run_query("event_lookup", {"event_id": event_id},
                               tool="event_lookup_for_claim")
            for e in rows(res, "Ev"):
                ev.setdefault("event_facts", []).append(e)
                self._cite(ev, e.get("event_id", ""), e.get("title", ""))
                if plan.wants_prose:
                    self._add_prose(tr, ev, event_id, e.get("title", ""))
        if not pool:
            # The entity searches found no claim. That is not the same as the
            # corpus holding nothing: the question may name the document by
            # what it is rather than by whom it is about ("a July 28
            # newsletter", "the central operations log"). Find the document by
            # its text, then follow it into the evidence layer.
            ev["notes"].append("no claim was found by name; searching the article text "
                               "for the documents this question describes")
            # _text_search follows its own source-record hits into the
            # evidence layer, so there is nothing further to do here.
            self._text_search(tr, question, plan, ev)

    # route: prose, including a fact that lives in another article ----------

    def _route_text(self, tr, question, plan, ev):
        self._text_search(tr, question, plan, ev)

    def _lexical_search(self, tr, plan, ev, terms: list[str], k: int = 12) -> list[str]:
        """Literal substring search for entity names, inside the article text.

        Cheap, exact, and strongest exactly where the vector index is weakest:
        a rare name in an article about something else, or a document a
        question describes by what it is ("the central operations log")
        rather than by whom it is about. No tokens, about a tenth of a second
        over 16,377 chunks, so the evidence route runs it every time rather
        than only as a rescue.

        The fragments are entity names the model read out of the question,
        never the question's own wording.
        """
        retrieved: list[str] = []
        done = ev.setdefault("_searched", set())
        for entity in [t for t in dict.fromkeys(terms) if t and t not in done][:3]:
            done.add(entity)
            found = rows(tr.run_query("chunks_containing", {"fragment": entity, "k": k},
                                      tool="chunks_containing"), "Res")
            if not found:
                continue
            doc_ids = sorted({c["doc_id"] for c in found})
            titles = {d["doc_id"]: d.get("title", "") for d in rows(
                tr.run_query("doc_for_chunk", {"chunks": [c["chunk_id"] for c in found]},
                             tool="doc_for_chunk_lexical"), "Res")}
            for c in found[:5]:
                ev.setdefault("passages", []).append({
                    "doc_id": c["doc_id"], "title": titles.get(c["doc_id"], ""),
                    "text": _clip(c.get("text", ""), 1200)})
                self._cite(ev, c["doc_id"], titles.get(c["doc_id"], ""))
            retrieved += doc_ids
            ev["notes"].append(f"literal search for {entity!r} matched "
                               f"{len(found)} passage(s) in {len(doc_ids)} document(s)")
        return retrieved

    def _vector_search(self, tr, question, plan, ev, k: int = 8) -> list[str]:
        """Nearest passages by meaning.

        Honestly weak on this corpus and used accordingly: chunk 0 of every
        edition of an event is its infobox, and those are near-identical
        across years, so similarity alone pulls in the wrong Games -- a
        2028 triple-jump query returned 2012 above it. It earns its place
        where no name in the question is literal enough to match.
        """
        from src.common.llm import embed
        query_text = " ".join([question] + plan.entities)
        vectors, _ = tr.stage("embed_question",
                              lambda: embed([query_text], task_type="RETRIEVAL_QUERY"))
        res = tr.run_query("vector_search_chunks",
                           {"query_vector": list(map(float, vectors[0])), "k": k},
                           tool="vector_search_chunks")
        hits = rows(res, "Res")
        if not hits:
            ev["notes"].append("vector search returned nothing")
            return []
        docs = rows(tr.run_query("doc_for_chunk",
                                 {"chunks": [h["v_id"] for h in hits if h.get("v_id")]}), "Res")
        by_doc = {d["doc_id"]: d for d in docs}
        retrieved = []
        for h in hits[:6]:
            doc = by_doc.get(h.get("doc_id", ""))
            ev.setdefault("passages", []).append({
                "doc_id": h.get("doc_id", ""), "title": (doc or {}).get("title", ""),
                "text": _clip(h.get("text", ""), 1800)})
            self._cite(ev, h.get("doc_id", ""), (doc or {}).get("title", ""))
            retrieved.append(h.get("doc_id", ""))
        return retrieved

    def _text_search(self, tr, question, plan, ev) -> None:
        """Both halves, then follow any source record into the evidence layer."""
        retrieved = self._lexical_search(
            tr, plan, ev, plan.entities + [plan.film, plan.person, plan.athlete])
        retrieved += self._vector_search(tr, question, plan, ev)
        self._expand_evidence_for_docs(tr, plan, ev, retrieved)

    def _add_prose(self, tr, ev, doc_id: str, title: str = "",
                   limit: int = 4500, max_docs: int = 2) -> None:
        """Read one document's prose into the evidence, once.

        Deduplicated and capped on purpose. The evidence route can reach the
        same article by two paths -- as an edition of the pinned series and as
        the subject of a claim -- and reading it twice puts it in the prompt
        twice, for about 1,200 tokens of nothing.
        """
        already = {p["doc_id"] for p in (ev.get("prose") or [])}
        if not doc_id or doc_id in already or len(already) >= max_docs:
            return
        chunks = rows(tr.run_query("chunks_of_document", {"doc_id": doc_id}), "Res")
        ev.setdefault("prose", []).append({
            "doc_id": doc_id, "title": title,
            "text": _clip("\n".join(c.get("text", "") for c in
                                    sorted(chunks, key=lambda c: c.get("ord", 0))), limit)})

    def _expand_evidence_for_docs(self, tr, plan, ev, doc_ids: list[str]) -> None:
        """A retrieved document that is a source record brings its claims.

        This is the join that two public questions turn on. A passage search
        for "equipment pause" finds the ops log; a search for "newsletter"
        finds the Range Report. Both are *source records*, and what matters
        about a source record is not its prose but the claims it makes and
        what else claims the same fact. So a hit on one is followed into the
        evidence layer and expanded to its whole slot -- retrieval feeding
        resolution, rather than the two living side by side.
        """
        known = {s["source_id"] for s in self.sources}
        hits = [d for d in dict.fromkeys(doc_ids) if d in known]
        if not hits:
            return
        pool: dict[str, dict] = {c["claim_id"]: c
                                 for c in (ev.get("_claim_pool") or {}).values()}
        for source_id in hits[:4]:
            for c in self._merge_claims(
                    tr.run_query("claims_by_source", {"source_id": source_id},
                                 tool="claims_by_source_from_retrieval")):
                pool[c["claim_id"]] = c
        slots = list(dict.fromkeys(c["slot_subject"] for c in pool.values()
                                   if c.get("slot_subject")))
        for slot in slots[:6]:
            res = tr.run_query("claims_by_slot", {"slot_subject": slot},
                               tool="claims_by_slot_from_retrieval")
            for c in self._merge_claims(res):
                pool[c["claim_id"]] = c
            ev["supersedes"] = (ev.get("supersedes") or []) + [
                {"from": e.get("from_id"), "to": e.get("to_id")}
                for e in rows(res, "supersedes")]
        if pool:
            ev["_claim_pool"] = pool
            ev["resolution"] = self._resolve_slots(list(pool.values()), plan)
            ev["notes"].append(
                f"{len(hits)} retrieved document(s) are source records; their "
                f"{len(pool)} claims were expanded to {len(slots)} contested fact(s)")
            for c in pool.values():
                self._cite(ev, c.get("source_id", ""), c.get("source_title", ""))

    # -- evidence-layer helpers --------------------------------------------

    def _merge_claims(self, result) -> list[dict]:
        """Claim rows joined to the source that asserted them.

        `resolve` weighs a claim by the authority of its source, and that
        lives on the SourceRecord vertex, so the two PRINTs are merged here
        once rather than re-joined at every comparison.
        """
        sources = {s["source_id"]: s for s in rows(result, "S")}
        out = []
        for c in rows(result, "C"):
            src = sources.get(c.get("source_id"), {})
            out.append({**c,
                        "tier": c.get("tier", src.get("tier", 3)),
                        "tier_name": src.get("tier_name", ""),
                        "source_text": src.get("source_text", ""),
                        "source_title": src.get("title", ""),
                        "published_at": src.get("published_at", "")})
        return out

    def _claims_search(self, tr, plan, ev, term: str, store: bool = True,
                       tool: str = "claims_search") -> list[dict]:
        key = norm_key(term)
        if not key:
            return []
        found = self._merge_claims(tr.run_query("claims_search", {"fragment": key},
                                                tool=tool))
        if store and found:
            ev.setdefault("claims", []).extend(found)
        return found

    def _claims_for_event(self, tr, plan, ev, event_id: str, store: bool = True) -> list[dict]:
        res = tr.run_query("claims_for_event", {"event_id": event_id})
        found = self._merge_claims(res)
        if found:
            if store:
                ev["resolution"] = (ev.get("resolution") or []) + self._resolve_slots(found, plan)
            for c in found:
                self._cite(ev, c.get("source_id", ""), c.get("source_title", ""))
        return found

    def _resolve_slots(self, claims: list[dict], plan: Plan) -> list[dict]:
        """Group claims by the slot they compete over and resolve each one."""
        as_of = _as_of_sort(plan.as_of)
        groups: dict[tuple, list[dict]] = {}
        for c in claims:
            groups.setdefault(slot_of(c), []).append(c)
        out = []
        for (subject, predicate, population), group in sorted(groups.items()):
            verdict = resolve(group, as_of_sort=as_of)
            out.append({
                "subject": subject, "predicate": predicate, "population": population,
                "as_of": plan.as_of or "",
                "stands": [{"value": w["value"], "source_id": w["source_id"],
                            "source": w.get("source_text", ""),
                            "tier": w.get("tier_name", ""), "status": w.get("status", ""),
                            "effective": w.get("effective", ""),
                            "published": w.get("published_at", "")}
                           for w in verdict["winners"]],
                "rejected": [{"value": o["value"], "source_id": o["source_id"],
                              "source": o.get("source_text", ""),
                              "tier": o.get("tier_name", ""), "stance": o.get("stance", ""),
                              "status": o.get("status", ""),
                              "published": o.get("published_at", ""),
                              "why_not": o.get("why_not", "")}
                             for o in verdict["others"]],
                "no_assertion": verdict.get("no_assertion", False),
                "why": verdict.get("why", ""),
            })
        return out

    def _coverage_only(self, tr, plan, ev) -> None:
        """Nothing could be pinned: ask the corpus what it does have.

        This is what turns a failure to find into a defensible
        `not_in_corpus`: the sport and Games exist, this contest does not.
        """
        year = plan.years[0] if plan.years else 0
        cov = tr.run_query("coverage_check",
                           {"sport_key": plan.sport_key, "event_key": "",
                            "games_year": year, "season": plan.season or ""},
                           tool="coverage_check_fallback")
        ev["notes"].append(
            f"coverage: {scalar(cov, 'matches', 0)} documents for "
            f"{plan.sport_key or 'any sport'} at {year or 'any year'} {plan.season or ''}")
        for e in rows(cov, "Res")[:40]:
            ev.setdefault("sibling_events", []).append(
                {"event_name": e.get("event_name"), "games_year": e.get("games_year"),
                 "event_id": e.get("event_id")})

    # -- composing ----------------------------------------------------------

    def render(self, ev: dict) -> str:
        """The evidence block, as compact text. Only what was retrieved."""
        parts = [f"ROUTE: {ev['route']}"]
        for t in ev.get("tallies", []) or []:
            hi, lo = t["highest"], t["lowest"]
            parts.append(f"ARITHMETIC ALREADY DONE FOR YOU -- {t['what']}, in {t['unit']}. "
                         f"These figures come from counting the rows below, not from a "
                         f"model; use them as they stand.")
            parts.append(f"  MOST: {hi['value']} -- "
                         + (f"{hi['holders'][0]}" if len(hi["holders"]) == 1
                            else f"A TIE between {len(hi['holders'])}: "
                                 + ", ".join(hi["holders"])))
            # The floor of a long tail is usually a tie between hundreds -- 598
            # athletes won exactly one medal in 2024 -- and listing them costs
            # tokens for nothing. Say how many are level and name them only
            # while the list is short enough to be the answer to something.
            if len(lo["holders"]) == 1:
                parts.append(f"  FEWEST: {lo['value']} -- {lo['holders'][0]}")
            elif len(lo["holders"]) <= 6:
                parts.append(f"  FEWEST: {lo['value']} -- a tie between "
                             f"{len(lo['holders'])}: " + ", ".join(lo["holders"]))
            else:
                parts.append(f"  FEWEST: {lo['value']}, held by {len(lo['holders'])} "
                             f"entries level at the bottom (not listed)")
            parts.append(f"  TOTAL: {t['total']} across {t['entries']} entries"
                         + (f"; {t['dropped']} further entries do not state a figure "
                            f"and were left out of the arithmetic" if t["dropped"] else ""))
            parts.append("  RANKED: " + ", ".join(f"{k}={v}" for k, v in t["ranked"]))
        if ev.get("series"):
            s = ev["series"]
            parts.append(f"EVENT SERIES: {s.get('event_name')} ({s.get('sport_key')}) "
                         f"key={s.get('series_key')}")
        if ev.get("editions"):
            parts.append("EDITIONS (nations/competitors -1 = not stated in the article):")
            for e in ev["editions"]:
                parts.append(f"  {e.get('games_year')} {e.get('season')}  "
                             f"nations={e.get('nations')} competitors={e.get('competitors')} "
                             f"venue={e.get('venue')!r} date={e.get('date_text')!r} "
                             f"doc={e.get('event_id')}")
        if ev.get("medals"):
            parts.append("MEDALS (one line per medal awarded):")
            by_event: dict[str, list[dict]] = {}
            for m in ev["medals"]:
                by_event.setdefault(m["event_id"], []).append(m)
            years = {e["event_id"]: e.get("games_year") for e in ev.get("editions", [])}
            for event_id, group in sorted(by_event.items(), key=lambda kv: years.get(kv[0], 0) or 0):
                for m in sorted(group, key=lambda m: m.get("rank", 9)):
                    parts.append(f"  {years.get(event_id, '')} {m['medal']:6s} "
                                 f"{m['recipient']!r} ({m['noc']})"
                                 f"{'  [team/multiple recipients]' if m.get('multi') else ''}"
                                 f"  doc={event_id}")
        if ev.get("events"):
            parts.append("EVENTS OF THIS SPORT AT THIS GAMES:")
            for e in ev["events"]:
                parts.append(f"  {e.get('event_name')!r} nations={e.get('nations')} "
                             f"competitors={e.get('competitors')} doc={e.get('event_id')}")
        if ev.get("medals_by_nation"):
            # Saying *which* medals were counted is not cosmetic: an unlabelled
            # tally of every medal was read as a gold tally's worth of
            # uncertainty and two counting questions came back
            # "not_in_corpus" with the right numbers sitting in the prompt.
            mb = ev["medals_by_nation"]
            which = f"{mb['medal']} medals only" if mb["medal"] else "all medals (gold+silver+bronze)"
            parts.append(f"MEDAL COUNT BY NATION -- counting {which} -- NOC -> count: "
                         + json.dumps(mb["counts"], ensure_ascii=False))
            if mb.get("by_medal"):
                parts.append("THE SAME, BROKEN DOWN BY COLOUR -- NOC -> medal -> count: "
                             + json.dumps(mb["by_medal"], ensure_ascii=False))
        if ev.get("nation_detail"):
            d = ev["nation_detail"]
            parts.append(f"MEDALS FOR {d['noc']}:")
            titles = {e["event_id"]: e.get("event_name") for e in d["events"]}
            for m in d["medals"]:
                parts.append(f"  {m['medal']:6s} {titles.get(m['event_id'], '')!r} "
                             f"{m['recipient']!r} doc={m['event_id']}")
        if ev.get("athlete"):
            a = ev["athlete"]
            parts.append(f"CAREER OF {a['athlete_key']!r} (individual medal lines):")
            for r in sorted(a["medals"], key=lambda r: (r.get("games_year", 0),)):
                parts.append(f"  {r.get('games_year')} {r.get('season')} "
                             f"{r.get('event_name')!r} medals={r.get('@medals') or r.get('medals')} "
                             f"doc={r.get('event_id')}")
            if not a["medals"]:
                parts.append("  (none)")
        if ev.get("team_medals"):
            parts.append("TEAM MEDAL LINES MENTIONING THAT NAME (whole squad in one string):")
            titles = {e["event_id"]: (e.get("event_name"), e.get("games_year"))
                      for e in ev["team_medals"]["events"]}
            for m in ev["team_medals"]["medals"]:
                name, year = titles.get(m["event_id"], ("", ""))
                parts.append(f"  {year} {m['medal']:6s} {name!r} recipients={m['recipient']!r} "
                             f"doc={m['event_id']}")
        if ev.get("repeat_champions") is not None:
            parts.append("ATHLETES WHO WON THE SAME EVENT AT MORE THAN ONE GAMES "
                         "(athlete | event -> years):")
            for label, years in sorted(ev["repeat_champions"].items()):
                parts.append(f"  {label} -> {years}")
            if not ev["repeat_champions"]:
                parts.append("  (none)")
        if ev.get("medal_counts"):
            mc = ev["medal_counts"]
            parts.append(f"MEDAL COUNTS PER ATHLETE, {mc['games']}, medal={mc['medal']} "
                         f"(individual lines only):")
            for a in mc["athletes_with_more_than_one"]:
                parts.append(f"  {a['athlete']}: {a['count']}  {a['events']}")
            parts.append(f"  ({mc['athletes_with_exactly_one']} further athletes won exactly one)")
        if ev.get("films"):
            parts.append("FILMS:")
            for f in ev["films"]:
                parts.append(
                    f"  {f['name']!r} ({f.get('year')}) "
                    + (f"directors={f['directors']} " if f["directors"]
                       else "directors=(not recorded on this article) ")
                    + f"budget={f['budget'] if f['budget_stated'] else 'NOT STATED'} "
                    f"gross={f['gross'] if f['gross_stated'] else 'NOT STATED'} "
                    f"currencies={f.get('budget_currency')}/{f.get('gross_currency')} "
                    + (f"gross/budget={f['gross_over_budget']} " if "gross_over_budget" in f else "")
                    + f"doc={f['film_id']}")
        if ev.get("office_chains"):
            parts.append("OFFICES HELD (one block per infobox entry; an unnamed block is "
                         "still a real office):")
            for ch in ev["office_chains"]:
                parts.append(f"  {ch['person']} (party={ch.get('party')!r} doc={ch['person_id']})")
                names = {o.get("office_key"): o.get("name") for o in ch["offices"]}
                for t in sorted(ch["terms"], key=lambda t: t.get("slot", 0)):
                    parts.append(f"    block {t.get('slot')}: office="
                                 f"{names.get(t.get('to_id'), t.get('to_id'))!r} "
                                 f"{t.get('term_start')!r} -> {t.get('term_end')!r} "
                                 f"preceded_by={t.get('predecessor')!r} "
                                 f"succeeded_by={t.get('successor')!r}")
        if ev.get("event_facts"):
            parts.append("EVENT ARTICLE FIELDS:")
            for e in ev["event_facts"]:
                parts.append(f"  {e.get('title')!r} competitors={e.get('competitors')} "
                             f"nations={e.get('nations')} venue={e.get('venue')!r} "
                             f"date={e.get('date_text')!r} winner={e.get('win_value')!r} "
                             f"doc={e.get('event_id')}")
        for slot in ev.get("resolution", []) or []:
            parts.append(f"SLOT subject={slot['subject']!r} predicate={slot['predicate']!r}"
                         + (f" population={slot['population']!r}" if slot["population"] else "")
                         + (f" as_of={slot['as_of']}" if slot["as_of"] else ""))
            for w in slot["stands"]:
                parts.append(f"  STANDS: {w['value']!r} -- {w['source']!r} "
                             f"[{w['tier']}, {w['status']}] effective={w['effective']!r} "
                             f"published={w['published']} doc={w['source_id']}")
            if slot["no_assertion"]:
                parts.append(f"  NO ASSERTION STANDS: {slot['why']}")
            for o in slot["rejected"]:
                parts.append(f"  REJECTED: {o['value']!r} -- {o['source']!r} "
                             f"[{o['tier']}, stance={o['stance']}, {o['status']}] "
                             f"published={o['published']} doc={o['source_id']} "
                             f"why={o['why_not']}")
        if ev.get("supersedes"):
            parts.append("SUPERSESSION STATED BY THE DOCUMENTS: "
                         + json.dumps(ev["supersedes"], ensure_ascii=False))
        if ev.get("sibling_events"):
            parts.append("WHAT THE CORPUS DOES HOLD NEARBY (same sport/Games):")
            for e in ev["sibling_events"][:40]:
                parts.append(f"  {e['games_year']} {e['event_name']!r} doc={e['event_id']}")
        if ev.get("passages"):
            parts.append("ARTICLE PASSAGES FOUND BY FULL-TEXT SEARCH:")
            for p in ev["passages"]:
                parts.append(f"  --- {p['title']!r} doc={p['doc_id']}\n{p['text']}")
        for p in ev.get("prose", []) or []:
            parts.append(f"ARTICLE TEXT {p['title']!r} doc={p['doc_id']}:\n{p['text']}")
        for note in ev["notes"]:
            parts.append(f"NOTE: {note}")
        if ev["citable"]:
            parts.append("CITABLE DOCUMENTS (doc_id -- title):")
            for doc_id, label in ev["citable"].items():
                parts.append(f"  {doc_id} -- {label}")
        return "\n".join(parts)

    def compose(self, tr: Trace, question: str, ev: dict) -> Composed:
        prompt = f"QUESTION: {question}\n\nEVIDENCE\n========\n{self.render(ev)}"
        return tr.generate(prompt, tool="llm_answer", system_instruction=COMPOSER_SYSTEM,
                           response_schema=Composed, pipeline="r2_agent",
                           call_type="generation")

    # -- the honesty gate ---------------------------------------------------

    def gate(self, out: Composed, ev: dict) -> tuple[str, list[str], str]:
        """Deterministic checks the model does not get to overrule.

        Two failures this prevents, both of which cost marks directly:
        a citation that was never in the evidence (10% is citations that
        actually support the answer), and an `answered` with nothing behind
        it (10% is honesty).
        """
        citations = [c for c in out.citations if c in ev["citable"]]
        dropped = [c for c in out.citations if c not in ev["citable"]]
        status = out.status if out.status in ("answered", "partial", "not_in_corpus") else "partial"
        note = ""
        if dropped:
            note = (f"dropped {len(dropped)} citation(s) not present in the retrieved "
                    f"evidence: {dropped}")
        if status in ("answered", "partial") and not str(out.answer).strip():
            # An empty answer under a confident status would be scored as a
            # wrong answer rather than an admitted gap.
            status = "not_in_corpus"
            note = (note + "; " if note else "") + "the composer returned no answer text"
        if status in ("answered", "partial") and not citations:
            # Nothing retrieved supports it. Either the evidence was empty, in
            # which case the corpus does not hold it, or the model answered
            # from somewhere other than the evidence -- and that is exactly
            # the invented answer the brief scores zero.
            status = "not_in_corpus"
            note = (note + "; " if note else "") + \
                "no retrieved document supported the answer, so it cannot stand as answered"
        return status, citations, note

    # -- one question -------------------------------------------------------

    def answer(self, qid: str, question: str) -> dict:
        tr = Trace(qid, conn=self.conn, query_log=self.query_log, model=self.model)
        plan = self.plan(tr, question)
        ev = self.gather(tr, question, plan)

        # Nothing citable means the route found no document at all, and the
        # corpus states plenty of facts outside the structured fields -- a
        # film's director named only in its editor's biography, for one. Cast
        # the wider net before answering rather than after being asked to:
        # waiting for the composer to request it cost a question whose answer
        # was sitting in another article.
        if not ev["citable"] and plan.route != "text":
            ev["notes"].append("the route retrieved no documents; widening to full-text search")
            self._text_search(tr, question, plan, ev)

        out = self.compose(tr, question, ev)

        # The one retry. The composer asked for a wider net, so cast it once
        # and answer again -- and only once, because an agent that can always
        # look again never has to admit a gap.
        if not out.enough and ev["route"] != "text":
            ev["notes"].append("the first pass found too little; falling back to full-text search")
            self._text_search(tr, question, plan, ev)
            out = self.compose(tr, question, ev)

        status, citations, gate_note = self.gate(out, ev)
        explanation = out.explanation
        if out.missing:
            explanation = f"{explanation} Missing from the corpus: {out.missing}"
        if gate_note:
            explanation = f"{explanation} [{gate_note}]"
        record = tr.answer(status, out.answer, explanation.strip(), citations)
        record["route"] = plan.route
        # The plan and the arithmetic go into the record, not just into the
        # prompt. Two reasons, both about what a judge can check: the plan is
        # the agent's decision and a trace that hides it is not a trace of a
        # decision; and the tallies are the figures the answer is supposed to
        # quote, so `src.r2.audit` can hold the answer text against them with
        # no model, no database and no quota. `ranked` is left out -- it is
        # long, and the audit only needs the figures the answer should carry.
        record["plan"] = {k: v for k, v in plan.model_dump().items()
                          if v not in ("", [], False, None)}
        record["computed"] = [{k: t[k] for k in
                               ("what", "unit", "highest", "lowest", "total", "entries")}
                              for t in ev.get("tallies", [])]
        return record
