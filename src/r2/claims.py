"""Source records -> claims, and claims -> one resolved answer.

A claim is one assertion by one source: who says what about which slot, from
when, and how much weight the source carries. The point of making them
first-class is that Round 2's hard questions are not about facts, they are
about *disagreements between facts* -- which means the losing claims have to
survive in the graph, with their reasons, so an answer can say "the blog
republished a provisional board" instead of silently dropping it.

Extraction is one LLM call per source record, 39 in total, cached to disk by
doc_id and prompt version. That keeps the build reproducible: a judge
rebuilding from the cache gets byte-identical claims, and a rebuild with the
cache cleared costs 39 calls once rather than anything per question. No
question is ever in scope here -- this is ingestion.

The resolution rule is in `src.r2.authority.outranks`: authority, then the
claim's own standing, then effective time. **Publication time never decides.**
Two public questions exist precisely to punish systems that take the newest
document (pub-041, pub-043) and in both the newest document is wrong.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.common.infobox import norm_key
from src.common.llm import generate
from src.r2.authority import claim_status, classify_source, outranks

# Bump when the prompt or schema changes, so stale extractions are not reused.
PROMPT_VERSION = "r2-claims-v2"
CACHE_PATH = Path("data/results/r2_claims_cache.json")

SYSTEM = """You extract assertions from a single source document for a graph database.

Report only what THIS document says. Never correct it, never add outside
knowledge, and never resolve disagreements -- other documents are handled
separately and the comparison happens later in code.

DIRECTION. Always make the subject the thing being described and the value the
property. For a result, the subject is the EVENT and the value is the athlete:
  "Gold: Zuzana Dvoracek (CZE)" in a women's 10 metre air pistol document
     -> subject "women's 10 metre air pistol", predicate "gold_medallist",
        value "Zuzana Dvoracek"
Never make the athlete the subject and the event the value; one athlete can win
several events and that must not look like a disagreement.

STANCE is the most important field. A document can state something, or merely
repeat someone else's unverified report, or deny it, or say it cannot be
established. These are completely different and must not be mixed up:
  asserts         this document states the value on its own authority
  reports         it relays a claim made elsewhere without endorsing it
  denies          it states the value is NOT the case
  cannot_confirm  it says the value could not be verified or no record exists
An operations log that says "reports of a pause of 11 and 14 minutes could not
be verified, and no official duration was recorded" is cannot_confirm for both
values. It is NOT asserting an 11-minute pause.

TIMES. Give every date or time as full ISO 8601 with the document's own UTC
offset, e.g. "2028-07-15T18:30:00-07:00". Convert nothing. The same instant is
often written in two zones and that is not a disagreement.

For each distinct assertion give:
  subject    the thing it is about, as this document names it
  predicate  a short snake_case relation, e.g. gold_medallist,
             bronze_medallist, venue, start_time, competitor_count,
             nation_count, stoppage_duration, cleared_to_start
  value      what is asserted, as written (ISO 8601 for dates and times)
  stance     asserts | reports | denies | cannot_confirm
  population what a count counts, when stated: registered_entries, starters,
             badges_scanned, finishers. Empty if not a count or not stated.
             42 registered entries and 40 starters are not a disagreement,
             they are different populations.
  effective  when the asserted fact takes effect (ISO 8601) or empty. NOT the
             publication time: a bulletin published 28 July can move an event
             to 30 July.
  replaces   what this document says it supersedes, e.g. "draft v0.7", else empty
  hedged     true if this document marks the value approximate, projected,
             provisional, unverified or under review
"""

class ExtractedClaim(BaseModel):
    subject: str
    predicate: str
    value: str
    stance: str = "asserts"
    population: str = ""
    effective: str = ""
    replaces: str = ""
    hedged: bool = False


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def extract_for_source(source: dict, cache: dict, model: Optional[str] = None) -> list[dict]:
    """Raw claims asserted by one source record, via cache when possible."""
    key = f"{source['source_id']}:{PROMPT_VERSION}"
    if key not in cache:
        prompt = (f"Source: {source['source_text']}\n"
                  f"Published: {source['published_raw']}\n"
                  f"Title: {source['title']}\n\n{source['body']}")
        parsed, _ = generate(
            prompt,
            model=model or os.environ.get("EXTRACTION_MODEL") or None,
            system_instruction=SYSTEM,
            response_schema=ExtractedClaims,
            thinking_budget=0,
            pipeline="r2_claim_extraction",
            call_type="extraction",
        )
        cache[key] = [c.model_dump() for c in parsed.claims]
    return cache[key]


_ISO_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?\s*"
    r"(Z|[+-]\d{2}:?\d{2})?")


def value_key_of(value: str) -> str:
    """A comparable form of a claim's value.

    Two documents writing the same instant in different zones are not in
    conflict, and the corpus contains exactly that trap: a venue schedule says
    15 July 18:30 PDT and an international feed says 16 July 01:30 UTC. They
    are the same moment, and a fan blog that compared the two strings reported
    a postponement that never happened. Comparing normalized instants is what
    stops us repeating the blog's mistake.
    """
    raw = (value or "").strip()
    if m := _ISO_RE.search(raw):
        y, mo, d, hh, mm, ss, off = m.groups()
        minutes = int(hh) * 60 + int(mm)
        if off and off != "Z":
            sign = -1 if off[0] == "-" else 1
            o = off[1:].replace(":", "")
            minutes -= sign * (int(o[:2]) * 60 + int(o[2:4]))
        base = dt.datetime(int(y), int(mo), int(d)) + dt.timedelta(minutes=minutes)
        return "instant:" + base.strftime("%Y-%m-%dT%H:%M:%SZ")
    # A bare number compares as a number: "40" and "40 athletes" agree.
    if m := re.fullmatch(r"[^\d]*(\d+(?:\.\d+)?)[^\d]*", raw):
        return "num:" + str(float(m.group(1)))
    return norm_key(raw)


def _effective_sort(effective: str, published_at: Optional[str]) -> int:
    """Sortable integer for a claim's effective moment.

    Falls back to publication when a document states no effective time, so a
    claim always has a position; `outranks` consults it only after authority
    and standing, so the fallback cannot promote a weak source.
    """
    s = "".join(ch for ch in (effective or published_at or "") if ch.isdigit())
    return int(s[:14].ljust(14, "0")) if s else 0


def build_claims(sources: list[dict], model: Optional[str] = None,
                 use_cache: bool = True) -> list[dict]:
    """Every claim in the corpus, each carrying its source's authority."""
    cache = _load_cache() if use_cache else {}
    out: list[dict] = []
    for source in sources:
        auth = classify_source(source["source_text"])
        standing = claim_status(source["body"])
        for raw in extract_for_source(source, cache, model=model):
            # A document's own hedge beats the body-level signal: a jury's
            # certified register can still mark one line as under review.
            status = "provisional" if raw.get("hedged") else standing["status"]
            out.append({
                "claim_id": f"{source['source_id']}#{len(out)}",
                "subject": raw["subject"],
                "subject_key": norm_key(raw["subject"]),
                "predicate": norm_key(raw["predicate"]).replace(" ", "_"),
                "value": raw["value"],
                "value_key": value_key_of(raw["value"]),
                "population": norm_key(raw.get("population", "")),
                "effective": raw.get("effective", ""),
                "effective_sort": _effective_sort(raw.get("effective"), source["published_at"]),
                "replaces": raw.get("replaces", ""),
                "stance": raw.get("stance", "asserts"),
                "status": status,
                "source_id": source["source_id"],
                "source_text": source["source_text"],
                "published_at": source["published_at"],
                "tier": auth["tier"],
                "tier_name": auth["tier_name"],
                "tier_reason": auth["reason"],
                "tier_matched": auth["matched"],
                "asserts_supersession": standing["supersedes_something"],
            })
    if use_cache:
        _save_cache(cache)
    return out


def slot_of(claim: dict) -> tuple:
    """The thing a claim competes over.

    Population is part of the slot on purpose. "42 registered entries" and
    "40 starters" are both true and must never be scored as a contradiction --
    pub-045 asks for both numbers and pub-047 asks why a newsletter reported
    the wrong one.
    """
    return (claim.get("slot_subject") or claim["subject_key"],
            claim["predicate"], claim["population"])


def resolve(claims: list[dict], as_of_sort: Optional[int] = None) -> dict:
    """What stands for one slot, what does not, and why.

    Three things this has to get right, each of which cost a wrong answer when
    it was missing:

    **A slot can hold several values.** The fencing jury's certified register
    awards two bronzes in men's sabre and says both are final. Picking a single
    winner per slot declared one of them beaten "on effective time" -- which is
    the aggregator's error, the very thing the question asks us to reject. So
    the winner is the authoritative *source*, and every value that source
    asserts for the slot stands together.

    **Repeating a claim is not making one.** The archery operations log lists
    the 11- and 14-minute reports precisely in order to say neither could be
    verified. Treating those as its own assertions answered "11 minutes" to a
    question whose answer is that no duration is established. Only `asserts`
    claims can win; `reports`, `denies` and `cannot_confirm` are evidence
    about the slot and are returned for the explanation.

    **No assertion is a real outcome.** When nothing asserts a value, the slot
    resolves to no winner, which is how `not_in_corpus` gets decided from
    evidence rather than from a model's reluctance.
    """
    pool = [c for c in claims
            if as_of_sort is None or (c["effective_sort"] or 0) <= as_of_sort]
    asserted = [c for c in pool if c.get("stance", "asserts") == "asserts"]
    reported = [c for c in pool if c.get("stance", "asserts") != "asserts"]

    if not asserted:
        return {"winners": [], "winner": None, "others": reported, "contested": bool(reported),
                "no_assertion": True,
                "why": _no_assertion_reason(reported)}

    best = asserted[0]
    for c in asserted[1:]:
        if outranks(c, best):
            best = c

    # An official record that says the value could not be established outranks
    # a witness who is sure of it. The archery operations log reports the 11-
    # and 14-minute accounts in order to state that neither was verified and no
    # official duration exists; without this, a volunteer's notebook became the
    # answer to a question whose answer is that nothing is confirmed.
    blockers = [c for c in reported
                if c.get("stance") in ("cannot_confirm", "denies")
                and c["tier"] < best["tier"]]
    if blockers:
        blocker = min(blockers, key=lambda c: c["tier"])
        return {"winners": [], "winner": None, "no_assertion": True,
                "contested": True,
                "others": [{**c, "why_not": _why_not(c, best, set())} for c in pool],
                "why": (f"{blocker['source_text']} ({blocker['tier_name']}) states this "
                        f"could not be established; the only assertions come from "
                        f"lower-authority sources")}
    # Every value the winning source asserts for this slot, in one voice.
    winners = [c for c in asserted if c["source_id"] == best["source_id"]]
    winning_values = {c["value_key"] for c in winners}

    others = []
    for c in pool:
        if any(c["claim_id"] == w["claim_id"] for w in winners):
            continue
        others.append({**c, "why_not": _why_not(c, best, winning_values)})
    return {"winners": winners, "winner": best, "others": others,
            "no_assertion": False,
            "contested": any(o["why_not"] != "agrees" for o in others)}


def _no_assertion_reason(reported: list[dict]) -> str:
    if not reported:
        return "no source in the corpus addresses this"
    kinds = {c.get("stance") for c in reported}
    if "cannot_confirm" in kinds:
        return ("the only official record states the value could not be verified; "
                "the remaining accounts are unverified reports")
    if kinds == {"denies"}:
        return "every source that addresses this denies it"
    return "no source asserts a value on its own authority; the accounts are unverified reports"


def _why_not(c: dict, best: dict, winning_values: set) -> str:
    stance = c.get("stance", "asserts")
    if stance == "reports":
        return "repeats a claim made elsewhere without endorsing it"
    if stance == "cannot_confirm":
        return "states this could not be verified"
    if stance == "denies":
        return "denies this value"
    if c["value_key"] in winning_values:
        return "agrees"
    later = (c["published_at"] or "") > (best["published_at"] or "")
    if c["tier"] > best["tier"]:
        tail = ("; published later but publication time confers no authority" if later else "")
        return f"lower authority ({c['tier_name']} vs {best['tier_name']}){tail}"
    if c["status"] in ("provisional", "projected"):
        return f"{c['status']} value, superseded by a {best['status']} one from an equal source"
    return "outranked on effective time"


def group_by_slot(claims: list[dict]) -> dict:
    out: dict[tuple, list[dict]] = {}
    for c in claims:
        out.setdefault(slot_of(c), []).append(c)
    return out


# --- grounding claims to the event graph ------------------------------------
#
# Sources name the same contest differently: the fencing jury writes "men's
# individual sabre", the aggregator writes "men's sabre"; the timing contractor
# attributes a split to "Caleb Tolliver" where the jury attributes it to the
# "men's 10,000 metres". Grouped by their own strings those are four slots, and
# a claim then wins a slot nothing contests -- which is how the aggregator's
# "Delaunay is the only bronze medallist" came to stand unopposed next to the
# certified register that awards two.
#
# Identity has to come from somewhere outside the sentence, and the corpus
# already provides it: the 127 LA 2028 event articles. Resolving a subject to
# an event vertex is what makes the slot canonical, and it is the reason this
# is a graph problem rather than a retrieval one.

_GENDER_RE = re.compile(r"\b(men|women|mixed)\b")
_DIGIT_RE = re.compile(r"\d+")

# Unit words, canonicalised for *matching only*. Primary ids keep whatever
# `norm_key` produced, because changing those would invalidate the graph.
#
# This earns its place from a real failure. A newsletter wrote "women's 10 m
# air pistol" and the event article is "Women's 10 metre air pistol". The
# token sets differ by one word -- "m" against "metre" -- so neither contained
# the other, the subject stayed ungrounded, and its claim of 42 competitors
# landed in a slot of its own. The event's 40 and the jury's 36 were in
# another. Three documents disagreeing about one number, and the
# disagreement was invisible: each claim won a slot nothing contested. Unit
# words are exactly the sort of thing sources abbreviate differently, and
# they are never what distinguishes two events.
_UNITS = {
    "m": "metre", "metres": "metre", "meter": "metre", "meters": "metre",
    "km": "kilometre", "kilometres": "kilometre", "kilometers": "kilometre",
    "kilometer": "kilometre",
    "kgs": "kg", "kilograms": "kg", "kilogram": "kg",
    "mtr": "metre", "mts": "metre",
}


def canon_units(key: str) -> str:
    """A normalised key with unit words spelled one way."""
    return " ".join(_UNITS.get(t, t) for t in (key or "").split())

# Words that appear in so many event names that sharing them says nothing.
# "Men's individual" is archery, fencing, and half a dozen other sports; what
# identifies a contest is the discipline word next to it. Grounding on shared
# generic tokens alone made "men's individual sabre" tie between Fencing's
# Men's sabre and Archery's Men's individual, so it matched neither and the
# aggregator's one-bronze claim ended up in a slot the certified register
# never reached.
_GENERIC_TOKENS = {
    "men", "women", "mixed", "s", "individual", "team", "final", "match",
    "event", "the", "at", "on", "olympic", "olympics", "summer", "winter",
    "round", "competition", "medal", "medals",
}


def _distinctive(key: str) -> set:
    return {t for t in canon_units(key).split() if t not in _GENERIC_TOKENS}


def _compatible(a: str, b: str) -> bool:
    """Refuse matches that differ on the things that distinguish events.

    Gender and numbers are never incidental here: "Men's 80 kg" and "Women's
    80 kg" are different events, and so are the 100 metres and the 10,000
    metres. Round 1 lost questions to exactly this kind of near-match.
    """
    if set(_GENDER_RE.findall(a)) and set(_GENDER_RE.findall(b)):
        if set(_GENDER_RE.findall(a)) != set(_GENDER_RE.findall(b)):
            return False
    da, db = set(_DIGIT_RE.findall(a)), set(_DIGIT_RE.findall(b))
    return not (da and db and da != db)


def ground_subject(subject: str, events: list[dict], year: int = 2028) -> Optional[dict]:
    """The event a claim is about, or None if it cannot be pinned down.

    Deliberately conservative: an ambiguous subject stays ungrounded and keeps
    its own slot rather than being attached to the wrong event. A wrong
    grounding would silently merge two contests, which is worse than leaving a
    claim where a question can still reach it by text.
    """
    key = canon_units(norm_key(subject))
    if not key:
        return None
    pool = [e for e in events if e["games_year"] == year]
    exact = [e for e in pool if canon_units(e["event_key"]) == key]
    if len(exact) == 1:
        return exact[0]

    kd = _distinctive(key)
    scored = []
    for e in pool:
        ek = canon_units(e["event_key"])
        if not _compatible(key, ek):
            continue
        kt, et = set(key.split()), set(ek.split())
        if not kt or not et:
            continue
        # Subject inside the event name ("men's sabre" in "men's sabre"), or
        # the event name inside a wordier subject ("men's 100 metres final").
        if not (et <= kt or kt <= et):
            continue
        shared = len(kd & _distinctive(ek))
        if kd and not shared:
            # Only generic words in common -- that is not an identification.
            continue
        scored.append(((shared, len(kt & et) / max(len(kt | et), 1)), e))
    if not scored:
        return None
    top = max(score for score, _ in scored)
    rivals = [e for score, e in scored if score == top]
    # Still tied: the discipline is what separates them and the claim did not
    # name it. Ungrounded is the honest outcome -- a wrong grounding would
    # merge two different contests.
    return rivals[0] if len(rivals) == 1 else None


def ground_claims(claims: list[dict], events: list[dict]) -> list[dict]:
    """Attach an event_id to every claim whose subject names an event."""
    cache: dict[str, Optional[dict]] = {}
    for c in claims:
        key = c["subject_key"]
        if key not in cache:
            cache[key] = ground_subject(c["subject"], events)
        ev = cache[key]
        c["event_id"] = ev["event_id"] if ev else ""
        c["event_title"] = ev["title"] if ev else ""
        # The canonical slot subject: the event when we could pin one down,
        # the claim's own wording when we could not.
        c["slot_subject"] = c["event_id"] or key
    return claims
