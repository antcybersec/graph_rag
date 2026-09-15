"""Deterministic parser for the `[Infobox Olympic event]` block that heads most
corpus documents, plus the key normalization shared by ingestion and query time.

Why this exists (see docs/architecture.md, "Structured event layer"): every
eval question is answered by infobox fields -- competitors, nations, venue,
date, gold -- of documents titled "<Sport> at the <YYYY> <Summer|Winter>
Olympics – <Event>". The LLM-extracted entity graph never stored those fields
as queryable properties, and top-k chunk retrieval can't count across the
8-43 documents an aggregation question spans. Parsing them directly costs no
LLM calls and has no extraction errors.
"""
import json
import re
import unicodedata
from typing import Optional

TITLE_RE = re.compile(r"^(.+?) at the (\d{4}) (Summer|Winter) Olympics – (.+)$")
_INFOBOX_RE = re.compile(r"\[Infobox ([^\]]+)\]\n((?:  [^\n]*\n)+)")

# The real Games calendar, not just the years present in the corpus:
# "the Games held immediately before 2022" is a fact about the Olympics
# (Winter moved off the Summer cycle after 1992 -> 1994).
OLYMPIC_CALENDAR = {
    "Summer": [1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024],
    "Winter": [1984, 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022],
}

_GAMES_SUFFIX_RE = re.compile(r"\s*(?:at|in|during) the \d{4} (?:Summer|Winter) Olympics\s*$", re.I)


def norm_key(s: Optional[str]) -> str:
    """Case/punctuation-insensitive key. Keeps `+` on purpose: "Men's 80 kg" and
    "Men's +80 kg" are different events."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"[^\w+]+", " ", s).strip()


def date_key(s: Optional[str]) -> str:
    """norm_key for a date, minus a trailing "at the 2012 Summer Olympics" --
    questions append that when the infobox date carries no year."""
    return norm_key(_GAMES_SUFFIX_RE.sub("", s or ""))


def games_id(year: int, season: str) -> str:
    return f"{year} {season}"


def parse_infobox(text: str) -> dict:
    """Fields of the Olympic-event infobox, or {} if the doc has none.

    A doc can stack several infoboxes -- tennis articles open with a
    "tennis tournament event" box and carry the Olympic one second."""
    blocks = _INFOBOX_RE.findall(text[:4000])
    body = next((b for kind, b in blocks if kind == "Olympic event"), None)
    if body is None:
        return {}
    fields = {}
    for line in body.splitlines():
        k, _, v = line.strip().partition(":")
        fields[k.strip()] = v.strip()
    return fields


def _leading_int(value: str) -> int:
    """"32 (16 pairs)" -> 32, "23 teams" -> 23, missing -> -1."""
    m = re.match(r"\d+", value or "")
    return int(m.group()) if m else -1


def parse_event_doc(doc: dict) -> Optional[dict]:
    """One Event record for an Olympic-event document, or None for any other doc."""
    t = TITLE_RE.match(doc["title"])
    if not t:
        return None
    sport, year, season, event_name = t.groups()
    f = parse_infobox(doc["text"])
    date_text = f.get("date") or f.get("dates", "")
    return {
        "event_id": doc["doc_id"],
        "title": doc["title"],
        "sport": sport,
        "sport_key": norm_key(sport),
        "games_year": int(year),
        "season": season,
        "games_id": games_id(int(year), season),
        "event_name": event_name,
        "event_key": norm_key(event_name),
        "venue": f.get("venue", ""),
        "venue_key": norm_key(f.get("venue", "")),
        "date_text": date_text,
        "date_key": date_key(date_text),
        "competitors": _leading_int(f.get("competitors", "")),
        "nations": _leading_int(f.get("nations", "")),
        "gold": f.get("gold", ""),
        "silver": f.get("silver", ""),
        "bronze": f.get("bronze", ""),
    }


def load_events(corpus_path: str = "data/raw_dataset/corpus/corpus.jsonl") -> list:
    events = []
    with open(corpus_path) as f:
        for line in f:
            ev = parse_event_doc(json.loads(line))
            if ev:
                events.append(ev)
    return events


if __name__ == "__main__":
    evs = load_events()
    print(f"{len(evs)} events, {sum(e['competitors'] >= 0 for e in evs)} with competitors, "
          f"{len({e['sport_key'] for e in evs})} sports, {len({e['venue_key'] for e in evs if e['venue_key']})} venues")
    print(evs[0])
