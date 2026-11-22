"""Deterministic corpus -> records parser for the Round 2 graph build.

No LLM calls and no network: every record here comes from structure that is
already in the documents (the `[Infobox ...]` block, or the `Source:` /
`Published:` header of an LA 2028 source record). That matters for three
reasons: it costs nothing, it cannot hallucinate, and it is parsing *document*
text rather than question text, so it stays inside Round 2's rule 4.

What the Round 1 parser got wrong for Round 2, measured over all 3,530 docs:

  * `goldNOC`/`silverNOC`/`bronzeNOC` were read and thrown away, so "which
    nation won the most wrestling golds" had no queryable field at all.
  * `bronze2` (357 events) was dropped entirely. Judo, wrestling and boxing
    award two bronze medals; keeping one silently under-counts every medal
    table and athlete career in those sports.
  * the second bronze's NOC is spelled `bronzeNOC2` in 325 events and
    `bronze2NOC` in 30 -- both appear in this corpus.
  * `gold2`/`silver2`/`silver3` exist (ties) and were dropped.
  * `prev`/`next` were dropped, so "the Games immediately after 2024" needed
    an external calendar instead of the edge the corpus already states.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from typing import Iterable, Iterator, Optional

from src.common.infobox import TITLE_RE, _INFOBOX_RE, games_id, norm_key

MEDALS = ("gold", "silver", "bronze")
MEDAL_RANK = {"gold": 1, "silver": 2, "bronze": 3}

# A medal line can hold more than one person. Two different concatenation
# styles appear in the corpus and only one of them is splittable:
#   "Dani KingLaura TrottJoanna Rowsell"      -- no separator at all
#   "Allison Jolly Lynne Jewell"              -- space-joined, ambiguous
# Rather than guess where one name ends, we flag the line as multi-recipient
# and keep it whole. Nation aggregation uses the NOC field and is unaffected;
# athlete lookups match by normalized containment, which works for both.
_GLUED_NAMES_RE = re.compile(r"[a-z][A-Z]")

_PAREN_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
_SOURCE_RE = re.compile(r"^Source:\s*(.+?)\s*$", re.M)
_PUBLISHED_RE = re.compile(r"^Published:\s*(.+?)\s*$", re.M)
_SYNTH_NOTICE_RE = re.compile(r"^\[Synthetic document:.*?\]\s*", re.S)

# Offsets for the zones the LA 2028 records are stamped in. The corpus writes
# them as "2028-07-18 09:00 PDT (UTC-07:00)", with a Unicode minus.
_TZ_OFFSETS = {"PDT": -7, "PST": -8, "CET": 1, "CEST": 2, "UTC": 0, "GMT": 0}


def _infobox_fields(text: str, kind: str) -> dict:
    """Fields of the first `[Infobox <kind>]` block, or {}.

    Documents can stack infoboxes -- tennis articles open with a tournament box
    and carry the Olympic one second -- so the kind is matched, never position.
    """
    for found, body in _INFOBOX_RE.findall(text[:4000]):
        if found == kind:
            fields = {}
            for line in body.splitlines():
                key, _, value = line.strip().partition(":")
                key = key.strip()
                if key:
                    fields[key] = value.strip()
            return fields
    return {}


def infobox_kind(text: str) -> str:
    m = re.search(r"\[Infobox ([^\]]+)\]", text[:4000])
    return m.group(1) if m else ""


def _int_or_none(value: Optional[str]) -> Optional[int]:
    m = re.match(r"\d+", (value or "").strip())
    return int(m.group()) if m else None


def medal_rows(fields: dict) -> list[dict]:
    """Every medal recorded by the infobox, including ties and second bronzes.

    Slot 1 is the primary field (`gold`), slots 2+ are the numbered variants
    (`bronze2`). The NOC key is checked under both spellings the corpus uses.
    """
    rows: list[dict] = []
    for medal in MEDALS:
        for slot in (1, 2, 3):
            name_key = medal if slot == 1 else f"{medal}{slot}"
            recipient = (fields.get(name_key) or "").strip()
            if not recipient:
                continue
            noc_keys = [f"{medal}NOC"] if slot == 1 else [f"{medal}NOC{slot}", f"{medal}{slot}NOC"]
            noc = next((fields[k].strip() for k in noc_keys if fields.get(k)), "")
            rows.append({
                "medal": medal,
                "rank": MEDAL_RANK[medal],
                "slot": slot,
                "recipient": recipient,
                "recipient_key": norm_key(recipient),
                "noc": noc.upper(),
                "multi": bool(_GLUED_NAMES_RE.search(recipient))
                         or bool(fields.get("teams"))
                         or len(recipient.split()) >= 4,
            })
    return rows


def parse_event(doc: dict) -> Optional[dict]:
    """One OlympicEvent record, or None if the doc is not an event article.

    The *title* year wins over the infobox `games` field: a handful of articles
    disagree with themselves (Sailing at the 1988 Summer Olympics carries
    `games: 1984 Summer`), and questions always name the title's Games.
    """
    t = TITLE_RE.match(doc["title"])
    if not t:
        return None
    sport, year, season, event_name = t.groups()
    f = _infobox_fields(doc["text"], "Olympic event")
    date_text = f.get("date") or f.get("dates") or ""
    year = int(year)
    return {
        "event_id": doc["doc_id"],
        "title": doc["title"],
        "sport": sport,
        "sport_key": norm_key(sport),
        "event_name": event_name,
        "event_key": norm_key(event_name),
        # Groups the same contest across editions, so a trend is one hop
        # instead of a scan: "Women's 800 metres" in 1988 and in 2028.
        "series_key": f"{norm_key(sport)}|{norm_key(event_name)}",
        "games_year": year,
        "season": season,
        "games_id": games_id(year, season),
        "venue": f.get("venue", ""),
        "venue_key": norm_key(f.get("venue", "")),
        "date_text": date_text,
        "competitors": _int_or_none(f.get("competitors")),
        "nations": _int_or_none(f.get("nations")),
        "teams": _int_or_none(f.get("teams")),
        "win_value": f.get("win_value", ""),
        "prev_year": _int_or_none(f.get("prev")),
        "next_year": _int_or_none(f.get("next")),
        "synthetic": bool(doc.get("synthetic")),
        "medals": medal_rows(f),
    }


_MAGNITUDE = {"crore": 1e7, "crores": 1e7, "lakh": 1e5, "lakhs": 1e5,
              "billion": 1e9, "billon": 1e9, "million": 1e6, "milion": 1e6,
              "millon": 1e6, "thousand": 1e3, "bn": 1e9}
# No boundary on the right-hand side, and two misspellings accepted. Both are
# answers to what this corpus actually contains rather than to tidiness:
# "$3 million14" carries a footnote marker glued to the word and
# "19 croredistributor share" runs straight into the next word, so a trailing
# \b loses the magnitude and the figure comes out as 3 and 19. And
# "\u00a3218.5 milion($245.7 million)" is misspelled in the source, which cost
# Four Weddings and a Funeral a gross of \u00a3218.50.
# No word boundary on either side, only a guard against the word being the tail
# of a longer word. This corpus writes "$55million" and "$65-70million" with no
# space, "$3 million14" with a footnote marker attached, and
# "19 croredistributor share" running into the next word -- a \b on either side
# loses the magnitude and Forrest Gump's budget becomes $55.
_MAGNITUDE_RE = re.compile(r"(?<![a-z])(crores?|lakhs?|billion|billon|bn|million|"
                           r"milion|millon|thousand)")
_CURRENCY = (("\u20b9", "INR"), ("\u00a3", "GBP"), ("\u20ac", "EUR"),
             ("\u20a9", "KRW"), ("\u00a5", "JPY"), ("$", "USD"))


def _money(raw: str) -> Optional[tuple[float, str]]:
    """(amount, currency) for an infobox money string, or None.

    The corpus mixes conventions -- "$13 million", "15 crore", "Rs 10-12
    crores" -- so the currency is returned alongside the amount and callers
    must refuse to compare across currencies. A range takes its first number.

    Two shapes made a first version of this wrong by six orders of magnitude,
    and both are common enough to have produced impossible figures in the
    graph: a budget of $600,000,000,000 for a Texas Chainsaw sequel and
    $800,000,000,000 for Lock, Stock and Two Smoking Barrels.

      * A parenthetical conversion -- "\u00a3800,000, ($1.4 million)". Taking the
        first number and then looking anywhere for a magnitude word applies the
        parenthetical's "million" to the real figure. It also read the currency
        as USD from a dollar sign that belongs to the conversion, not to the
        budget. Anything from the first "(" is dropped when a figure precedes
        it, which also fixes "11 billion(11 million)".

      * A range whose two figures have different magnitudes --
        "$600,000-$1 million". The magnitude word sits after the *second*
        number and belongs only to it. The tell is arithmetic rather than
        grammar: a range's upper bound is never below its lower bound, so when
        the number before the magnitude word is smaller than the first number,
        the word cannot apply to both. "89-97 million" passes that test and is
        multiplied; "$600,000-$1 million" fails it and is not.
    """
    s = unicodedata.normalize("NFKC", raw or "").replace(",", "")
    cut = s.find("(")
    if cut > 0 and re.search(r"\d", s[:cut]):
        s = s[:cut]
    nums = [(m.end(), float(m.group()))
            for m in re.finditer(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return None
    amount = nums[0][1]
    mag = _MAGNITUDE_RE.search(s.lower())
    if mag:
        before_word = [n for n in nums[1:] if n[0] <= mag.start()]
        if not before_word or before_word[-1][1] > amount:
            amount *= _MAGNITUDE[mag.group(1)]
    low = s.lower()
    currency = ""
    if "rs" in low or "inr" in low or "crore" in low or "lakh" in low:
        currency = "INR"
    else:
        for symbol, code in _CURRENCY:
            if symbol in s:
                currency = code
                break
    return amount, currency


def _split_people(raw: str) -> list[str]:
    """Infobox person lists use newlines, commas or no separator at all."""
    s = re.sub(r"\s*\n\s*", ", ", (raw or "").strip())
    parts = [p.strip() for p in re.split(r",|&|\band\b", s) if p.strip()]
    out: list[str] = []
    for p in parts:
        # "Joel CoenEthan Coen" -> two directors
        out.extend(x for x in re.split(r"(?<=[a-z])(?=[A-Z])", p) if x.strip())
    return [o.strip() for o in out if o.strip()]


def parse_film(doc: dict) -> Optional[dict]:
    f = _infobox_fields(doc["text"], "film")
    if not f:
        return None
    released = f.get("released", "")
    budget, gross = _money(f.get("budget", "")), _money(f.get("gross", ""))
    return {
        "film_id": doc["doc_id"],
        # "Splash (film)" and "How the Grinch Stole Christmas (2000 film)" are
        # article titles; the answer wants the film's name.
        "name": f.get("name") or _PAREN_SUFFIX_RE.sub("", doc["title"]),
        "title": doc["title"],
        "year": _int_or_none(released[:4]),
        "released": released,
        "directors": _split_people(f.get("director", "")),
        "starring": _split_people(f.get("starring", "")),
        "country": f.get("country", ""),
        "language": f.get("language", ""),
        "budget_raw": f.get("budget", ""),
        "gross_raw": f.get("gross", ""),
        "budget": budget[0] if budget else None,
        "budget_currency": budget[1] if budget else "",
        "gross": gross[0] if gross else None,
        "gross_currency": gross[1] if gross else "",
    }


def parse_person(doc: dict) -> Optional[dict]:
    """A person record with every office block the infobox carries.

    Office blocks are numbered by suffix (`office`, `office1`, `office2`) and
    the *name* of the office is often missing -- Angus King's governorship is
    the `1` block with no `office1` field, while the unsuffixed block is his
    Senate seat under the key `jr/sr`. Rather than guess, every block is kept
    with whatever label exists and the agent reasons over the set; pub-019
    turns on exactly that ("the governorship is the second office").
    """
    kind = next((k for k in ("officeholder", "person", "writer", "scientist")
                 if _infobox_fields(doc["text"], k)), None)
    if not kind:
        return None
    f = _infobox_fields(doc["text"], kind)
    blocks: dict[str, dict] = {}
    for key, value in f.items():
        m = re.match(r"^(office|term_start|term_end|predecessor|successor|jr/sr)(\d*)$", key)
        if not m:
            continue
        field, idx = m.groups()
        blocks.setdefault(idx or "0", {})[field] = value
    offices = []
    for idx in sorted(blocks, key=lambda i: (i != "0", i)):
        b = blocks[idx]
        offices.append({
            "slot": int(idx) if idx.isdigit() else 0,
            "office": b.get("office") or b.get("jr/sr") or "",
            "term_start": b.get("term_start", ""),
            "term_end": b.get("term_end", ""),
            "predecessor": b.get("predecessor", ""),
            "successor": b.get("successor", ""),
        })
    return {
        "person_id": doc["doc_id"],
        "name": f.get("name") or _PAREN_SUFFIX_RE.sub("", doc["title"]),
        "title": doc["title"],
        "kind": kind,
        "party": f.get("party", ""),
        "birth_date": f.get("birth_date", ""),
        "offices": offices,
    }


def parse_published(raw: str) -> Optional[str]:
    """"2028-07-18 09:00 PDT (UTC-07:00)" -> "2028-07-18T16:00:00Z".

    Normalizing to UTC is what makes publication times comparable at all; the
    records are stamped in PDT, PST and CET. An explicit "(UTC-07:00)" wins
    over the abbreviation when both are present.
    """
    s = unicodedata.normalize("NFKC", raw or "").replace("−", "-")
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", s)
    if not m:
        return None
    y, mo, d, hh, mm = (int(x) for x in m.groups())
    off = re.search(r"UTC\s*([+-])\s*(\d{1,2}):?(\d{2})?", s)
    if off:
        sign = -1 if off.group(1) == "-" else 1
        hours = sign * (int(off.group(2)) + int(off.group(3) or 0) / 60)
    else:
        abbr = re.search(r"\b([A-Z]{2,4})\b(?!.*\b[A-Z]{2,4}\b)", s)
        hours = _TZ_OFFSETS.get(abbr.group(1) if abbr else "", 0)
    naive = dt.datetime(y, mo, d, hh, mm)
    return (naive - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_source_record(doc: dict) -> Optional[dict]:
    """An LA 2028 source record: who said it, when, and what it says.

    These are the documents Round 2 is really about. Each states its own
    `Source:` and `Published:` and nothing else -- none of them admits to being
    provisional, so authority has to be read off the source description.
    """
    text = doc["text"]
    src, pub = _SOURCE_RE.search(text), _PUBLISHED_RE.search(text)
    if not (src and pub):
        return None
    body = text[pub.end():].strip()
    return {
        "source_id": doc["doc_id"],
        "title": doc["title"],
        "source_text": src.group(1).strip(),
        "published_raw": pub.group(1).strip(),
        "published_at": parse_published(pub.group(1)),
        "body": body,
        "synthetic": bool(doc.get("synthetic")),
    }


def iter_docs(path: str) -> Iterator[dict]:
    import json
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def classify(docs: Iterable[dict]) -> dict:
    """Every doc routed to exactly one record type, so nothing is silently lost."""
    out = {"events": [], "films": [], "people": [], "sources": [], "text_only": []}
    for doc in docs:
        if (ev := parse_event(doc)) is not None:
            out["events"].append(ev)
        elif (fl := parse_film(doc)) is not None:
            out["films"].append(fl)
        elif (sr := parse_source_record(doc)) is not None:
            out["sources"].append(sr)
        elif (pe := parse_person(doc)) is not None:
            out["people"].append(pe)
        else:
            out["text_only"].append({"doc_id": doc["doc_id"], "title": doc["title"],
                                     "infobox": infobox_kind(doc["text"])})
    return out
