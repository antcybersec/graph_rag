"""How much weight a source carries, read off its own description.

Round 2's corpus states who said each thing ("Source: LA28 shooting jury
(signed certificate)") and when, but never how reliable it is -- as the brief
puts it, some sources are provisional or wrong and none of them says so about
itself. So authority has to be inferred, and two public questions turn on
getting it right in a specific way:

  pub-041  a blog published on 24 July republishes a provisional 23 July
           scoreboard. It is the most recent document and it is wrong.
  pub-043  a directory entry published on 27 July recycles a draft venue
           allocation that was superseded on 1 July. Also recent, also wrong.

Hence the rule this module exists to support: **recency never decides.** The
winner is the most authoritative claim whose effective time precedes the
as-of time; publication time only ever breaks ties within one tier.

The vocabulary below is deliberately generic -- signing, issuing body, witness
-- rather than a list of the 39 sources in this corpus. Judges will ask new
questions in the live call and may add sources, so matching on "signed" and
"jury" generalizes where matching on "LA28 shooting jury" would not. Every
decision returns the phrase that produced it, so the agent can cite why one
source outranks another instead of asserting it.
"""
from __future__ import annotations

import re

# Tier 1 is a decision that binds; tier 5 is somebody's recollection. The
# ordering is what matters, not the numbers: lower always wins.
OFFICIAL_DECISION = 1
OFFICIAL_RECORD = 2
SECONDARY_MEDIA = 3
AGGREGATOR = 4
UNVERIFIED_WITNESS = 5

TIER_NAMES = {
    OFFICIAL_DECISION: "official decision",
    OFFICIAL_RECORD: "official record",
    SECONDARY_MEDIA: "secondary media",
    AGGREGATOR: "aggregator or third-party feed",
    UNVERIFIED_WITNESS: "unverified witness",
}

# Checked in order: the first match wins, so the narrowest signals come first.
# An unverified witness is tested before the official-body test on purpose --
# "transcribed by a volunteer" must not be promoted by the word "volunteer"
# appearing near an official venue name.
_SIGNALS: list[tuple[int, str, re.Pattern]] = [
    (UNVERIFIED_WITNESS, "self-described witness or personal account", re.compile(
        r"\b(blog|fan\s+account|fan\s+schedule|spectator|voice\s+note|notebook|"
        r"social\s+media|eyewitness|rumour|rumor)\b", re.I)),
    (AGGREGATOR, "third-party aggregation, not the issuing body", re.compile(
        r"\b(aggregator|third[-\s]?party|legacy\s+format|directory|tourism|"
        r"listings?\s+site|stats\s+site)\b", re.I)),
    (OFFICIAL_DECISION, "signed, certified or binding instrument", re.compile(
        r"\b(signed|certified|certificate|binding|sealed|countersigned)\b", re.I)),
    (OFFICIAL_DECISION, "adjudicating body issuing a decision", re.compile(
        r"\b(jury|chief\s+inspector|adjudicat\w+|tribunal|"
        r"(?:decision|ruling|verdict)\b)", re.I)),
    (SECONDARY_MEDIA, "reporting outlet, not the issuing body", re.compile(
        r"\b(newsletter|bulletin|roundup|headline|media\s+feed|news|press|"
        r"magazine|analytics\s+memo|coaching\s+memo)\b", re.I)),
    (OFFICIAL_RECORD, "operational record kept by the organising body", re.compile(
        r"\b(LA28|organising\s+committee|federation|operations|control\s+desk|"
        r"timing\s+contractor|registration\s+(?:system|database)|security\s+desk|"
        r"planning\s+office|scoreboard\s+operator|inspector|ticketing|"
        r"delegation|schedule|race\s+director|competition\s+director)\b", re.I)),
]

# Phrases a document uses about its own content. These qualify a claim rather
# than the source: a draft from the planning office is still an official
# record, but it says of itself that it may change.
_STATUS_SIGNALS: list[tuple[str, re.Pattern]] = [
    ("provisional", re.compile(
        r"\b(provisional|draft|preliminary|subject\s+to\s+change|unofficial|"
        r"v0\.\d+|pending|under\s+review)\b", re.I)),
    ("projected", re.compile(
        r"\b(project\w+|estimat\w+|pacing\s+model|extrapolat\w+|approx\w*|"
        r"i'?d\s+say|about\s+\d+\s*min)\b", re.I)),
    # "final" alone is a trap: in this corpus it is usually the name of a
    # round -- "men's 100 metres final", "men's triathlon final" -- and
    # matching it marked the PROVISIONAL RESULTS scoreboard capture as final.
    # It only counts as a standing when it qualifies a result or an
    # assignment, or when something is asserted to *be* final.
    ("final", re.compile(
        r"\b(confirm\w+|certified|verified|binding|effective|replaces|"
        r"supersed\w+|is\s+unchanged|(?:are|is)\s+final|"
        r"final\s+(?:placings?|results?|order|standings?|classification|"
        r"venue|assignment|allocation|medal\w*))\b", re.I)),
]

_SUPERSESSION_RE = re.compile(
    r"\b(replaces|supersed\w+|revised|rescheduled\s+from|moved\s+from|"
    r"interchanged|no\s+longer|corrected)\b", re.I)


def classify_source(source_text: str) -> dict:
    """Authority tier for a `Source:` line, with the phrase that decided it."""
    for tier, reason, pattern in _SIGNALS:
        if m := pattern.search(source_text or ""):
            return {"tier": tier, "tier_name": TIER_NAMES[tier],
                    "reason": reason, "matched": m.group(0)}
    # Nothing matched: treat as secondary rather than authoritative. Assuming
    # the worst of an unknown source is the honest default here -- it can only
    # ever lose to a signed decision, never silently outrank one.
    return {"tier": SECONDARY_MEDIA, "tier_name": TIER_NAMES[SECONDARY_MEDIA],
            "reason": "unrecognised source, defaulted to secondary", "matched": ""}


def claim_status(body: str) -> dict:
    """What a document says about the standing of its own content.

    The *earliest* signal wins, because a record states its own standing up
    front and refers to the material it replaces afterwards. Position, not
    precedence, is what separates these two:

        "Final venue assignment (v1.0). ... This assignment replaces draft v0.7."
        "Draft venue allocation (v0.7). ... remain subject to change."

    Both contain "v0.7" and one contains "Final"; only the leading phrase says
    which document you are holding. Ranking "provisional" above "final"
    unconditionally marked the signed assignment and the photo-finish jury's
    signed result as provisional -- they each describe the preliminary thing
    they overturned -- which would have handed pub-042 and pub-040 to the
    superseded value.
    """
    hits = [(m.start(), name, m.group(0)) for name, pattern in _STATUS_SIGNALS
            if (m := pattern.search(body or ""))]
    supersedes = bool(_SUPERSESSION_RE.search(body or ""))
    if hits:
        _, name, matched = min(hits)
        return {"status": name, "matched": matched, "supersedes_something": supersedes}
    return {"status": "stated", "matched": "", "supersedes_something": supersedes}


def outranks(a: dict, b: dict) -> bool:
    """True if claim `a` beats claim `b`.

    Authority first, then the claim's own standing (a final statement beats a
    provisional or projected one from the same tier), and only then effective
    time. Publication time is never consulted, which is the whole point.
    """
    rank = {"final": 0, "stated": 1, "provisional": 2, "projected": 3, "unverified": 4}
    key_a = (a["tier"], rank.get(a.get("status", "stated"), 1), -(a.get("effective_sort") or 0))
    key_b = (b["tier"], rank.get(b.get("status", "stated"), 1), -(b.get("effective_sort") or 0))
    return key_a < key_b
