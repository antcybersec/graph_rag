"""Regression tests for bugs that actually cost measured accuracy.

Every case here is a bug that shipped and was caught by re-measuring, not a
hypothetical. They are pinned because each one is invisible in normal use --
the pipeline keeps answering, it just answers wrong:

  * the agent recounting capped evidence instead of reading the count its tool
    computed (99/100 -> 82/100, seven questions answered "12", the cap itself);
  * one venue existing as two vertices because of a corpus typo, so an exact
    match on the wrong spelling reported a real event as not existing;
  * impossible dates ("31 February") stored as if real;
  * a scorer strict enough to mark correct answers wrong over punctuation.

No LLM calls and no database: everything here is a pure function, so this runs
in a second and needs no quota.

    python -m tests.test_regressions
"""
import re

from src.common.infobox import norm_key
from src.eval.exact_match import is_correct
from src.ingestion.build_temporal_facts import _fact, find_conflicts, parse_date
from src.pipelines.event_graph.pipeline import _venue_candidates
from src.pipelines.investigator.pipeline import _format_history

CHECKS = []


def check(name):
    def register(fn):
        CHECKS.append((name, fn))
        return fn
    return register


@check("agent sees what its tools computed")
def _():
    # The bug: tool_result was stored on the step but never rendered into the
    # action history, so at answer time the model could not see the computed
    # count and recounted the (capped) evidence instead.
    history = _format_history([{
        "action": "query_events",
        "event_query": {"operation": "count_events_over"},
        "new_evidence": 12,
        "tool_result": "8 of the 13 Shooting events at the 2004 Summer Olympics had more than 37 competitors",
    }])
    assert "8 of the 13" in history, history
    # Errors must be visible too: a failed tool and an empty one are different.
    assert "error: boom" in _format_history([{"action": "graph_traverse", "error": "boom"}])


@check("venue spellings union, distinct venues stay distinct")
def _():
    keys = {
        "beijing science and technologyuniversity gymnasium",   # corpus typo: holds the judo final
        "beijing science and technology university gymnasium",  # normal spelling: other events
        "riocentro pavilion 4",
        "riocentro pavilion 6",
    }
    for spelling in ("Beijing Science and TechnologyUniversity Gymnasium",
                     "Beijing Science and Technology University Gymnasium"):
        found = _venue_candidates(spelling, keys)
        assert len(found) == 2, (spelling, found)
        assert norm_key(spelling) == found[0], (spelling, found)  # exact spelling queried first
    # A digit is a real difference, not a typo.
    assert _venue_candidates("Riocentro – Pavilion 4", keys) == ["riocentro pavilion 4"]


@check("impossible dates are rejected, real ones parsed")
def _():
    assert parse_date("31 February 2012")[0] == 20120201  # degrades to month precision
    assert parse_date("2012-02-30")[0] == 20120101        # degrades to year precision
    assert parse_date("7 May 2012") == (20120507, "day")
    assert parse_date("March 2, 2017") == (20170302, "day")
    assert parse_date("29 February 2012") == (20120229, "day")
    assert parse_date("present")[1] == "open"


@check("a fact's slot is the side that holds one value at a time")
def _():
    # Orientation differs per predicate and cannot be inferred from position:
    # an office has one holder; an event series has one reigning champion.
    doc = {"doc_id": "Q1", "url": "u"}
    office = _fact("Vladimir Putin", "held_office", "President of Russia", 20120507, 99991231, doc, 1.0, slot="object")
    champ = _fact("Speed skating – Men's 5000 metres", "olympic_champion", "Sven Kramer", 20140208, 99991231, doc, 1.0, slot="subject")
    assert office["slot_key"] == "president of russia" and office["value_key"] == "vladimir putin"
    assert champ["slot_key"] == "speed skating men s 5000 metres" and champ["value_key"] == "sven kramer"

    # Two people in one office at once is a conflict; one person in two offices is not.
    rivals = find_conflicts([
        office,
        _fact("Someone Else", "held_office", "President of Russia", 20130101, 20140101, {"doc_id": "Q2", "url": "u"}, 1.0, slot="object"),
    ])
    assert len(rivals["rival_claims"]) == 1 and not rivals["cross_source"]
    concurrent = find_conflicts([
        office,
        _fact("Vladimir Putin", "held_office", "Chairman of United Russia", 20080507, 20120526, doc, 1.0, slot="object"),
    ])
    assert not concurrent["rival_claims"] and len(concurrent["concurrent"]) == 1


@check("scorer: tolerant of formatting, strict about being right")
def _():
    title = ["Sailing at the 2008 Summer Olympics – Men's 470"]
    assert is_correct("", title, short_answer="Men's 470")               # event part of a title gold
    assert is_correct("", ["Dani KingLaura TrottJoanna Rowsell"],
                      short_answer="Dani King, Laura Trott, and Joanna Rowsell")  # concatenated gold
    assert not is_correct("", title, short_answer="Men's 49er")          # a different event
    assert not is_correct("", ["8"], short_answer="12")                  # the capped-count bug
    # Listing several candidates is an ambiguous answer, not a correct one --
    # the free-text pipelines must state one answer, so this gets no hedge credit.
    assert not is_correct("", ["Erik LesserDaniel Böhm"], short_answer="Marit Bjørgen | Erik LesserDaniel Böhm")

    q = "According to the provided corpus, how many fencing events at the 1988 Summer Olympics had more than 68 competitors?"
    assert is_correct("There were 3 fencing events with more than 68 competitors [Q1_c0].", ["3"], q)
    assert is_correct("Twenty-four nations competed [Q2_c0]", ["24"], "How many nations competed in X?")
    assert not is_correct("there were no such events", ["3"], q)
    # Numbers copied from the question must not be mistaken for the answer.
    assert is_correct("In the Men's 50 metre rifle three positions, 28 nations competed.", ["28"],
                      "How many nations competed in Shooting – Men's 50 metre rifle three positions?")


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
