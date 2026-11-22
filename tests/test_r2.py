"""Regression tests for the Round 2 agent's deterministic parts.

Round 2's score is 40% accuracy, 10% honesty and 10% citations, and the pieces
that decide all three are pure functions: the arithmetic over graph rows, the
gate that can overrule the composer's own verdict, the unit canonicalisation
that decides whether two records are talking about the same thing, and the
date tolerance that decides whether a document is the one a question names.

Each case below is a bug that shipped and cost measured marks, or the exact
failure a guard was added to prevent -- not a hypothetical:

  * the composer being left to find the maximum itself, which Round 1 measured
    at 1/21 on superlatives against 21/21 for the same rows run through code,
    and which in Round 1's investigator produced seven answers of "12" where
    12 was the evidence cap rather than the count;
  * a tie at the top reported as a single holder, which is a wrong answer
    rather than an incomplete one;
  * "10 m" and "10 metre" grounding as two different events, which hid a
    three-way disagreement by giving each figure a slot nothing contested;
  * a citation the composer invented, and an "answered" with nothing behind
    it, both of which the gate must downgrade rather than pass on;
  * a question naming "the July 24 blog" against a document stored in UTC,
    where the day boundary is an offset and not a mismatch.

No model calls and no database: every function here is pure, so this runs in a
second and needs no quota.

    python -m tests.test_r2
"""
from src.r2.agent import Agent, Composed, _tally, _within_a_day
from src.r2.claims import canon_units


# -- the arithmetic the composer is not allowed to do -----------------------

def test_tally_picks_the_maximum():
    t = _tally("golds by nation", "gold", [("USA", 14), ("KEN", 4), ("CAN", 3)])
    assert t["highest"] == {"value": 14, "holders": ["USA"]}, t["highest"]
    assert t["total"] == 21, t["total"]
    assert t["entries"] == 3


def test_tally_reports_every_holder_of_a_tie():
    """Naming one of two level holders is wrong, not brief."""
    t = _tally("Games won", "Games", [("Bolt | 100 m", 3), ("Lewis | long jump", 3),
                                      ("Rudisha | 800 m", 2)])
    assert t["highest"]["value"] == 3
    assert t["highest"]["holders"] == ["Bolt | 100 m", "Lewis | long jump"], t["highest"]


def test_tally_drops_unstated_figures_rather_than_counting_them_as_zero():
    """-1 means the article does not state the figure. Counting it as zero
    invents a smallest, and "which event had the fewest competitors" then
    answers with an event whose article is simply silent."""
    t = _tally("competitors", "competitors",
               [("Men's 100 metres", 102), ("Men's decathlon", 22), ("Men's marathon", -1)])
    assert t["lowest"] == {"value": 22, "holders": ["Men's decathlon"]}, t["lowest"]
    assert t["dropped"] == 1
    assert t["total"] == 124, t["total"]


def test_tally_on_nothing_countable_returns_none():
    assert _tally("x", "y", []) is None
    assert _tally("x", "y", [("a", -1), ("b", None)]) is None


def test_tally_ranking_is_stable_for_equal_counts():
    """Ties sort by name, so the evidence block is byte-identical across runs --
    a trace that does not reproduce is worth nothing to a judge."""
    a = _tally("x", "y", [("b", 2), ("a", 2), ("c", 1)])["ranked"]
    b = _tally("x", "y", [("a", 2), ("c", 1), ("b", 2)])["ranked"]
    assert a == b == [("a", 2), ("b", 2), ("c", 1)], (a, b)


# -- the gate the composer does not get to overrule -------------------------

def _gate(out, citable):
    return Agent.gate(None, out, {"citable": citable})


def test_gate_drops_a_citation_that_was_never_in_the_evidence():
    status, cites, note = _gate(
        Composed(status="answered", explanation="", answer="USA", citations=["Q1", "Q999"]),
        {"Q1": "Athletics at the 2024 Summer Olympics"})
    assert cites == ["Q1"], cites
    assert status == "answered"
    assert "Q999" in note, note


def test_gate_refuses_an_answer_with_no_surviving_citation():
    status, cites, note = _gate(
        Composed(status="answered", explanation="", answer="USA", citations=["Q999"]), {"Q1": "t"})
    assert status == "not_in_corpus", status
    assert cites == []
    assert "cannot stand as answered" in note, note


def test_gate_refuses_an_empty_answer():
    status, _, note = _gate(
        Composed(status="answered", explanation="", answer="   ", citations=["Q1"]), {"Q1": "t"})
    assert status == "not_in_corpus", status
    assert "no answer text" in note, note


def test_gate_keeps_a_defensible_not_in_corpus():
    """not_in_corpus with no citations is the honest answer, not a failure --
    honesty is 10% and over-abstention and under-abstention both cost it."""
    status, cites, note = _gate(
        Composed(status="not_in_corpus", explanation="", answer="", citations=[]), {})
    assert status == "not_in_corpus", status
    assert cites == [] and note == "", (cites, note)


def test_gate_normalises_an_unknown_status():
    status, _, _ = _gate(
        Composed(status="probably", explanation="", answer="USA", citations=["Q1"]), {"Q1": "t"})
    assert status == "partial", status


# -- unit canonicalisation, which decides what contradicts what -------------

def test_units_canonicalise_so_the_same_event_lands_in_one_slot():
    """Three records said 42, 40 and 36 competitors. Written "10 m", "10 metre"
    and "10 meters", each won a slot nothing contested and the disagreement was
    invisible."""
    assert canon_units("10 m sprint") == canon_units("10 metre sprint")
    assert canon_units("10 meters sprint") == canon_units("10 metre sprint")
    assert canon_units("80 kgs") == canon_units("80 kg")


def test_units_do_not_collapse_distinct_weight_classes():
    """"+80 kg" is a different class from "80 kg"; normalisation that loses the
    plus sign answers about the wrong event."""
    assert canon_units("+80 kg") != canon_units("80 kg")


# -- the day boundary ------------------------------------------------------

def test_a_published_time_in_utc_still_matches_the_day_it_names():
    assert _within_a_day("2028-07-25T02:00:00Z", "2028-07-24")
    assert _within_a_day("2028-07-24T23:00:00Z", "2028-07-24")


def test_a_different_week_does_not_match():
    assert not _within_a_day("2028-07-30T12:00:00Z", "2028-07-24")
    assert not _within_a_day("", "2028-07-24")
    assert not _within_a_day("not a date", "2028-07-24")


CHECKS = [(name, fn) for name, fn in sorted(globals().items())
          if name.startswith("test_") and callable(fn)]


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
