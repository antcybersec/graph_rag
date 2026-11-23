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
from src.r2.audit import audit_one
from src.r2.claims import canon_units
from src.r2.parse import _money
from src.r2.repro import compare as _compare


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


def test_gate_refuses_an_answer_when_nothing_at_all_was_retrieved():
    """No evidence and a confident answer is the invented answer the brief
    scores zero."""
    status, cites, note = _gate(
        Composed(status="answered", explanation="", answer="USA", citations=["Q999"]), {})
    assert status == "not_in_corpus", status
    assert cites == []
    assert "no support in the corpus" in note, note


def test_gate_keeps_an_uncited_answer_when_documents_were_retrieved():
    """r2-pub-029: a correct account of which editions of the men's +100 kg
    judo the corpus holds, Teddy Riner named twice from retrieved passages,
    and no citations. Discarding it lost the accuracy, honesty and citation
    marks together; demoting it loses only the citation mark."""
    status, cites, note = _gate(
        Composed(status="partial", explanation="",
                 answer="Teddy Riner won in 2012 and 2016; 2004 and 2008 are missing.",
                 citations=["Q999"]), {"Q1005663": "Judo at the 2012 Games"})
    assert status == "partial", status
    assert cites == []
    assert "no citation does" in note, note


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


# -- the offline audit, which is the only check the hidden set gets ---------

def _rec(**kw):
    base = {"qid": "t", "status": "answered", "route": "sport_games",
            "answer": "", "citations": ["Q1"], "computed": [],
            "trace": [{"step": 1, "tool": "medals_by_nation",
                       "tigergraph": True, "rows": 27}]}
    base.update(kw)
    return base


def _golds(hi=14, holders=("USA",)):
    return [{"what": "golds by nation", "unit": "gold",
             "highest": {"value": hi, "holders": list(holders)},
             "lowest": {"value": 1, "holders": ["a", "b"]},
             "total": 48, "entries": 27}]


def test_audit_passes_an_answer_that_quotes_the_computed_figure():
    assert audit_one(_rec(answer="The USA, with 14 golds.", computed=_golds())) == []


def test_audit_catches_the_evidence_cap_answered_as_a_count():
    """Round 1's worst regression: three hidden answers said "12", which was
    the cap on the evidence rather than the count. Nothing in the file said so."""
    flags = audit_one(_rec(answer="12 events had more than 25 nations.", computed=_golds()))
    assert any("FIGURE NOT IN THE ARITHMETIC" in f for f in flags), flags


def test_audit_catches_a_tie_reported_as_one_holder():
    flags = audit_one(_rec(route="repeat_champions",
                           answer="Usain Bolt won the most, at three Games.",
                           computed=_golds(3, ("Usain Bolt | 100 m", "Carl Lewis | long jump"))))
    assert any("TIE NOT REPORTED" in f for f in flags), flags


def test_audit_accepts_a_tie_that_is_named_as_one():
    flags = audit_one(_rec(route="repeat_champions",
                           answer="Usain Bolt and Carl Lewis tie on 3 Games each.",
                           computed=_golds(3, ("Usain Bolt | 100 m", "Carl Lewis | long jump"))))
    assert not any("TIE NOT REPORTED" in f for f in flags), flags


def test_audit_catches_a_status_that_contradicts_its_own_text():
    assert any("refusal" in f for f in
               audit_one(_rec(answer="The corpus does not state this.")))


def test_audit_accepts_not_in_corpus_that_cites_the_documents_showing_the_gap():
    """The public set's own key gives gold documents for two of its four
    not_in_corpus questions: the entry list naming only anonymous reserve
    slots, and the three accounts each reporting a different unverified
    duration. The documents are what establish the gap, so citing them is
    right -- and a check that called it a defect would push the system toward
    citing nothing on the questions where citations are worth most."""
    assert audit_one(_rec(status="not_in_corpus",
                          answer="The corpus does not establish a duration.",
                          citations=["SYN-LA28-S003", "SYN-LA28-S027"])) == []


def test_audit_catches_a_route_that_retrieved_nothing():
    flags = audit_one(_rec(answer="USA.", trace=[{"step": 1, "tool": "medals_by_nation",
                                                 "tigergraph": True, "rows": 0}]))
    assert any("zero rows" in f for f in flags), flags


def test_audit_ignores_a_year_that_is_not_arithmetic():
    """Years are everywhere in this corpus. Flagging 2024 as an unexplained
    figure would bury the one flag that matters."""
    flags = audit_one(_rec(answer="At the 2024 Summer Olympics the USA won 14.",
                           computed=_golds()))
    assert flags == [], flags


def test_audit_ignores_a_number_that_is_part_of_an_event_name():
    """"Men's 100 metres" is a name, not arithmetic. Flagging the 100 buries
    the flag that matters."""
    flags = audit_one(_rec(route="repeat_champions",
                           answer="Usain Bolt and Carl Lewis both won 3 times "
                                  "(Men's 100 metres, Men's long jump).",
                           computed=[{"what": "Games won", "unit": "Games",
                                      "highest": {"value": 3,
                                                  "holders": ["Usain Bolt | Men's 100 metres",
                                                              "Carl Lewis | Men's long jump"]},
                                      "lowest": {"value": 2, "holders": ["x"]},
                                      "total": 67, "entries": 31}]))
    assert flags == [], flags


def test_audit_does_not_flag_a_widening_that_ended_in_not_in_corpus():
    """Widening and then reporting the corpus does not hold it is the route
    working, not a defect."""
    steps = [{"step": 1, "tool": "coverage_check", "tigergraph": True, "rows": 1},
             {"step": 2, "tool": "vector_search_chunks", "tigergraph": True, "rows": 8}]
    assert audit_one(_rec(status="not_in_corpus", answer="", citations=[],
                          trace=steps)) == []
    assert any("full-text" in f for f in
               audit_one(_rec(status="answered", answer="Paris.", trace=steps)))


# -- money, which decides every budget and box-office answer ----------------

def test_a_parenthetical_conversion_does_not_multiply_the_real_figure():
    """"£800,000, ($1.4 million)" was read as £800,000,000,000, because the
    first number was found here and the magnitude word in the conversion. The
    currency came out USD from a dollar sign belonging to the conversion."""
    assert _money("\u00a3800,000, ($1.4 million)") == (800000.0, "GBP")
    assert _money("11 billion(11 million)") == (11000000000.0, "")


def test_a_range_of_two_magnitudes_keeps_the_lower_bound_unmultiplied():
    """"$600,000-$1 million" gave a Texas Chainsaw sequel a $600 billion
    budget. A range's upper bound is never below its lower bound, so a
    magnitude word after a smaller second number cannot apply to the first."""
    assert _money("$600,000\u2013$1 million") == (600000.0, "USD")


def test_a_range_of_one_magnitude_still_multiplies():
    """"89-97 million" is one magnitude across both bounds: 97 > 89, so the
    word applies to the range and the lower bound is 89 million, not 89."""
    assert _money("89\u201397 million") == (89000000.0, "")
    assert _money("Rs 10-12 crores") == (100000000.0, "INR")


def test_a_magnitude_word_glued_to_the_next_token_still_counts():
    """This corpus writes "$3 million14" with a footnote marker attached and
    "19 croredistributor share" running into the next word. Requiring a word
    boundary on the right of the magnitude lost both, giving a $3 budget and a
    19-rupee gross."""
    assert _money("$3 million14") == (3000000.0, "USD")
    assert _money("19 croredistributor share") == (190000000.0, "INR")


def test_a_magnitude_with_no_space_before_it_still_counts():
    """This corpus writes "$55million" and "$65-70million" with no space.
    Requiring a word boundary on the left gave Forrest Gump a $55 budget and
    Saving Private Ryan a $65 one -- both out by a factor of a million, and
    both prominent enough to be asked about."""
    assert _money("$55million") == (55000000.0, "USD")
    assert _money("$65\u201370million") == (65000000.0, "USD")
    assert _money("$678million") == (678000000.0, "USD")


def test_an_unknown_abbreviation_is_left_alone():
    """Chalte Chalte's infobox reads "11 c". That is probably crore, and
    probably is not good enough: guessing would be inventing a figure the
    corpus does not state. Both of its figures carry the same unknown unit, so
    their ratio survives and the absolute value is honestly unscaled."""
    assert _money("11 c") == (11.0, "")


def test_a_magnitude_misspelled_in_the_source_still_counts():
    """"\u00a3218.5 milion($245.7 million)" is misspelled in the corpus. Reading
    it literally gave Four Weddings and a Funeral a gross of \u00a3218.50."""
    assert _money("\u00a3218.5 milion($245.7 million)") == (218500000.0, "GBP")


def test_a_multi_part_budget_takes_the_first_component():
    """El Mariachi's infobox reads "Production: $7,225Post-production:
    $200,000". The first figure is what this returns, which is the production
    budget the film is famous for -- documented rather than clever, because
    every rule that summed the parts also summed a footnote marker."""
    assert _money("Production: $7,225Post-production: $200,000") == (7225.0, "USD")


def test_the_indian_conventions_and_the_plain_cases():
    assert _money("$28.1 million") == (28100000.0, "USD")
    assert _money("$185,898") == (185898.0, "USD")
    assert _money("\u20b915 crore") == (150000000.0, "INR")
    assert _money("") is None
    assert _money("not a number") is None
    assert _money("5 bn") == (5000000000.0, "")


# --- reproduction: what counts as the same answer twice -------------------
# Repeatability is 15% of the score and the brief disqualifies answers the
# system cannot reproduce. The composed sentence is model-written, so the
# check cannot be byte equality; these fix what it is instead.

def _pair(answer="France won 13 silver", status="answered",
          citations=("a", "b"), tools=("q1",)):
    return {"qid": "x", "status": status, "answer": answer,
            "citations": list(citations), "route": "r",
            "trace": [{"tigergraph": True, "tool": t} for t in tools]}


def test_reproduction_ignores_rewording_that_keeps_every_figure():
    """The one case byte equality would get wrong: same facts, new sentence."""
    c = _compare(_pair(), _pair(answer="13 silver went to France",
                                citations=("b", "a")))
    assert c["verdict"] == "MATCH", c
    assert c["same_figures"] and c["same_citations"] and c["same_retrieval"]


def test_reproduction_fails_when_a_figure_moves():
    """A moved count means something non-deterministic is feeding the
    arithmetic, which is the finding this tool exists to catch."""
    assert _compare(_pair(), _pair(answer="France won 14 silver"))["verdict"] == "DIFFERENT"


def test_reproduction_fails_when_the_status_moves():
    assert _compare(_pair(), _pair(status="partial"))["verdict"] == "DIFFERENT"


def test_reproduction_is_close_not_different_when_only_citations_move():
    """Worth distinguishing: a lost citation costs citation marks, a moved
    figure costs the answer. They are not the same defect."""
    c = _compare(_pair(), _pair(citations=("a",)))
    assert c["verdict"] == "CLOSE"
    assert c["citations_lost"] == ["b"] and c["citations_gained"] == []


def test_reproduction_counts_figures_in_order_not_as_a_set():
    """"13 silver, 28 in all" and "28 silver, 13 in all" are different
    answers, so the comparison is a list and not a set."""
    c = _compare(_pair(answer="13 silver and 28 in all"),
                 _pair(answer="28 silver and 13 in all"))
    assert c["verdict"] == "DIFFERENT"


def test_reproduction_notices_a_changed_query_sequence():
    c = _compare(_pair(), _pair(tools=("q1", "q2")))
    assert not c["same_retrieval"]


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
