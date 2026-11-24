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

def _gate(out, citable, tallies=None):
    ev = {"citable": citable}
    if tallies is not None:
        ev["tallies"] = tallies
    return Agent.gate(None, out, ev)


_TALLY = [{"what": "gold medals by nation", "unit": "gold",
           "highest": {"value": 9, "holders": ["JPN"]}, "entries": 12}]


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


# --- metrics: every number derived from the answer files -------------------

def test_run_time_is_the_span_of_the_run():
    """total_run_time_s was null because it was a parameter nobody passed.
    Every other number in metrics.json is derived from the answer files, so a
    judge recomputing them gets the same values; this one has to be too."""
    from src.r2.metrics import _run_time_s
    recs = [{"started_at": "2026-11-23T10:18:56Z", "finished_at": "2026-11-23T10:19:14Z"},
            {"started_at": "2026-11-23T10:19:14Z", "finished_at": "2026-11-23T10:32:55Z"}]
    assert _run_time_s(recs) == 839


def test_run_time_is_none_when_the_records_do_not_carry_times():
    from src.r2.metrics import _run_time_s
    assert _run_time_s([{"qid": "x"}]) is None
    assert _run_time_s([]) is None
    assert _run_time_s([{"started_at": "not a time", "finished_at": "also not"}]) is None


# --- the audit's noise floor, measured against real answers ----------------
# Eight of the first twenty answers of the 23 Nov run were flagged, and six of
# the eight were correct answers. A worklist that is three-quarters noise is
# not read, and the hidden set has no other check. These fix the three
# reconciliations that brought it to one in twenty-eight.

def _editions():
    """What an `event_series` question computes: two tallies, one asked for."""
    return [{"what": "competitors per edition of this contest", "unit": "competitors",
             "highest": {"value": 17, "holders": ["2024 Summer", "2028 Summer"]},
             "lowest": {"value": 14, "holders": ["2000 Summer"]},
             "total": 112, "entries": 7},
            {"what": "nations per edition of this contest", "unit": "nations",
             "highest": {"value": 17, "holders": ["2024 Summer"]},
             "lowest": {"value": 14, "holders": ["2000 Summer"]},
             "total": 110, "entries": 7}]


def test_audit_checks_the_tally_the_plan_asked_for():
    """r2-pub-010, from the real run. The answer is right row for row against
    the published key; it was flagged because the tally the question did NOT
    ask about had a tie at the same value."""
    answer = ("the number of nations competing was 14 in 2000, 16 in 2004, 16 in 2008, "
              "16 in 2012, 16 in 2016, 17 in 2024, and 15 in 2028. The 2024 Summer "
              "Olympics had the most nations (17), and the 2000 Summer the fewest (14).")
    flags = audit_one(_rec(route="event_series", answer=answer,
                           plan={"metric": "nations", "extreme": "most"},
                           computed=_editions()))
    assert flags == [], flags


def test_audit_still_reports_a_tie_in_the_tally_that_was_asked_for():
    """The suppression must not be a way out of the tie check."""
    flags = audit_one(_rec(route="event_series",
                           answer="2024 had the most competitors, 17.",
                           plan={"metric": "competitors", "extreme": "most"},
                           computed=_editions()))
    assert any("TIE NOT REPORTED" in f for f in flags), flags


def test_audit_keeps_every_tally_when_the_plan_names_none():
    """A check that silently stops running is worse than a noisy one."""
    from src.r2.audit import _asked_about
    assert len(_asked_about(_editions(), {})) == 2
    assert len(_asked_about(_editions(), {"metric": "nations"})) == 1
    # a metric that matches nothing falls back rather than checking nothing
    assert len(_asked_about(_editions(), {"metric": "qqq"})) == 2


def test_audit_knows_the_per_entry_figures_a_ranking_holds():
    """r2-pub-038 answered "22 athletes competed in the men's decathlon",
    which is exactly right and cites the gold document -- and was flagged,
    because 22 is one row of the ranking and the record used to carry only
    the top, the bottom, the total and the count."""
    tally = [{"what": "competitors per edition", "unit": "competitors",
              "highest": {"value": 27, "holders": ["2016 Summer"]},
              "lowest": {"value": 9, "holders": ["1988 Summer"]},
              "total": 189, "entries": 8,
              "ranked": [["2016 Summer", 27], ["2024 Summer", 22],
                         ["1988 Summer", 9]]}]
    flags = audit_one(_rec(route="event_series",
                           answer="22 athletes competed in the men's decathlon.",
                           plan={"metric": "competitors"}, computed=tally))
    assert flags == [], flags


def test_audit_still_catches_an_invented_figure_with_a_ranking_present():
    """The ranking must not become a blanket excuse for any number."""
    tally = [{"what": "competitors per edition", "unit": "competitors",
              "highest": {"value": 27, "holders": ["2016 Summer"]},
              "lowest": {"value": 9, "holders": ["1988 Summer"]},
              "total": 189, "entries": 8,
              "ranked": [["2016 Summer", 27], ["2024 Summer", 22]]}]
    flags = audit_one(_rec(route="event_series",
                           answer="31 athletes competed in the men's decathlon.",
                           plan={"metric": "competitors"}, computed=tally))
    assert any("FIGURE NOT IN THE ARITHMETIC" in f for f in flags), flags


def test_audit_reconciles_a_listing_that_adds_up_to_the_computed_total():
    """The per-edition figures live in `ranked`, which the record leaves out,
    so they cannot be matched one by one -- but eight numbers that add up to
    the computed total of 186 are that ranking written out."""
    tally = [{"what": "nations per edition", "unit": "nations",
              "highest": {"value": 32, "holders": ["2012 Summer"]},
              "lowest": {"value": 16, "holders": ["1988 Summer"]},
              "total": 186, "entries": 8}]
    answer = ("1988: 16, 2000: 24, 2004: 24, 2008: 26, 2012: 32, 2016: 24, "
              "2024: 21, 2028: 19. The highest was 2012 with 32.")
    flags = audit_one(_rec(route="event_series", answer=answer,
                           plan={"metric": "nations", "extreme": "most"},
                           computed=tally))
    assert flags == [], flags


def test_audit_reports_a_listing_that_does_not_add_up():
    """The stronger half of the same check: a listing whose figures do not
    reconcile with the total is a finding, not noise."""
    tally = [{"what": "nations per edition", "unit": "nations",
              "highest": {"value": 32, "holders": ["2012 Summer"]},
              "lowest": {"value": 16, "holders": ["1988 Summer"]},
              "total": 186, "entries": 8}]
    answer = ("1988: 16, 2000: 24, 2004: 24, 2008: 26, 2012: 32, 2016: 24, "
              "2024: 21, 2028: 77. The highest was 2012 with 32.")
    flags = audit_one(_rec(route="event_series", answer=answer,
                           plan={"metric": "nations", "extreme": "most"},
                           computed=tally))
    assert any("FIGURE NOT IN THE ARITHMETIC" in f for f in flags), flags


def test_audit_does_not_read_a_weight_class_as_arithmetic():
    """"Men's Greco-Roman 60 kg" put a 60 on the worklist."""
    flags = audit_one(_rec(answer="Japan won the Men's Greco-Roman 60 kg, "
                                  "with 14 golds in all.", computed=_golds()))
    assert flags == [], flags


def test_audit_does_not_read_a_hyphen_as_a_minus_sign():
    flags = audit_one(_rec(answer="six men with top-25 all-time records, and 14 golds.",
                           computed=_golds()))
    assert not any("-25" in f for f in flags), flags


# --- the superlative check: Round 1's failure class, stated as a check ------

def test_audit_catches_a_superlative_that_ignored_the_ranking():
    """The ranking puts the USA top with 14. The answer names France and no
    figure, so the model answered the superlative from somewhere other than
    the arithmetic it was handed."""
    flags = audit_one(_rec(answer="France led the medal table.",
                           plan={"extreme": "most"}, computed=_golds()))
    assert any("SUPERLATIVE NOT TAKEN FROM THE ARITHMETIC" in f for f in flags), flags


def test_audit_accepts_a_superlative_that_names_the_top_holder():
    flags = audit_one(_rec(answer="The USA led the medal table.",
                           plan={"extreme": "most"}, computed=_golds()))
    assert not any("SUPERLATIVE" in f for f in flags), flags


def test_audit_accepts_a_superlative_answered_with_the_figure_alone():
    """"How many golds did the leading nation win?" wants 14, not a name.
    Flagging that would be noise, and a worklist with noise goes unread."""
    flags = audit_one(_rec(answer="14.", plan={"extreme": "most"}, computed=_golds()))
    assert not any("SUPERLATIVE" in f for f in flags), flags


def test_audit_superlative_check_reads_the_right_end_of_the_ranking():
    """A `fewest` question is checked against `lowest`, not `highest`: naming
    the top holder is the wrong answer to it."""
    tally = [{"what": "golds by nation", "unit": "gold",
              "highest": {"value": 14, "holders": ["USA"]},
              "lowest": {"value": 1, "holders": ["Kenya", "Fiji"]},
              "total": 48, "entries": 27}]
    flags = audit_one(_rec(answer="The USA, with 14.",
                           plan={"extreme": "fewest"}, computed=tally))
    assert any("SUPERLATIVE" in f for f in flags), flags
    ok = audit_one(_rec(answer="Kenya and Fiji, with one each.",
                        plan={"extreme": "fewest"}, computed=tally))
    assert not any("SUPERLATIVE" in f for f in ok), ok


def test_audit_superlative_check_stays_quiet_when_it_cannot_check():
    """Holders too short to match leave nothing to test against, and a check
    that cannot run must not report a defect."""
    unmatchable = [{"what": "x", "unit": "n",
                    "highest": {"value": 14, "holders": ["USA"]},
                    "lowest": {"value": 1, "holders": ["a", "b"]},
                    "total": 48, "entries": 27}]
    flags = audit_one(_rec(answer="The USA, with 14.",
                           plan={"extreme": "fewest"}, computed=unmatchable))
    assert not any("SUPERLATIVE" in f for f in flags), flags


def test_audit_superlative_check_is_satisfied_by_any_one_tally():
    """One question can compute several tallies -- medals by nation plus one
    per colour -- when only one of them is what was asked. Satisfying any of
    them is enough, or the check fires on correct answers."""
    computed = _golds() + [{"what": "silvers by nation", "unit": "silver",
                            "highest": {"value": 9, "holders": ["GBR"]},
                            "lowest": {"value": 1, "holders": ["c"]},
                            "total": 30, "entries": 20}]
    flags = audit_one(_rec(answer="GBR won the most silver, nine of them.",
                           plan={"extreme": "most"}, computed=computed))
    assert not any("SUPERLATIVE" in f for f in flags), flags


def test_audit_superlative_check_stays_quiet_without_a_plan_extreme():
    """A question that did not ask for an extreme is not checked for one."""
    flags = audit_one(_rec(answer="France won 13 silver.", computed=_golds()))
    assert not any("SUPERLATIVE" in f for f in flags), flags


def test_audit_ignores_a_holder_too_short_to_match_anything():
    """A holder called "1" is inside almost any answer; matching on it would
    make the check pass unconditionally."""
    from src.r2.audit import _names
    assert _names(["1", "GB", "USA", "athletics | men s 100 metres"]) == [
        "usa", "athletics"]


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


def _checks() -> list:
    """Collected when the run starts, not when this line is read.

    As a module-level list this was built at import time, so every check
    written below it was silently skipped -- four of them, and the suite still
    reported all green. Calling it from main() means a test counts wherever it
    is written in the file.
    """
    return [(name, fn) for name, fn in sorted(globals().items())
            if name.startswith("test_") and callable(fn)]


def main() -> int:
    failures = 0
    checks = _checks()
    for name, fn in checks:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0




# -- the citation set code can supply for itself ---------------------------
# r2-pub-007 measured this: "Japan dominated judo at the 2020 Summer Olympics
# with 9 golds" -- the right nation and the right figure, both computed in
# code -- scored `partial` with no citation, because the composer offered the
# literal string "sport_games" instead of a document. Code knew the 15
# documents it had counted over the whole time.

def test_gate_supplies_the_computed_documents_when_the_composer_cites_nothing():
    citable = {f"Q{i}": f"Judo event {i}" for i in range(15)}
    status, cites, note = _gate(
        Composed(status="answered", explanation="",
                 answer="Japan dominated judo at the 2020 Summer Olympics with 9 golds.",
                 citations=["sport_games"]),
        citable, tallies=_TALLY)
    assert status == "answered", status
    assert set(cites) == set(citable), cites
    assert "arithmetic was computed over" in note, note


def test_gate_still_demotes_an_uncited_answer_with_no_arithmetic_behind_it():
    """The stand-in is only sound because code did the counting. Without a
    tally there is nothing saying which document supports what, so the
    honesty demotion stays."""
    status, cites, note = _gate(
        Composed(status="answered", explanation="", answer="Teddy Riner won in 2012 and 2016.",
                 citations=[]),
        {"Q1": "Judo at the 2012 Summer Olympics"})
    assert status == "partial", status
    assert cites == [], cites
    assert "no citation does" in note, note


def test_gate_does_not_dump_a_sweep_as_a_citation_set():
    """478 films are retrieved to compare two figures. Citing all of them
    would buy a citation mark with a precision one, so past the bound the
    demotion stays. The largest gold set in the public key is 16."""
    citable = {f"Q{i}": f"Film {i}" for i in range(400)}
    status, cites, note = _gate(
        Composed(status="answered", explanation="", answer="Titanic grossed the most.",
                 citations=[]),
        citable, tallies=_TALLY)
    assert status == "partial", status
    assert cites == [], cites
    assert "no citation does" in note, note


def test_gate_still_refuses_when_the_arithmetic_rested_on_nothing():
    """A tally does not rescue an answer with no document behind it at all --
    that is still the invented answer the brief scores zero."""
    status, cites, note = _gate(
        Composed(status="answered", explanation="", answer="Japan won 9 golds.", citations=[]),
        {}, tallies=_TALLY)
    assert status == "not_in_corpus", status
    assert cites == [], cites
    assert "no support in the corpus" in note, note



# -- the configuration a fresh clone actually gets -------------------------
# `.env` is gitignored, and `load_dotenv()` only finds it on the machine that
# wrote it. So for a judge cloning the repository, the code defaults ARE the
# configuration. They had drifted from the validated setup in five places at
# once: embeddings defaulted to the Gemini API (a different vector space from
# the BGE vectors the graph holds, so retrieval returns noise, and it needs a
# key and burns quota), the answering model defaulted to a model never
# measured, and the pacing and timeout defaults did not match the run the
# reported efficiency figures came from. Nothing errored -- it just quietly
# became a different system. This check is the thing that would have caught it.

def _env_example() -> dict:
    out = {}
    for line in open("/Users/anantkumar/graph_rag/.env.example", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _code_defaults() -> dict:
    import re
    src = open("/Users/anantkumar/graph_rag/src/common/llm.py", encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'os\.environ\.get\(\s*"([A-Z_]+)"\s*,\s*([^),]+?)\s*\)', src):
        out.setdefault(m.group(1), m.group(2).strip().strip('"').replace("_", ""))
    return out


def test_code_defaults_match_the_validated_configuration():
    """Every variable that has both a code default and a shipped value must
    agree, so a clone with no .env is the system that was measured."""
    ex, code = _env_example(), _code_defaults()
    mismatched = {k: (code[k], ex[k]) for k in set(code) & set(ex)
                  if code[k].replace("_", "") != ex[k].replace("_", "")}
    assert not mismatched, f"code default != .env.example: {mismatched}"


def test_the_embedding_default_is_the_local_model():
    """A Gemini embedding default is the dangerous one: it does not error, it
    writes vectors into a different space and retrieval silently degrades."""
    code = _code_defaults()
    assert code.get("EMBED_PROVIDER") == "local", code.get("EMBED_PROVIDER")
    assert "bge" in code.get("EMBED_MODEL", "").lower(), code.get("EMBED_MODEL")


def test_the_answering_default_is_the_measured_model():
    code = _code_defaults()
    assert code.get("GEN_MODEL") == "gemini-3.1-flash-lite", code.get("GEN_MODEL")


# -- the reproduction tool must not disclaim a real reproduction -----------
# repro.py hardcoded the provider name as "google" and compared against it, so
# in the one configuration that produced the submission it printed "this is not
# the provider the submitted file was produced with" -- on every run. The
# provider is recorded on every model step of the submitted file; read it there.

def test_repro_does_not_name_a_provider_that_the_project_never_uses():
    src = open("/Users/anantkumar/graph_rag/src/r2/repro.py", encoding="utf-8").read()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert '"google"' not in body, 'repro.py still compares against "google"'
    assert 'step.get("provider")' in body, \
        "repro.py should read the provider from the submitted record"



# -- Rule 4: nothing in the code may know a question or its answer ---------
# Round 2 rule 4 forbids regexes on question wording, question ids, and
# answers in the code. The answering code never read a question's wording --
# and the *prompt* did. COMPOSER_SYSTEM carried four "calibration" examples
# that paraphrased four real public questions near-verbatim and prescribed
# each one's key status, plus a worked example naming a film and its director
# that are the gold answer to a fifth. A prompt is a string literal in a
# source file; a judge reading the code reads it. The risk register's own
# rule-4 defence ("no code reads the wording of a question") was true and did
# not cover this, which is why this check tests the prompts against the key
# itself rather than against a list of things someone remembered to look for.

_KEY_PATH = "/Users/anantkumar/Downloads/r2/questions_public.jsonl"


def _prompt_text() -> str:
    from src.r2.agent import COMPOSER_SYSTEM, PLANNER_SYSTEM, ROUTES
    return "\n".join([COMPOSER_SYSTEM, PLANNER_SYSTEM, *ROUTES.values()])


def _key_rows() -> list:
    import json, os
    if not os.path.exists(_KEY_PATH):
        return []
    return [json.loads(l) for l in open(_KEY_PATH, encoding="utf-8") if l.strip()]


def _shingles(text: str, n: int = 6) -> set:
    import re
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def test_no_question_id_appears_anywhere_in_the_answering_code():
    """A qid in a comment is a rationale; a qid in a string the model sees is
    not. Check the prompts, which are what actually reach the model."""
    import re
    assert not re.search(r"r2-(pub-)?\d{3}", _prompt_text()), \
        "a question id appears in a prompt"


def test_no_public_question_is_paraphrased_in_a_prompt():
    """Six-word overlap against every real question. This is the check that
    would have caught the four calibration examples."""
    rows = _key_rows()
    if not rows:
        return                      # key not present: nothing to check against
    prompt = _shingles(_prompt_text())
    hits = {}
    for q in rows:
        shared = _shingles(q["question"]) & prompt
        if shared:
            hits[q["qid"]] = sorted(shared)[:2]
    assert not hits, f"prompt text overlaps real questions: {hits}"


def test_no_gold_answer_string_appears_in_a_prompt():
    """The heading example named a director and a film that are the gold
    answer to a public question. Any gold proper name of two or more words
    must not appear in prompt text."""
    import json, re
    rows = _key_rows()
    if not rows:
        return
    prompt = _prompt_text().lower()
    hits = {}
    for q in rows:
        blob = json.dumps(q.get("answer"), ensure_ascii=False)
        for name in re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+", blob):
            if len(name) > 6 and name.lower() in prompt:
                hits.setdefault(q["qid"], []).append(name)
    assert not hits, f"gold answer text appears in a prompt: {hits}"



# -- the scorer must not grade its own work --------------------------------
# The committed 91.6% was graded by gemini-3.1-flash-lite, the model that
# wrote the answers. score.py read JUDGE_MODEL at module scope, but the only
# load_dotenv() in the project runs in src/common/llm.py, which score.py
# imports lazily inside judge_one -- so the read came back None and the
# fallback silently took GEN_MODEL. metrics.json meanwhile asserted the judge
# was "different from the one that produced the answers".

def test_score_loads_the_environment_before_reading_the_judge_model():
    src = open("/Users/anantkumar/graph_rag/src/r2/score.py", encoding="utf-8").read()
    assert "load_dotenv()" in src, "score.py must load .env itself"
    assert src.index("load_dotenv()") < src.index('os.environ.get("JUDGE_MODEL")'), \
        "load_dotenv() must run before JUDGE_MODEL is read"


def test_the_judge_model_resolves_to_a_different_model_than_the_answerer():
    """The two ids must not be equal once the environment is loaded."""
    import os
    from dotenv import load_dotenv
    load_dotenv("/Users/anantkumar/graph_rag/.env")
    judge, gen = os.environ.get("JUDGE_MODEL"), os.environ.get("GEN_MODEL")
    if not judge and not gen:
        return                      # no .env on this machine: nothing to check
    assert judge, "JUDGE_MODEL is not set"
    assert judge != gen, f"judge and answerer are the same model: {judge}"


def test_score_reports_whether_the_grade_was_independent():
    """The artifact must carry the fact, so a reader of the file alone cannot
    be misled the way metrics.json was."""
    src = open("/Users/anantkumar/graph_rag/src/r2/score.py", encoding="utf-8").read()
    assert '"judge_independent"' in src, "score output must record independence"
    assert '"answered_by"' in src, "score output must record who wrote the answers"


if __name__ == "__main__":
    raise SystemExit(main())
