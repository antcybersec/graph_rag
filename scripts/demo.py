"""Live demo: answer questions with no generative model in the loop.

Built for presenting. Every earlier pipeline here depends on Gemini, whose free
tier is 500 calls a day -- a rate limit mid-demo is unrecoverable. This runner
uses the Jev selection planner, so the only calls it makes are TypeSafe System
One requests, and it prints the generation-call count so the audience can see
it stay at zero.

    python scripts/demo.py                  # the five question shapes
    python scripts/demo.py "your question"  # anything you like
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tg_conn import get_connection          # noqa: E402
from src.common.token_tracker import TokenTracker      # noqa: E402
from src.pipelines.jev_planner import pipeline as jev  # noqa: E402

DEFAULT_QUESTIONS = [
    "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
    "According to the provided corpus, which sailing event at the 2000 Summer Olympics had the highest number of competitors?",
    "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
    "Who won the gold medal in the event held at Richmond Olympic Oval on 14 February 2010?",
    "Who won the gold medal in the men's pole vault athletics event at the Summer Olympics held immediately before 2016?",
]

BOLD, DIM, GREEN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[0m"


def main(questions: list) -> int:
    conn = get_connection()
    # Build the candidate catalog up front. It is one read of the graph, but on
    # the first question it looks like an 8-second stall in front of an audience.
    print(f"{DIM}warming up: reading sport/Games/venue/event candidates from the graph...{RESET}")
    jev._build_catalog(conn)

    system_one = generation = 0
    latencies = []

    for i, question in enumerate(questions, 1):
        tracker = TokenTracker()
        started = time.time()
        result = jev.answer_question(conn, question, tracker=tracker, question_id=f"demo{i}")
        elapsed = time.time() - started
        latencies.append(elapsed)

        records = tracker.all_records()
        system_one += sum(1 for r in records if "typesafe" in r["note"])
        generation += sum(1 for r in records if "typesafe" not in r["note"] and r["call_type"] != "embedding")

        print(f"\n{BOLD}{i}. {question}{RESET}")
        for step in result.get("trace", []):
            if step.get("stage") == "select_operation_sport_games":
                print(f"   {DIM}plan   {RESET}{step['operation']} · {step['sport']} · {step['games']} "
                      f"{DIM}(confidence {step['operation_confidence']:.2f}){RESET}")
            elif step.get("stage", "").startswith("select_"):
                print(f"   {DIM}select {RESET}{step.get('event') or step.get('venue')} "
                      f"{DIM}({step.get('confidence', 0):.2f} from {step.get('options', '?')} candidates){RESET}")
            elif step.get("stage") == "error":
                print(f"   {DIM}error  {RESET}{step['error']}")
        print(f"   {GREEN}answer {RESET}{BOLD}{result.get('short_answer')}{RESET}")
        print(f"   {DIM}via {result['route']} in {elapsed:.2f}s, citing "
              f"{len(result.get('retrieved_doc_ids', []))} source document(s){RESET}")

    mean = sum(latencies) / len(latencies)
    print(f"\n{BOLD}{len(questions)} questions{RESET} · {system_one} System One calls · "
          f"{BOLD}{generation} generation-model calls{RESET} · {mean:.2f}s average")
    if generation:
        print(f"{DIM}(generation calls came from a fallback: the graph could not answer alone){RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or DEFAULT_QUESTIONS))
