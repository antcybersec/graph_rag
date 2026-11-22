"""Answer one question from the command line and show the work.

    python -m src.r2.ask "Who directed Cocoon?"
    python -m src.r2.ask --json "Which nation won the most gold in 2024 athletics?"

This is the path the judging call uses. It takes the question as an argument
rather than reading a question file, so there is no question id anywhere in it
and nothing to look up: the same code answers an unseen question and a question
from the public set, which is what rule 4 asks for.

`--json` prints the record exactly as `src.r2.run` writes it to the answer
files, so an answer produced here can be compared line for line with a
submitted one.
"""
import argparse
import json
import sys

from src.r2.agent import Agent
from src.r2.conn import describe, get_connection
from src.r2.trace import QueryLog


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question", nargs="+", help="the question, in quotes")
    ap.add_argument("--json", action="store_true",
                    help="print the full record instead of a readable summary")
    ap.add_argument("--log", default="data/results/r2/ask_querylog.jsonl",
                    help="where to append the GSQL query log")
    ap.add_argument("--qid", default="ask-1", help="an id for the trace")
    args = ap.parse_args(argv)

    question = " ".join(args.question).strip()
    if not question:
        print("nothing to answer", file=sys.stderr)
        return 2

    conn = get_connection()
    agent = Agent(conn, query_log=QueryLog(args.log))
    record = agent.answer(args.qid, question)

    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    print(f"\n{describe()}")
    print(f"\nQUESTION  {question}")
    print(f"STATUS    {record['status']}")
    print(f"\nANSWER    {record['answer']}")
    if record.get("explanation"):
        print(f"\nWHY       {record['explanation']}")
    if record.get("citations"):
        print(f"\nCITED     {len(record['citations'])} document(s): "
              + ", ".join(record["citations"][:12])
              + (" ..." if len(record["citations"]) > 12 else ""))
    else:
        print("\nCITED     nothing -- no retrieved document supported an answer")
    if record.get("note"):
        print(f"\nGATE      {record['note']}")

    print("\nHOW IT GOT THERE")
    for step in record.get("trace", []):
        where = "TigerGraph" if step.get("tigergraph") else (step.get("provider") or "model")
        tok = ""
        if step.get("input_tokens") or step.get("output_tokens"):
            tok = f"  {step.get('input_tokens', 0)} in / {step.get('output_tokens', 0)} out"
        print(f"  {step['step']:2d}  {step.get('tool', '?'):32s} {where:12s}{tok}")
    print(f"\n{record.get('latency_ms', 0) / 1000:.1f}s, "
          f"{record.get('input_tokens', 0)} input and "
          f"{record.get('output_tokens', 0)} output tokens in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
