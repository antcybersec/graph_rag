"""Exercise every route against the live graph with no model call at all.

Why this is separate from the tests. `tests/test_r2.py` pins pure functions and
needs nothing; this needs the database, and what it checks is the other half of
the system: that each of the nine routes still returns rows from the graph it
was built against. A route that has quietly stopped matching -- a renamed key,
a query reinstalled with a changed signature, a schema reload that dropped an
edge -- answers `not_in_corpus` perfectly politely, and the only sign is a
score at the end of a run that costs 130 requests to produce.

The plans here are built by hand, so the planner never runs. Two routes still
reach a model inside gather, and pretending otherwise would be the kind of
claim this file exists to catch: `event_series` and `evidence` resolve a
contest name by asking the model to choose among candidates the graph supplied,
and `film` and `office` do the same when a name is ambiguous. The model calls
are counted and printed per route, and `--local` puts them on a local Ollama
model so the whole check needs no API key and no quota -- which is how it was
run the first time, fifteen hours into a spent daily limit.

The question text beside each plan is a label for a human reading the output;
nothing reads it. Rule 4 is not at issue because no code here is on the
answering path -- `run.sh` does not import this module.

    python -m src.r2.smoke                  # every route
    python -m src.r2.smoke --local          # selections on Ollama, no API key
    python -m src.r2.smoke --route film
"""
import argparse
import json
import sys
import time

from src.r2.agent import Agent, Plan
from src.r2.conn import describe, get_connection
from src.r2.trace import QueryLog, Trace

# route -> (a plan that should find something, what it is for, what must come back)
CASES: list[tuple[Plan, str, str]] = [
    (Plan(route="sport_games", sport_key="athletics", season="Summer", years=[2024],
          medal="gold"),
     "which nation won the most gold in athletics at one Games", "tallies"),
    (Plan(route="multi_medal_games", season="Summer", years=[2024]),
     "who won more than one medal at one Games", "tallies"),
    (Plan(route="repeat_champions", sport_key="athletics", season="Summer"),
     "who won the same event at more than one Games", "tallies"),
    (Plan(route="event_series", sport_key="athletics",
          event_phrase="Men's 100 metres", season="Summer", years=[2024]),
     "one contest across its editions", "citable"),
    (Plan(route="athlete", athlete="Usain Bolt"),
     "one athlete's medals", "citable"),
    (Plan(route="film", film="Cocoon"),
     "one film by name", "citable"),
    (Plan(route="film"),
     "every film, compared", "tallies"),
    (Plan(route="office", person="Angus King"),
     "offices and successions", "citable"),
    (Plan(route="evidence", event_phrase="100 metres", sport_key="athletics",
          years=[2028], wants_evidence=True),
     "records that disagree", "citable"),
    (Plan(route="text", entities=["Cocoon"]),
     "a fact that lives only in prose", "citable"),
]


def run(agent: Agent, log: QueryLog, plan: Plan, label: str, needs: str) -> dict:
    tr = Trace(f"smoke-{plan.route}", agent.conn, log)
    t0 = time.time()
    try:
        ev = agent.gather(tr, label, plan)
    except Exception as exc:
        return {"route": plan.route, "label": label, "ok": False,
                "why": f"{type(exc).__name__}: {exc}", "secs": time.time() - t0}
    calls = len([s for s in tr.steps if not s.get("tigergraph")])
    cited, tallies = len(ev.get("citable") or {}), len(ev.get("tallies") or [])
    got = cited if needs == "citable" else tallies
    rows = sum(s.get("rows", 0) for s in tr.steps if s.get("tigergraph"))
    errors = [s for s in tr.steps if s.get("error")]
    return {"route": plan.route, "label": label,
            "ok": got > 0 and not errors,
            "why": ("a query failed: " + errors[0]["error"][:160]) if errors
                   else ("" if got > 0 else
                         (f"the queries returned {rows} rows and the selection "
                          f"rejected all of them -- the route works and the model "
                          f"choosing among its candidates did not"
                          if rows and calls else f"no {needs} came back")),
            "cited": cited, "tallies": tallies, "rows": rows,
            "queries": len([s for s in tr.steps if s.get("tigergraph")]),
            "model_calls": calls,
            "secs": time.time() - t0,
            "block_chars": len(agent.render(ev))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", help="run only this route")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--local", action="store_true",
                    help="run the selection calls on a local Ollama model "
                         "(OFFLINE_MODEL, default qwen3:8b) so this needs no key")
    ap.add_argument("--log", default="data/results/r2/smoke_querylog.jsonl")
    args = ap.parse_args(argv)

    cases = [c for c in CASES if not args.route or c[0].route == args.route]
    if not cases:
        print(f"no case for route {args.route!r}; known: "
              + ", ".join(sorted({c[0].route for c in CASES})), file=sys.stderr)
        return 2

    conn = get_connection()
    agent = Agent(conn, query_log=QueryLog(args.log, fresh=True))
    if args.local:
        import os
        agent.small_provider = "local"
        agent.small_model = os.environ.get("OFFLINE_MODEL", "qwen3:8b")
        os.environ.setdefault("OLLAMA_NUM_CTX", "8192")
        os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")
    print(f"{describe()}  selections on "
          f"{agent.small_provider or 'the default provider'}\n")
    results = [run(agent, agent.query_log, *c) for c in cases]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'route':20s} {'what it is for':42s} {'cited':>5s} {'math':>4s} "
              f"{'rows':>6s} {'q':>2s} {'llm':>3s} {'chars':>6s} {'s':>5s}")
        for r in results:
            mark = "ok  " if r["ok"] else "FAIL"
            print(f"{mark} {r['route']:15s} {r['label'][:42]:42s} "
                  f"{r.get('cited', 0):5d} {r.get('tallies', 0):4d} "
                  f"{r.get('rows', 0):6d} {r.get('queries', 0):2d} "
                  f"{r.get('model_calls', 0):3d} "
                  f"{r.get('block_chars', 0):6d} {r['secs']:5.1f}")
            if r["why"]:
                print(f"     ^ {r['why']}")

    bad = [r for r in results if not r["ok"]]
    calls = sum(r.get("model_calls", 0) for r in results)
    print(f"\n{len(results) - len(bad)}/{len(results)} routes returned evidence. "
          f"{sum(r.get('queries', 0) for r in results)} GSQL queries and {calls} "
          f"model call(s) -- the planner never ran, and the calls that remain are "
          f"the selections named in this module's docstring.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
