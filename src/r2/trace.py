"""The per-question trace, and the only route to TigerGraph or an LLM.

Round 2 checks the trace against itself: the LLM steps' tokens must add up to
the answer's `input_tokens` and `output_tokens`, the step timings must roughly
add up to `latency_ms`, and every TigerGraph step must appear in the submitted
query log with a matching timestamp. A trace assembled after the fact will not
survive that, and the brief says so outright -- "written by your code as it
runs (not reconstructed afterwards)".

So the recorder owns the calls rather than observing them. `run_query` is the
only way the agent reaches the database and `generate` is the only way it
reaches a model; both time themselves, append their own step, and in the
database's case write the query log line. A step cannot be forgotten, and the
totals cannot drift from the steps, because nothing else is able to make the
call in the first place.

Cost is computed from a published per-token rate rather than guessed, and the
rate is in the environment so the figure in metrics.json can be audited
against the model actually used.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional


def _rate(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# USD per million tokens. Defaults match gemini-3.1-flash-lite's published
# rate; override in .env if the model changes so the cost stays honest.
INPUT_USD_PER_MTOK = _rate("INPUT_USD_PER_MTOK", 0.10)
OUTPUT_USD_PER_MTOK = _rate("OUTPUT_USD_PER_MTOK", 0.40)


def _count_rows(result: Any) -> int:
    """Rows a TigerGraph result carries, across every PRINT in the query.

    An installed query returns one object per PRINT statement; a PRINT can be a
    vertex set (a list), a map accumulator (a dict), or a scalar. Counting the
    list lengths and dict sizes gives the number the brief asks for -- "the
    number of rows or vertices returned" -- without pretending a scalar is a row.
    """
    total = 0
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict):
                for value in block.values():
                    if isinstance(value, list):
                        total += len(value)
                    elif isinstance(value, dict):
                        total += len(value)
                    else:
                        total += 1
            elif isinstance(block, list):
                total += len(block)
    elif isinstance(result, dict):
        total += len(result)
    return total


def _vertex_tuples(params: dict) -> dict:
    """Wrap vertex-id lists as 1-tuples.

    pyTigerGraph sends a bare value for a VERTEX<T> parameter as a plain id,
    which TigerGraph 4.2 rejects ("'id' is not found in the VERTEX parameter")
    before silently retrying over GET. Round 1 hit this and the fix is the
    same: a VERTEX parameter is a 1-tuple of its id.
    """
    out = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            out[key] = [(v,) for v in value]
        else:
            out[key] = value
    return out


class QueryLog:
    """Append-only TigerGraph request log -- submission deliverable 5.

    One line per database call, with a wall-clock timestamp that matches the
    corresponding trace step, so judges can line the two up.
    """

    def __init__(self, path: str | Path, fresh: bool = False):
        """`fresh` starts a new log for a new run.

        Appending is right for a crashed run -- the lines already written stay
        usable. It is wrong for a submitted one: deliverable 5 is "your query
        log from both runs" and its timestamps have to line up with the traces
        in the answer file beside it. Appending quietly left the previous
        run's calls in the same file, so the log described more work than the
        answers did. A run truncates its own log exactly as it truncates its
        own answer file; the old one is kept with a timestamp rather than
        deleted.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists() and self.path.stat().st_size:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            self.path.rename(self.path.with_suffix(f".{stamp}.bak"))
        self._handle = open(self.path, "w" if fresh else "a", encoding="utf-8")

    def write(self, **row) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()  # a crashed run must still leave a usable log

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


class Trace:
    def __init__(self, qid: str, conn=None, query_log: Optional[QueryLog] = None,
                 model: Optional[str] = None):
        self.qid = qid
        self.conn = conn
        self.query_log = query_log
        self.model = model
        self.steps: list[dict] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.tg_queries = 0
        self._t0 = time.perf_counter()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- the two instrumented call paths -----------------------------------

    def run_query(self, name: str, params: Optional[dict] = None, tool: Optional[str] = None):
        """Run an installed GSQL query, timed, traced and logged."""
        params = _vertex_tuples(params or {})
        started_wall = time.time()
        t = time.perf_counter()
        error = None
        try:
            result = self.conn.runInstalledQuery(name, params)
        except Exception as exc:  # a failed query is evidence, not a crash
            result, error = [], str(exc)
        ms = int((time.perf_counter() - t) * 1000)
        rows = _count_rows(result)
        self.tg_queries += 1

        step = {"step": len(self.steps) + 1, "tool": tool or name, "tigergraph": True,
                "query": name, "params": params, "rows": rows, "ms": ms}
        if error:
            step["error"] = error
        self.steps.append(step)
        if self.query_log:
            self.query_log.write(ts=time.strftime("%Y-%m-%dT%H:%M:%S",
                                                  time.gmtime(started_wall)) + "Z",
                                 qid=self.qid, query=name, params=params,
                                 rows=rows, ms=ms, error=error)
        return result

    def generate(self, prompt: str, *, tool: str, **kwargs):
        """Call the model, accumulating tokens onto this question's totals."""
        from src.common.llm import generate
        t = time.perf_counter()
        result, usage = generate(prompt, model=kwargs.pop("model", self.model), **kwargs)
        ms = int((time.perf_counter() - t) * 1000)

        # Provider usage keys vary, and `generate` hands back a CallRecord
        # dataclass rather than a dict -- reading it as a dict silently
        # recorded zero tokens for every call, which would have shipped two
        # answer files whose totals were all 0 and failed the brief's own
        # "tokens add up" check on every line. Read attributes and keys both.
        def pick(*names):
            for n in names:
                v = usage.get(n) if isinstance(usage, dict) else getattr(usage, n, None)
                if v is not None:
                    return int(v)
            return 0

        inp = pick("input_tokens", "prompt_tokens", "prompt_token_count")
        out = pick("output_tokens", "completion_tokens", "candidates_token_count")
        # Thinking tokens are billed as output and must be included, or the
        # per-question totals will not reconcile with the provider's bill.
        out += pick("thoughts_tokens", "thinking_tokens", "thoughts_token_count")

        self.input_tokens += inp
        self.output_tokens += out
        self.llm_calls += 1
        self.steps.append({"step": len(self.steps) + 1, "tool": tool,
                           "tigergraph": False, "input_tokens": inp,
                           "output_tokens": out, "ms": ms})
        return result

    # --- plain stages -------------------------------------------------------

    def stage(self, tool: str, fn: Callable[[], Any]) -> Any:
        """A step that neither queries the graph nor calls a model."""
        t = time.perf_counter()
        result = fn()
        self.steps.append({"step": len(self.steps) + 1, "tool": tool,
                           "tigergraph": False,
                           "ms": int((time.perf_counter() - t) * 1000)})
        return result

    # --- output -------------------------------------------------------------

    @property
    def latency_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    @property
    def used_tigergraph(self) -> bool:
        return any(s["tigergraph"] for s in self.steps)

    def cost_usd(self) -> float:
        return (self.input_tokens / 1e6 * INPUT_USD_PER_MTOK
                + self.output_tokens / 1e6 * OUTPUT_USD_PER_MTOK)

    def answer(self, status: str, answer, explanation: str, citations: list) -> dict:
        """The submission record for this question, in the brief's format."""
        # Citations must be doc_ids, deduplicated, order kept stable so a
        # re-run produces an identical file.
        seen, cites = set(), []
        for c in citations:
            if c and c not in seen:
                seen.add(c)
                cites.append(c)
        return {
            "qid": self.qid,
            # Deliverable 5 says the query log's timestamps should match the
            # traces. A trace step records only how long it took, so without
            # a wall-clock anchor on the record there is nothing to match it
            # against. These two make the alignment checkable: every log line
            # for this qid falls inside this window.
            "started_at": self.started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "answer": answer,
            "explanation": explanation,
            "citations": cites,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "trace": self.steps,
        }


def verify(record: dict) -> list[str]:
    """Self-check one answer record against the brief's stated constraints.

    Run over both answer files before submitting: the judges recompute every
    number from these files and say a mismatch counts against repeatability,
    so finding it ourselves is free and finding it afterwards is not.
    """
    problems = []
    steps = record.get("trace") or []
    if [s.get("step") for s in steps] != list(range(1, len(steps) + 1)):
        problems.append("trace steps are not numbered 1..n in order")

    inp = sum(s.get("input_tokens", 0) for s in steps)
    out = sum(s.get("output_tokens", 0) for s in steps)
    if inp != record.get("input_tokens"):
        problems.append(f"input_tokens {record.get('input_tokens')} != step sum {inp}")
    if out != record.get("output_tokens"):
        problems.append(f"output_tokens {record.get('output_tokens')} != step sum {out}")

    step_ms = sum(s.get("ms", 0) for s in steps)
    if record.get("latency_ms", 0) + 1 < step_ms:
        problems.append(f"latency_ms {record.get('latency_ms')} < step time {step_ms}")

    for s in steps:
        if s.get("tigergraph"):
            if not s.get("query"):
                problems.append(f"step {s.get('step')} queried TigerGraph without a query name")
            if "rows" not in s:
                problems.append(f"step {s.get('step')} has no row count")
        elif "input_tokens" in s and "ms" not in s:
            problems.append(f"step {s.get('step')} is an LLM step with no timing")

    if record.get("status") not in ("answered", "partial", "not_in_corpus"):
        problems.append(f"status {record.get('status')!r} is not one of the three allowed values")
    if record.get("status") == "answered" and not record.get("citations"):
        problems.append("answered with no citations")
    if not any(s.get("tigergraph") for s in steps):
        problems.append("no TigerGraph step (the brief says these get a closer look)")
    return problems
