"""Per-call token/cost logging shared across all three pipelines.

Every LLM call (generation, embedding, judge) should route through
`TokenTracker.log(...)` so the metrics dashboard can later break down
token usage by pipeline, call_type, and question -- see the eval
methodology this project follows: log input/output/context/thinking
tokens separately, not just a single "tokens used" number.
"""
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CallRecord:
    pipeline: str            # "rag" | "graphrag" | "agentic"
    call_type: str           # "generation" | "embedding" | "judge" | "extraction"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0  # tokens attributable to retrieved/injected context, subset of input_tokens
    latency_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)
    question_id: Optional[str] = None
    note: str = ""


class TokenTracker:
    """Thread-safe accumulator. One instance per benchmark run (or per question)."""

    def __init__(self):
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    def log(self, record: CallRecord) -> None:
        with self._lock:
            self._records.append(record)

    def records_for(self, question_id: str) -> list[CallRecord]:
        return [r for r in self._records if r.question_id == question_id]

    def totals_for(self, question_id: str) -> dict:
        recs = self.records_for(question_id)
        return {
            "num_llm_calls": len(recs),
            "total_input_tokens": sum(r.input_tokens for r in recs),
            "total_output_tokens": sum(r.output_tokens for r in recs),
            "total_thoughts_tokens": sum(r.thoughts_tokens for r in recs),
            "total_context_tokens": sum(r.context_tokens for r in recs),
            "total_tokens": sum(r.total_tokens for r in recs),
            "total_latency_sec": sum(r.latency_sec for r in recs),
        }

    def all_records(self) -> list[dict]:
        with self._lock:
            return [asdict(r) for r in self._records]

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self.all_records():
                f.write(json.dumps(r) + "\n")


# A process-wide default tracker; pipelines can also create their own per-run instance.
default_tracker = TokenTracker()
