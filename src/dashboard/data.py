"""Loading + aggregation for the three-pipeline benchmark dashboard.

Reads data/results/benchmark_results.jsonl (one row per (qid, pipeline)) and
turns it into a tidy DataFrame. The file is append-only and checkpointed by
run_benchmark.py, so it carries two kinds of noise this module filters out:

  * error rows -- a pipeline or its dependency raised, so the row is just
    {qid, pipeline, qtype, error}. These are deliberately NOT treated as done
    by the benchmark's checkpoint (see run_benchmark.load_done_keys), which is
    why a failed-then-retried pair appears more than once in the file.
  * superseded successes -- if a pair were ever written twice successfully, the
    LAST occurrence is the current one.

Deliberately free of any streamlit import so it stays testable from a plain
`python -c` and so caching policy lives in the app layer.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd

PIPELINES = ["rag", "graphrag", "agentic"]
PIPELINE_LABELS = {
    "rag": "RAG",
    "graphrag": "GraphRAG",
    "agentic": "Agentic GraphRAG",
}
PIPELINE_ORDER = [PIPELINE_LABELS[p] for p in PIPELINES]

# Canonical display order; any qtype not listed is appended alphabetically so a
# new question type in the eval set shows up instead of being silently dropped.
QTYPE_ORDER = ["lookup", "aggregation", "multi_hop", "superlative", "temporal"]

JUDGE_DIMS = ["accuracy", "completeness", "groundedness"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "completeness": "Completeness",
    "groundedness": "Groundedness",
    "judge_mean": "Mean judge score",
    "doc_precision": "Doc precision",
    "doc_recall": "Doc recall",
    "latency_sec": "Pipeline latency (s)",
    "total_latency_sec": "End-to-end latency (s)",
    "total_tokens": "Total tokens",
    "num_llm_calls": "LLM calls",
}

_ROOT_MARKERS = ("requirements.txt", "pyproject.toml", ".git")
RESULTS_RELPATH = os.path.join("data", "results", "benchmark_results.jsonl")

_NUMERIC_FIELDS = [
    "doc_precision",
    "doc_recall",
    "latency_sec",
    "total_latency_sec",
    "num_llm_calls",
    "total_input_tokens",
    "total_output_tokens",
    "total_thoughts_tokens",
    "total_context_tokens",
    "total_tokens",
]


def project_root() -> Path:
    """Repo root, resolved from this file rather than the cwd.

    `streamlit run src/dashboard/app.py` can be launched from anywhere, so the
    results path must never depend on where the process started.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return here.parents[2]  # src/dashboard/data.py -> src/dashboard -> src -> root


def results_path() -> Path:
    return project_root() / RESULTS_RELPATH


def _number(value):
    """Coerce to float, mapping anything non-numeric (incl. None) to NaN."""
    if isinstance(value, bool) or value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _judge_score(judge, dim):
    if not isinstance(judge, dict):
        return math.nan
    return _number(judge.get(dim))


def _flatten(row: dict) -> dict:
    judge = row.get("judge") or {}
    scores = {dim: _judge_score(judge, dim) for dim in JUDGE_DIMS}
    present = [v for v in scores.values() if not math.isnan(v)]
    retrieved = row.get("retrieved_doc_ids") or []
    gold = row.get("gold_doc_ids") or []

    flat = {
        "qid": row["qid"],
        "pipeline": row["pipeline"],
        "pipeline_label": PIPELINE_LABELS.get(row["pipeline"], row["pipeline"]),
        "qtype": row.get("qtype") or "unknown",
        "question": row.get("question") or "",
        "reference_answer": row.get("reference_answer") or "",
        "answer": row.get("answer") or "",
        "judge_reasoning": judge.get("reasoning") if isinstance(judge, dict) else "",
        "judge_mean": sum(present) / len(present) if present else math.nan,
        "n_retrieved": len(retrieved) if isinstance(retrieved, list) else 0,
        "n_gold": len(gold) if isinstance(gold, list) else 0,
    }
    flat.update(scores)
    for field in _NUMERIC_FIELDS:
        flat[field] = _number(row.get(field))
    return flat


def load_results(path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (tidy DataFrame of valid rows, load stats).

    A row is valid only if it has no truthy "error" key and carries the fields
    a comparison needs. Error rows are missing most fields, so they are dropped
    before any column access rather than defended against downstream.
    """
    path = Path(path) if path is not None else results_path()
    stats = {
        "path": str(path),
        "total_lines": 0,
        "error_rows": 0,
        "malformed_rows": 0,
        "superseded_duplicates": 0,
        "valid_rows": 0,
    }

    if not path.exists():
        stats["missing_file"] = True
        return pd.DataFrame(), stats

    by_key: dict[tuple[str, str], dict] = {}
    successes_read = 0

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats["total_lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_rows"] += 1
                continue
            if not isinstance(row, dict) or not row.get("qid") or not row.get("pipeline"):
                stats["malformed_rows"] += 1
                continue
            if row.get("error"):
                stats["error_rows"] += 1
                continue
            if "answer" not in row or "judge" not in row:
                # A success-shaped row that never got scored is not comparable.
                stats["malformed_rows"] += 1
                continue
            successes_read += 1
            by_key[(row["qid"], row["pipeline"])] = row  # last write wins

    stats["superseded_duplicates"] = successes_read - len(by_key)
    stats["valid_rows"] = len(by_key)
    stats["excluded_rows"] = (
        stats["error_rows"] + stats["malformed_rows"] + stats["superseded_duplicates"]
    )

    if not by_key:
        return pd.DataFrame(), stats

    frame = pd.DataFrame.from_records([_flatten(r) for r in by_key.values()])
    frame["pipeline_label"] = pd.Categorical(
        frame["pipeline_label"], categories=pipeline_order(frame), ordered=True
    )
    frame["qtype"] = pd.Categorical(frame["qtype"], categories=qtype_order(frame), ordered=True)
    frame = frame.sort_values(["qid", "pipeline_label"]).reset_index(drop=True)

    stats["unique_questions"] = int(frame["qid"].nunique())
    stats["rows_per_pipeline"] = (
        frame["pipeline_label"].value_counts().sort_index().to_dict()
    )
    return frame, stats


def pipeline_order(frame: pd.DataFrame) -> list[str]:
    """Known pipelines first, in their canonical order, then any newcomers."""
    present = set(frame["pipeline_label"].astype(str))
    ordered = [p for p in PIPELINE_ORDER if p in present]
    return ordered + sorted(present - set(ordered))


def qtype_order(frame: pd.DataFrame) -> list[str]:
    present = set(frame["qtype"].astype(str))
    ordered = [q for q in QTYPE_ORDER if q in present]
    return ordered + sorted(present - set(ordered))


def qtype_label(qtype: str) -> str:
    return str(qtype).replace("_", " ")


SUMMARY_METRICS = [
    "accuracy",
    "completeness",
    "groundedness",
    "doc_precision",
    "doc_recall",
    "latency_sec",
    "total_latency_sec",
    "total_tokens",
    "num_llm_calls",
]


def pipeline_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean of every headline metric, one row per pipeline."""
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("pipeline_label", observed=True)
    summary = grouped[SUMMARY_METRICS].mean()
    summary.insert(0, "questions", grouped.size())
    summary = summary.rename(columns=METRIC_LABELS)
    summary.index.name = "Pipeline"
    return summary.reset_index()


def long_judge_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (qid, pipeline, judge dimension) -- the shape charts want."""
    if frame.empty:
        return pd.DataFrame(columns=["pipeline_label", "qtype", "dimension", "score"])
    melted = frame.melt(
        id_vars=["qid", "pipeline_label", "qtype"],
        value_vars=JUDGE_DIMS,
        var_name="dimension",
        value_name="score",
    )
    melted["dimension"] = pd.Categorical(
        melted["dimension"].map(METRIC_LABELS),
        categories=[METRIC_LABELS[d] for d in JUDGE_DIMS],
        ordered=True,
    )
    return melted.dropna(subset=["score"])


def mean_by(frame: pd.DataFrame, metrics: list[str], by: list[str]) -> pd.DataFrame:
    """Mean of `metrics` grouped by `by`, as a flat frame (no MultiIndex)."""
    if frame.empty:
        return pd.DataFrame()
    return frame.groupby(by, observed=True)[metrics].mean().reset_index()
