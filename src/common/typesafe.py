"""TypeSafe System One (Jev) client: verify that an answer's claims are actually
supported by the evidence the pipeline retrieved.

Why this exists. Groundedness was judged by an LLM on a 20% sample, and that
judge proved unreliable -- on the first benchmark it gave accuracy 4-5 to 14 of
298 answers that were wrong, including several that said "the corpus does not
contain this". A System One model answers a narrower question instead: given
this claim and this evidence section, does the section support it, contradict
it, or say nothing? That is a typed judgment with a probability, so code can
gate on it rather than trusting prose.

Design follows TypeSafe's own guidance (https://docs.typesafe.ai):
  * one narrow, coherent judgment per question;
  * independent questions over the same state are asked TOGETHER, so verifying
    k claims costs one request, not k;
  * the model supplies semantic judgment; thresholds and policy stay in code
    (see CONFIDENCE_THRESHOLD and `verdicts_to_status`).

Entirely optional: with no TYPESAFE_API_KEY the checker reports itself disabled
and callers carry on unchanged, so cloning the repo without a key still works.
"""
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from src.common.token_tracker import CallRecord, TokenTracker, default_tracker

load_dotenv()

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
REQUEST_TIMEOUT_SEC = int(os.environ.get("TYPESAFE_TIMEOUT_SEC", 60))

# Claims per request. They share one state object and run as independent
# questions, so this is the batch factor: 8 claims cost one call, not eight.
MAX_CLAIMS_PER_REQUEST = 8

# Below this, a verdict is reported but flagged for human review rather than
# acted on. 0.8 is TypeSafe's cookbook starting point and is NOT validated on
# this corpus yet -- calibrate it against labelled data before trusting it.
CONFIDENCE_THRESHOLD = float(os.environ.get("TYPESAFE_CONFIDENCE_THRESHOLD", 0.8))

VERDICT_CRITERIA = {
    "supports": "The evidence states the claim or directly implies that it is true",
    "contradicts": "The evidence states the opposite of the claim or implies it is false",
    "says_nothing": "The evidence does not address what the claim asserts, either way",
}


class TypeSafeUnavailable(RuntimeError):
    """No API key configured -- callers should skip verification, not fail."""


def is_enabled() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


class _Retryable(RuntimeError):
    """429/529: the docs prescribe exponential backoff for both."""


@retry(retry=retry_if_exception_type(_Retryable),
       wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(4), reraise=True)
def _post(body: dict) -> dict:
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    if response.status_code in (429, 529):
        raise _Retryable(f"{response.status_code} from TypeSafe")
    response.raise_for_status()
    return response.json()


def check_claims(claims: list, *, tracker: Optional[TokenTracker] = None, question_id: Optional[str] = None,
                 pipeline: str = "") -> list:
    """Verify each (claim, evidence) pair. Returns one dict per input pair:

        {"claim", "verdict", "confidence", "probabilities", "grounded",
         "needs_review"}

    `verdict` is supports/contradicts/says_nothing; `grounded` is the
    probability that every detail in the claim is traceable to the evidence;
    `needs_review` marks confidence below CONFIDENCE_THRESHOLD.
    """
    if not claims:
        return []
    if not is_enabled():
        raise TypeSafeUnavailable("TYPESAFE_API_KEY is not set")

    results = []
    for start in range(0, len(claims), MAX_CLAIMS_PER_REQUEST):
        batch = claims[start:start + MAX_CLAIMS_PER_REQUEST]
        # One shared state, one pair of questions per claim. Question ids are for
        # code only and are not sent to the model, so each question repeats the
        # backticked state path it is about.
        state, questions = {}, {}
        for i, (claim, evidence) in enumerate(batch):
            state[f"claim_{i}"] = claim
            state[f"evidence_{i}"] = evidence
            questions[f"verdict_{i}"] = {
                "type": "choice",
                "instructions": f"Does `evidence_{i}` support the claim in `claim_{i}`?",
                "criteria": VERDICT_CRITERIA,
            }
            questions[f"grounded_{i}"] = {
                "type": "noul",
                "instructions": f"Is every factual detail in `claim_{i}` traceable to `evidence_{i}`?",
            }

        t0 = time.time()
        data = _post({"state": state, "model": MODEL, "questions": questions})
        latency = time.time() - t0

        usage = data.get("usage", {})
        (tracker or default_tracker).log(CallRecord(
            pipeline=pipeline,
            call_type="citation_check",
            model=data.get("model", MODEL),
            input_tokens=usage.get("input_tokens", 0) or 0,
            output_tokens=usage.get("output_tokens", 0) or 0,
            total_tokens=(usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
            latency_sec=latency,
            question_id=question_id,
            note=f"provider=typesafe claims={len(batch)}",
        ))

        answers = data.get("answers", {})
        for i, (claim, _evidence) in enumerate(batch):
            verdict = answers.get(f"verdict_{i}", {})
            grounded = answers.get(f"grounded_{i}", {})
            confidence = verdict.get("confidence", 0.0)
            results.append({
                "claim": claim,
                "verdict": verdict.get("choice"),
                "confidence": confidence,
                "probabilities": verdict.get("probabilities", {}),
                "grounded": grounded.get("noul"),
                "needs_review": confidence < CONFIDENCE_THRESHOLD,
            })
    return results


def verdicts_to_status(results: list) -> dict:
    """Policy lives here, not in the model: how a set of verdicts rolls up.

    Deliberately not a weighted average -- one contradicted claim is
    disqualifying however many others check out, which a mean would hide.
    """
    if not results:
        return {"status": "unverified", "verified": 0, "contradicted": 0, "unsupported": 0, "needs_review": 0}
    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in VERDICT_CRITERIA}
    review = sum(1 for r in results if r["needs_review"])
    if counts["contradicts"]:
        status = "contradicted"
    elif counts["says_nothing"]:
        status = "partly_unsupported"
    elif review:
        status = "needs_review"
    else:
        status = "verified"
    return {
        "status": status,
        "verified": counts["supports"],
        "contradicted": counts["contradicts"],
        "unsupported": counts["says_nothing"],
        "needs_review": review,
    }


if __name__ == "__main__":
    evidence = ("Athletics at the 2000 Summer Olympics - Women's 1500 metres: competitors=27, nations=19, "
                "venue=Stadium Australia, gold=Nouria Merah-Benida, silver=Violeta Szekely")
    checked = check_claims([
        ("Nouria Merah-Benida won the women's 1500 metres at the 2000 Summer Olympics.", evidence),
        ("Gabriela Szabo won the women's 1500 metres at the 2000 Summer Olympics.", evidence),
        ("The event was held at the Sydney Opera House.", evidence),
    ])
    for r in checked:
        print(f"  {r['verdict']:13s} conf={r['confidence']:.2f} grounded={r['grounded']} "
              f"review={r['needs_review']}  {r['claim'][:60]}")
    print(verdicts_to_status(checked))
