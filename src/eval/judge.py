"""LLM-as-judge scoring: accuracy, completeness, groundedness on a 1-5 scale,
each with a short chain-of-thought reason logged for auditability/demo
credibility. Uses JUDGE_MODEL -- deliberately a different/stronger model
than GEN_MODEL (see .env) to reduce self-preference bias, per the eval
methodology research this project follows.

Accuracy/completeness are scored against the REFERENCE answer (we have gold
answers for eval_public.jsonl); groundedness is scored against the actual
retrieved CONTEXT the pipeline used, not the reference -- a pipeline can be
faithful to bad context and still be wrong, and unfaithful to good context.
"""
import os

from pydantic import BaseModel

from src.common.llm import generate

JUDGE_SYSTEM_INSTRUCTION = """You are a strict, consistent evaluator of question-answering systems.
Score only what is stated in the candidate answer; do not reward length or style."""

JUDGE_PROMPT_TEMPLATE = """Question: {question}

Reference answer (ground truth): {reference_answer}

Retrieved context (evidence given to the system):
{context}

System's answer: {candidate_answer}

Score three dimensions, 1-5 each:
1. ACCURACY: Does the answer's core claim match the reference answer? 5=fully correct, 1=contradicts.
2. COMPLETENESS: Does the answer cover all parts of the reference answer (especially for multi-part /
   multi-hop questions)? 5=all parts covered, 1=missing most parts.
3. GROUNDEDNESS: Is every claim in the answer traceable to the retrieved context (not just to the
   reference)? 5=fully supported by context, 1=fabricated/unsupported by any retrieved evidence.

Give a brief reason for each score, then the scores."""


class JudgeScore(BaseModel):
    accuracy: int
    completeness: int
    groundedness: int
    reasoning: str


def score_answer(question: str, reference_answer: str, context: str, candidate_answer: str, tracker=None, question_id=None, pipeline: str = "") -> JudgeScore:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        context=context[:12000],  # cap context length fed to the judge to bound its own cost
        candidate_answer=candidate_answer,
    )
    result, _rec = generate(
        prompt,
        model=os.environ.get("JUDGE_MODEL"),  # was missing -- generate() silently fell back to
        # GEN_MODEL without this, defeating the different-model self-preference-bias mitigation.
        system_instruction=JUDGE_SYSTEM_INSTRUCTION,
        response_schema=JudgeScore,
        pipeline=pipeline,
        call_type="judge",
        tracker=tracker,
        question_id=question_id,
    )
    return result
