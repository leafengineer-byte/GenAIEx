"""Experiment 3 - FLAN-T5 question answering.

Pretrained weights download on first use, so the first request is slow.
"""
from ._models import require_model
from ._runner import run_task

QA_MODEL = "google/flan-t5-base"


def answer(question, max_tokens=128) -> str:
    """Answer a question with the pretrained FLAN-T5 model."""
    text = (question or "").strip()
    if not text:
        raise ValueError("Please enter a question.")

    require_model(QA_MODEL, 1.0)

    result = run_task("qa", "qa_answer", {"question": text, "max_tokens": int(max_tokens)})
    return result["answer"]
