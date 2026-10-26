"""Experiment 9 - personalised assistant with retrieval-augmented generation.

The LLM and embedding model are already present in the local Hugging Face cache,
so this runs offline. Documents are extracted in-process and passed to the worker
along with each question, keeping the subprocess stateless.
"""
import io

from ._runner import run_task

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def extract_text(filename, data) -> str:
    """Extract plain text from an uploaded PDF / TXT / MD file."""
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Supported formats: PDF, TXT, MD")


def ask(question, document="", use_rag=True, max_tokens=180) -> dict:
    """Answer a question, optionally grounded in the supplied document text."""
    text = (question or "").strip()
    if not text:
        raise ValueError("Please enter a question.")

    return run_task(
        "assistant",
        "assistant_answer",
        {
            "question": text,
            "document": document or "",
            "use_rag": bool(use_rag),
            "max_tokens": int(max_tokens),
        },
    )
