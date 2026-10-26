"""Minimal RAG utilities for PDF/TXT files and a Hugging Face causal LM."""
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"


def read_document(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Supported formats: PDF, TXT, MD")


def chunk_text(text, chunk_size=700, overlap=120):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class RAGAssistant:
    def __init__(self, model_name=DEFAULT_LLM):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model.to(self.device)
        self.index = None
        self.chunks = []

    def add_document(self, path):
        text = read_document(path)
        new_chunks = chunk_text(text)
        if not new_chunks:
            raise ValueError("No extractable text found in document.")
        embeddings = self.embedder.encode(new_chunks, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def retrieve(self, question, k=3):
        if self.index is None or not self.chunks:
            return []
        q = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        _, indices = self.index.search(np.asarray(q, dtype="float32"), min(k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]

    def answer(self, question, use_rag=True, max_new_tokens=180):
        context_chunks = self.retrieve(question) if use_rag else []
        context = "\n\n".join(context_chunks) or "No document context supplied."
        prompt = (
            "You are a helpful personal AI assistant. If document context is provided, "
            "prioritize it and say when the answer is not present. Do not invent facts.\n\n"
            f"Document context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                         temperature=0.7, top_p=0.9,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True), context_chunks
