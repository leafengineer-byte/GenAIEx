"""Question-answer system using pretrained google/flan-t5-base and Gradio."""
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def answer_question(question, max_tokens=128):
    question = (question or "").strip()
    if not question:
        return "Please enter a question."
    prompt = f"Answer the following question clearly and concisely:\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Enter your question", placeholder="What is artificial intelligence?", lines=3),
        gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Maximum answer tokens")
    ],
    outputs=gr.Textbox(label="Generated answer", lines=6),
    title="FLAN-T5 Question–Answer System",
    description="Ask a question and generate an answer using a pretrained Transformer model.",
    examples=[
        ["What is machine learning?", 128],
        ["Explain the attention mechanism in simple words.", 128],
        ["What are the applications of generative AI?", 128]
    ]
)

if __name__ == "__main__":
    demo.launch()
