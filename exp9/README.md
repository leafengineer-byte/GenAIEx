# Experiment 9: Personalized AI Assistant using LLM, RAG, and LoRA/QLoRA

This starter project demonstrates three stages:
1. Base open-source language model response
2. Retrieval-Augmented Generation (RAG) over user-provided documents
3. Optional parameter-efficient fine-tuning using LoRA/QLoRA

## Important compute note
The included default model is `Qwen/Qwen2.5-0.5B-Instruct`, chosen as a small teaching example. A GPU is recommended. Model availability, dependency compatibility, VRAM and access terms can vary. The notebook's fine-tuning section is optional and may need a compatible GPU/runtime; it is not required for the base/RAG demo.

This ZIP includes source code and a notebook, not pretrained model weights, a vector index, private documents, or a fine-tuned adapter. These are downloaded or created when you run it.

## Files
- `personalized_ai_assistant_colab.ipynb` — step-by-step Colab notebook
- `assistant_rag.py` — reusable document chunking, embedding, retrieval and generation functions
- `finetune_lora.py` — optional LoRA fine-tuning template
- `requirements.txt` — dependencies
- `README.md` — instructions
- `.gitignore` — ignores private documents, model weights and outputs

## Run in Google Colab
1. Upload/open the notebook in Colab and select a GPU runtime if available.
2. Run setup/install cells.
3. Upload a small PDF or TXT document you are permitted to use. Avoid confidential personal data.
4. Run the base model question, then the RAG question and compare answers.
5. Optional: prepare a small JSONL training dataset and run the LoRA example only if you have sufficient GPU memory.

## What the RAG code does
- Extracts PDF or TXT text
- Splits text into overlapping chunks
- Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`
- Uses FAISS nearest-neighbor search to retrieve relevant chunks
- Adds retrieved chunks to the prompt and asks the LLM to answer from context

## Fine-tuning
`finetune_lora.py` is an educational starter template for LoRA using PEFT/TRL. Fine-tuning is optional and depends on the selected model, dataset format, library versions and GPU memory. RAG and fine-tuning solve different problems; compare base and RAG outputs first, then fine-tune if resources allow.

## Security
Do not upload `.env`, API keys, private documents, raw datasets or large model checkpoints to GitHub. Only use documents and training examples you are authorized to use.
