# Experiment 3: Text Generation using FLAN-T5 for a Question–Answer System

This project loads the pretrained `google/flan-t5-base` model and creates a simple question-answering interface using Gradio. It performs inference only: no dataset collection or model training is needed.

## Files
- `qa_gradio.py` — Python application with model loading and Gradio interface
- `flan_t5_qa_colab.ipynb` — Google Colab notebook
- `requirements.txt` — dependencies
- `README.md` — setup instructions
- `.gitignore` — common files to ignore

## Run in Google Colab
1. Upload/open `flan_t5_qa_colab.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU** (optional, but recommended).
3. Run the cells in order. The first run downloads pretrained model weights.
4. Enter questions in the Gradio interface and review generated answers.

## Run locally
Use Python 3.10 or 3.11 in a virtual environment:
```bash
pip install -r requirements.txt
python qa_gradio.py
```
Open the local Gradio URL printed in the terminal.

## Notes
- The model is pretrained; no training code or dataset is required.
- `google/flan-t5-base` can take time and memory to load. CPU is supported but may be slower.
- The model may generate inaccurate answers. Treat outputs as generated text, not guaranteed facts.
- The ZIP does not include model weights or an already-running hosted app; weights download on first run.

## GitHub
Recommended files to commit: `qa_gradio.py`, `flan_t5_qa_colab.ipynb`, `requirements.txt`, `README.md`, and a screenshot of the interface/output after you run it. Do not commit virtual environments or credentials.
