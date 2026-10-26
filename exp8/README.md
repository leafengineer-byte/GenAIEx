# Experiment 8: Text-to-Music Generation with MusicGen

This project uses a **pretrained MusicGen model** to generate music from natural-language prompts. It does not train a model or require a training dataset.

## Contents
- `musicgen_colab.ipynb` — Google Colab notebook
- `generate_music.py` — reusable inference script
- `requirements.txt` — dependencies
- `.gitignore` — avoids committing credentials and generated audio
- `outputs/` — generated audio files are saved here

## Run in Google Colab
1. Open `musicgen_colab.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU**.
3. Run the cells in order. The first run downloads pretrained model weights, so allow time and disk space.
4. Enter a text prompt and generate audio.
5. Play the audio in the notebook and download the WAV file.

## Run locally
Python 3.10 or 3.11 is recommended. Install PyTorch for your system from the official PyTorch selector first, then:
```bash
pip install -r requirements.txt
python generate_music.py --prompt "Peaceful instrumental piano with soft flute, warm ambient textures, slow tempo" --duration 8
```

## Model and resource notes
The notebook uses `facebook/musicgen-small` through AudioCraft. The first run downloads model weights and needs internet access. GPU is strongly recommended. CPU inference may be slow or memory-intensive. The small model is used to make the lab easier to run; it is not the highest-quality MusicGen variant.

The model is pretrained; this project does not include model weights. Generated audio quality depends on the prompt and runtime. Respect the model and dataset licenses/terms for your intended use.

## GitHub
Recommended to commit the notebook, source script, README and requirements. Generated `.wav` files can be large; include a short sample only if useful. Never commit access tokens or credentials.
