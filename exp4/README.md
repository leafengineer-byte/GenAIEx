# Experiment 4: Text-to-Image Generation using a Pre-trained Diffusion Model

## Aim
Implement a text-to-image generation system using the pre-trained Stable Diffusion XL (SDXL) model and an interactive Gradio interface.

## What is included
- `experiment_4_sdxl_colab.ipynb` — step-by-step Google Colab notebook
- `text_to_image_gradio.py` — Gradio app for prompt-based image generation
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes caches, local environments, secrets, and generated outputs

## Model
`stabilityai/stable-diffusion-xl-base-1.0` via Hugging Face Diffusers. Model weights are downloaded when the notebook/app runs; they are **not** included in this ZIP.

## Run in Google Colab (recommended)
1. Upload/open `experiment_4_sdxl_colab.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the cells from top to bottom.
4. In the Gradio cell, open the generated public or local app link shown by Colab.
5. Enter a prompt, adjust inference steps and guidance scale, then click **Generate Image**.

## Run locally
A GPU with enough VRAM is strongly recommended for SDXL. Install Python dependencies, then run:
```bash
pip install -r requirements.txt
python text_to_image_gradio.py
```
If running on a CPU or a GPU with insufficient memory, generation may be very slow or fail. A Colab GPU is usually the easiest option.

## Example prompts
- `A tiny glass greenhouse on Mars at sunrise, cinematic lighting, highly detailed`
- `A watercolor illustration of a sleepy orange cat reading a book in a cozy library`
- `A futuristic eco-city built around waterfalls, concept art, wide composition`

## Parameters
- **Inference steps**: more steps can improve refinement but increase generation time. Try 20–35 to start.
- **Guidance scale**: controls how strongly generation follows the prompt. Try around 5–9; very high values do not always improve quality.
- **Negative prompt**: describes elements to avoid, such as `blurry, low quality, distorted anatomy`.

## Notes
- The first run downloads model files and may take time.
- Hugging Face model access and current library compatibility may change; if an error occurs, restart the runtime and rerun the notebook from the installation cell.
- Never commit access tokens, private credentials, model caches, or large generated files to GitHub.
- This is an educational starter, not a guarantee that every prompt will produce a perfect image.
