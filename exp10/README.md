# Experiment 10: Text-to-Video Generation with a Pretrained Model

This educational starter project generates a short video from a text prompt using a pretrained open-source text-to-video model. It does not train a model or require collecting a dataset.

## Model choice and compute warning
The notebook uses `damo-vilab/text-to-video-ms-1.7b` through Hugging Face Diffusers as an example. Model availability, access terms, library compatibility, and GPU memory requirements can change. A GPU with substantial VRAM is strongly recommended; this may not run on a free Colab GPU every time. If loading fails due to memory, use a smaller supported model or a hosted inference option permitted by your course.

## Files
- `text_to_video_colab.ipynb` — Google Colab notebook
- `generate_video.py` — command-line inference script
- `requirements.txt` — dependencies
- `README.md` — setup instructions
- `.gitignore` — avoids committing generated videos and secrets
- `outputs/` — generated MP4s

## Run in Google Colab
1. Open `text_to_video_colab.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU**.
3. Run the notebook cells in order.
4. Use a concise prompt describing subject, action, setting, camera and style.
5. Preview the MP4 in the notebook and download it.

The first run downloads model weights and may take time. Generation is memory-intensive. This package does not include model weights or a pre-generated video.

## Example prompt
`A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot`

## Run locally
Install a compatible PyTorch build for your machine, then:
```bash
pip install -r requirements.txt
python generate_video.py --prompt "A small boat sailing across a calm lake at sunrise, cinematic wide shot" --frames 16 --steps 25
```

## GitHub advice
Commit source code, notebook, README and requirements. Add a short generated MP4 or a few screenshots only after you generate them. Do not commit large model weights, private tokens, `.env` files, or large generated videos unnecessarily.
