# Experiment 7: StyleGAN for High-Quality Face Generation

This starter project is designed for **Google Colab with a GPU**. It demonstrates a StyleGAN-style workflow: latent mapping (`z` → `w`), style injection, style mixing, and noise-driven detail variation.

## Important note
This package contains **training code and model architecture**, not pretrained weights or the CelebA dataset. Training StyleGAN from scratch is compute-intensive and can take a long time; the included lightweight implementation is for educational experimentation and is not equivalent to the original StyleGAN2/StyleGAN3 quality. Start at 64×64 resolution for a lab demonstration. Do not claim a model was trained until training actually completes.

## Files
- `train.py` — training loop and checkpoint saving
- `models/stylegan.py` — educational StyleGAN-inspired mapping/synthesis architecture
- `dataset.py` — image-folder dataset loader
- `generate.py` — generate sample images from a checkpoint
- `style_mixing.py` — style-mixing demo
- `noise_demo.py` — stochastic noise demo
- `requirements.txt` — dependencies
- `colab/Experiment_7_StyleGAN.ipynb` — Colab notebook outline with commands
- `templates/index.html`, `app.py` — optional Flask demo
- `outputs/` — placeholder for generated images
- `checkpoints/` — placeholder for trained checkpoints

## Setup in Colab
1. Upload this ZIP to Google Drive and extract it, or upload the folder to Colab.
2. In Colab, enable **Runtime → Change runtime type → T4 GPU** (or available GPU).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Prepare CelebA images in a folder. The dataset loader expects images under `data/celeba/` (subfolders are okay).
5. Run a short smoke test first:
   ```bash
   python train.py --data data/celeba --epochs 1 --batch-size 8 --image-size 64 --limit-images 500
   ```
6. Train longer after confirming the pipeline works:
   ```bash
   python train.py --data data/celeba --epochs 20 --batch-size 16 --image-size 64
   ```
   Adjust batch size down if CUDA runs out of memory.
7. Generate:
   ```bash
   python generate.py --checkpoint checkpoints/latest.pt --count 8
   python style_mixing.py --checkpoint checkpoints/latest.pt
   python noise_demo.py --checkpoint checkpoints/latest.pt
   ```

## CelebA dataset
Download CelebA through a source you are authorized to use and follow its license/terms. If using Kaggle, configure the Kaggle API token securely in Colab. Never commit `kaggle.json`, API tokens, or dataset images to GitHub.

## Model quality and scope
This is an educational, compact StyleGAN-inspired model. It includes a mapping network and style-conditioned synthesis blocks with noise injection, but it is **not an exact reproduction of the original StyleGAN architecture** and may produce low-quality outputs unless trained sufficiently. For realistic, high-quality faces, use a properly licensed pretrained StyleGAN checkpoint and compatible inference code; large checkpoints are not bundled in this ZIP.

## GitHub recommendation
Push the source code, notebook, README, requirements file, and a small output screenshot only after you actually generate it. Do not push `data/`, `.env`, `kaggle.json`, virtual environments, or large checkpoints unless you have a clear reason.
