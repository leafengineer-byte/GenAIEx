"""Generate a short music clip with a pretrained MusicGen model."""
import argparse
from pathlib import Path
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Peaceful instrumental piano with soft flute, warm ambient textures, slow tempo")
    parser.add_argument("--duration", type=int, default=8, help="Duration in seconds; keep short for Colab memory")
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument("--out", default="outputs/musicgen_sample")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (first run downloads pretrained weights)...")
    model = MusicGen.get_pretrained(args.model, device=device)
    model.set_generation_params(duration=max(1, min(args.duration, 30)))
    wav = model.generate([args.prompt], progress=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_write(str(out), wav[0].cpu(), model.sample_rate, strategy="loudness")
    print(f"Saved: {out}.wav")

if __name__ == "__main__":
    main()
