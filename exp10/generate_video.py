"""Generate a short video from a prompt using Hugging Face Diffusers."""
import argparse
from pathlib import Path
import torch
import imageio
from diffusers import DiffusionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot")
    parser.add_argument("--model", default="damo-vilab/text-to-video-ms-1.7b")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--out", default="outputs/generated_video.mp4")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: no CUDA GPU detected; this model may be extremely slow or run out of memory.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {args.model}; the first run downloads model weights.")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    result = pipe(args.prompt, num_frames=max(8, args.frames), num_inference_steps=max(1, args.steps))
    frames = result.frames[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, [frame for frame in frames], fps=8, macro_block_size=1)
    print(f"Saved video: {out}")


if __name__ == "__main__":
    main()
