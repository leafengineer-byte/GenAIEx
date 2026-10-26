import torch
import gradio as gr
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def load_pipeline():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        # CPU inference can be very slow for SDXL.
        pipe = pipe.to("cpu")
    return pipe

pipe = load_pipeline()

def generate_image(prompt, negative_prompt, steps, guidance_scale, seed):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a text prompt first.")

    generator = None
    if seed is not None and int(seed) >= 0:
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    return result.images[0]

with gr.Blocks(title="SDXL Text-to-Image Generator") as demo:
    gr.Markdown("# Text-to-Image Generation using Stable Diffusion XL")
    gr.Markdown("Enter a prompt and generate an image using a pre-trained SDXL diffusion model.")
    prompt = gr.Textbox(
        label="Prompt",
        placeholder="Describe the image you want to create...",
        lines=3,
    )
    negative_prompt = gr.Textbox(
        label="Negative prompt (optional)",
        placeholder="blurry, low quality, distorted...",
        lines=2,
    )
    with gr.Row():
        steps = gr.Slider(minimum=10, maximum=50, value=25, step=1, label="Inference steps")
        guidance = gr.Slider(minimum=1, maximum=12, value=7, step=0.5, label="Guidance scale")
    seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
    generate_btn = gr.Button("Generate Image", variant="primary")
    output = gr.Image(label="Generated image", type="pil")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance, seed],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
