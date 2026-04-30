#!/usr/bin/env python3
"""
Generate fake disaster images using a fine-tuned SDXL + LoRA adapter.

Input:  ../checkpoints/sdxl_lora/final/  (peft adapter weights)
Output: ../data/fake/sdxl_lora/<class>/   (PNG images)
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.models import UNet2DConditionModel
from peft import PeftModel
from tqdm import tqdm

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]

PROMPTS = {
    "earthquake": [
        "a photograph of earthquake damage, collapsed concrete buildings and rubble, disaster scene",
        "aerial photograph of earthquake devastation, destroyed city blocks and dust clouds",
        "a high-resolution disaster photo of earthquake aftermath, crumbled infrastructure",
    ],
    "fire": [
        "a photograph of a wildfire engulfing a hillside neighborhood, orange flames and smoke",
        "a disaster photo of a large building fire, firefighters and thick black smoke",
        "aerial photograph of a wildfire burning through dry forest, smoke columns rising",
    ],
    "flood": [
        "a photograph of extreme urban flooding, cars and streets fully submerged in water",
        "a disaster photo of floodwaters sweeping through a town, rescue operations ongoing",
        "aerial photograph of severe flooding, brown water covering entire neighborhoods",
    ],
    "hurricane": [
        "a photograph of major hurricane damage, shattered buildings and uprooted trees",
        "a disaster photo of hurricane aftermath, debris-strewn streets and collapsed roofs",
        "aerial photograph of hurricane destruction, widespread devastation across a coastal area",
    ],
    "landslide": [
        "a photograph of a catastrophic landslide, mud and rocks burying a mountain village",
        "a disaster photo of a mudslide covering a road, rescue teams working in debris",
        "aerial photograph of a large landslide scar, brown mud flow over green hillside",
    ],
}


def load_pipeline(model_id: str, adapter_path: Path, device: torch.device):
    print(f"Loading base SDXL model ...")
    base_unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    )
    unet = PeftModel.from_pretrained(base_unet, str(adapter_path))
    unet = unet.merge_and_unload()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, unet=unet, torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--adapter_path", default="../checkpoints/sdxl_lora/final")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--n_images", type=int, default=500)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}")
        print("Run train_sdxl_lora.py first.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.model_id, adapter_path, device)

    out_root = Path(args.data_dir) / "fake" / "sdxl_lora"

    for cls in args.classes:
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        prompts = PROMPTS[cls]
        print(f"\n[{cls}] generating {args.n_images} images ...")

        for i in tqdm(range(args.n_images)):
            prompt = prompts[i % len(prompts)]
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                ).images[0]
            # Resize to 256x256 to match the rest of the dataset
            image = image.resize((256, 256), Image.LANCZOS)
            image.save(out_dir / f"{i:05d}.png")

        print(f"  Saved {args.n_images} images -> {out_dir}")

    print("\nSDXL LoRA generation complete.")


if __name__ == "__main__":
    main()
