#!/usr/bin/env python3
"""
Generate fake disaster images using a fine-tuned SD v1.5 + LoRA adapter.

Prompts are sampled from the VLM-generated captions (data/captions.json).
Falls back to hardcoded class-level prompts if captions are unavailable.

Input:  ../checkpoints/sd_lora/final/  (peft adapter weights)
        ../data/captions.json          (VLM captions, optional)
Output: ../data/fake/sd_lora/<class>/   (PNG images)
"""
import argparse
import json
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from peft import PeftModel
from tqdm import tqdm

MODEL_ID = "runwayml/stable-diffusion-v1-5"

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]

FALLBACK_PROMPTS = {
    "earthquake": [
        "a photograph of earthquake damage, collapsed concrete buildings and rubble in the streets",
        "aerial view of earthquake devastation, destroyed structures and debris fields",
        "a disaster photograph of earthquake aftermath, crumbled walls and displaced people",
    ],
    "fire": [
        "a photograph of a wildfire raging through a residential neighborhood with flames and smoke",
        "a disaster photograph of a building fire with thick black smoke and emergency response",
        "aerial view of a wildfire consuming a forest, orange flames and smoke plumes",
    ],
    "flood": [
        "a photograph of severe urban flooding with cars and buildings submerged in water",
        "a disaster photograph of a flooded street with rescue boats and displaced residents",
        "aerial view of floodwaters inundating a town, brown water covering roads and fields",
    ],
    "hurricane": [
        "a photograph of hurricane damage with uprooted trees and destroyed buildings",
        "a disaster photograph of hurricane aftermath, collapsed roofs and scattered debris",
        "aerial view of hurricane destruction, widespread structural damage and flooding",
    ],
    "landslide": [
        "a photograph of a landslide with mud and rocks covering a mountain road",
        "a disaster photograph of a mudslide destroying homes in a hillside community",
        "aerial view of a massive landslide, mud flow burying structures and roads",
    ],
}


def load_class_captions(captions_path: Path, data_dir: Path) -> dict[str, list[str]]:
    """Return {class: [caption, ...]} built from captions.json Task-1 entries."""
    if not captions_path.exists():
        return {}
    all_captions = json.loads(captions_path.read_text())
    class_caps: dict[str, list[str]] = {cls: [] for cls in DISASTER_CLASSES}
    for rel_key, caption in all_captions.items():
        for cls in DISASTER_CLASSES:
            if rel_key.startswith(f"real_256/{cls}/"):
                class_caps[cls].append(caption)
                break
    return class_caps


def load_pipeline(model_id: str, adapter_path: Path, device: torch.device):
    print(f"Loading base model {model_id} ...")
    base_unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    )
    unet = PeftModel.from_pretrained(base_unet, str(adapter_path))
    unet = unet.merge_and_unload()  # bake LoRA into weights for faster inference

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, unet=unet, torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--captions_file", default="../data/captions.json")
    parser.add_argument("--adapter_path", default="../checkpoints/sd_lora/final")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--n_images", type=int, default=500,
                        help="Images to generate per class (default: 500)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}")
        print("Run train_sd_lora.py first.")
        return 1

    class_captions = load_class_captions(Path(args.captions_file), Path(args.data_dir))
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.model_id, adapter_path, device)

    out_root = Path(args.data_dir) / "fake" / "sd_lora"

    for cls in args.classes:
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        pool = class_captions.get(cls) or FALLBACK_PROMPTS[cls]
        if class_captions.get(cls):
            print(f"\n[{cls}] generating {args.n_images} images (sampling from {len(pool)} VLM captions) ...")
        else:
            print(f"\n[{cls}] generating {args.n_images} images (no captions found, using fallback prompts) ...")

        for i in tqdm(range(args.n_images)):
            prompt = rng.choice(pool)
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    height=256,
                    width=256,
                ).images[0]
            image.save(out_dir / f"{i:05d}.png")

        print(f"  Saved {args.n_images} images -> {out_dir}")

    print("\nSD v1.5 LoRA generation complete.")


if __name__ == "__main__":
    main()
