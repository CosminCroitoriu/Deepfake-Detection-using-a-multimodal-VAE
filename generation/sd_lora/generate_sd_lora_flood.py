#!/usr/bin/env python3
"""
Generate flood fakes using the flood-only SD v1.5 + LoRA adapter.

Flood-only counterpart to generate_sd_lora.py. Prompts are sampled from the
flood VLM captions (data/captions.json), falling back to hardcoded flood prompts
when none are available.

Input:  ../../checkpoints/sd_lora_flood/final/   (peft adapter weights)
        ../../data/captions.json                 (VLM captions, optional)
Output: ../../data/fake/sd_lora_flood/flood/     (PNG images)
"""
import argparse
import json
import random
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
from diffusers import StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from peft import PeftModel
from tqdm import tqdm

MODEL_ID = "runwayml/stable-diffusion-v1-5"

FLOOD_CLASS = "flood"
CAPTION_PREFIX = "A social media photograph of a disaster scene. "

FALLBACK_FLOOD_PROMPTS = [
    "a photograph of severe urban flooding with cars and buildings submerged in water",
    "a disaster photograph of a flooded street with rescue boats and displaced residents",
    "aerial view of floodwaters inundating a town, brown water covering roads and fields",
]


def load_flood_captions(captions_path: Path) -> list[str]:
    """Return the list of flood captions from captions.json Task-1 entries."""
    if not captions_path.exists():
        return []
    all_captions = json.loads(captions_path.read_text())
    caps = []
    for rel_key, caption in all_captions.items():
        if rel_key.startswith(f"real_512/{FLOOD_CLASS}/") or rel_key.startswith(f"real_256/{FLOOD_CLASS}/"):
            caps.append(caption)
    return caps


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
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--captions_file", default=str(SCRIPT_DIR / "../../data/captions.json"))
    parser.add_argument("--adapter_path", default=str(SCRIPT_DIR / "../../checkpoints/sd_lora_flood/final"))
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--n_images", type=int, default=1000,
                        help="Flood images to generate (default: 1000)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"ERROR: adapter not found at {adapter_path}")
        print("Run train_sd_lora_flood.py first.")
        return 1

    flood_caps = load_flood_captions(Path(args.captions_file))
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_pipeline(args.model_id, adapter_path, device)

    out_dir = Path(args.data_dir) / "fake" / "sd_lora_flood" / FLOOD_CLASS
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = flood_caps or FALLBACK_FLOOD_PROMPTS
    using_vlm = bool(flood_caps)
    if using_vlm:
        print(f"\n[flood] generating {args.n_images} images (sampling from {len(pool)} VLM captions) ...")
    else:
        print(f"\n[flood] generating {args.n_images} images (no captions found, using fallback prompts) ...")

    for i in tqdm(range(args.n_images)):
        prompt = rng.choice(pool)
        if using_vlm:
            prompt = CAPTION_PREFIX + prompt
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=512,
                width=512,
            ).images[0]
        image.save(out_dir / f"{i:05d}.png")

    print(f"  Saved {args.n_images} images -> {out_dir}")
    print("\nFlood-only SD v1.5 LoRA generation complete.")


if __name__ == "__main__":
    main()
