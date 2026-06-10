#!/usr/bin/env python3
"""
Generate flood fakes using the flood-only SD3-medium + LoRA adapter.

Flood-only counterpart to generate_sd3_lora.py. Prompts are sampled from the
flood VLM captions (data/captions.json), falling back to hardcoded flood prompts
when none are available. Images are generated at 512x512.

Prerequisites:
  Accept the model license at huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
  Run: huggingface-cli login
  Run: train_sd3_lora_flood.py first to produce a LoRA adapter.

Input:  ../../checkpoints/sd3_lora_flood/final   (LoRA adapter)
        ../../data/captions.json                 (VLM captions, optional)
Output: ../../data/fake/sd3_lora_flood/flood/
"""
import argparse
import json
import random
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

FLOOD_CLASS = "flood"
CAPTION_PREFIX = "A social media photograph of a disaster scene. "

FALLBACK_FLOOD_PROMPTS = [
    "a photograph of extreme urban flooding, cars fully submerged, "
    "rescue boats on flooded streets, brown floodwater, realistic disaster photo",
    "a photo of a flooded residential neighborhood, houses partially underwater, "
    "floating debris, emergency response, realistic documentary photography",
    "aerial photograph of severe flooding, brown water covering an entire town, "
    "only rooftops visible, realistic disaster documentation",
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--captions_file", default=str(SCRIPT_DIR / "../../data/captions.json"))
    parser.add_argument("--lora_dir", default=str(SCRIPT_DIR / "../../checkpoints/sd3_lora_flood/final"))
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--n_images", type=int, default=1000,
                        help="Flood images to generate (default: 1000)")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=512,
                        help="Resize generated images to this square size (default: 512)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    if not lora_dir.exists():
        print(f"ERROR: adapter not found at {lora_dir}")
        print("Run train_sd3_lora_flood.py first.")
        return 1

    flood_caps = load_flood_captions(Path(args.captions_file))
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )

    print(f"Loading LoRA adapter from {lora_dir} ...")
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_dir))
    pipe.transformer = pipe.transformer.merge_and_unload()

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.data_dir) / "fake" / "sd3_lora_flood" / FLOOD_CLASS
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
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            ).images[0]
        if args.output_size != args.height or args.output_size != args.width:
            image = image.resize((args.output_size, args.output_size), Image.LANCZOS)
        image.save(out_dir / f"{i:05d}.png")

    print(f"  Saved {args.n_images} images -> {out_dir}")
    print("\nFlood-only SD3-medium LoRA generation complete.")


if __name__ == "__main__":
    main()
