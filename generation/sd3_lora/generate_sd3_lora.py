#!/usr/bin/env python3
"""
Generate fake disaster images using SD3-medium + LoRA adapter.

Prompts are sampled from the VLM-generated captions (data/captions.json).
Falls back to hardcoded class-level prompts if captions are unavailable.

Images are generated at 512×512 and downsampled to 256×256 to match the
CrisisNLP dataset resolution.

Prerequisites:
  Accept the model license at huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
  Run: huggingface-cli login
  Run: train_sd3_lora.py first to produce a LoRA adapter.

Input:  ../checkpoints/sd3_lora/final  (LoRA adapter)
        ../data/captions.json          (VLM captions, optional)
Output: ../data/fake/sd3_lora/<class>/
"""
import argparse
import json
import random
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]

CAPTION_PREFIX = "A social media photograph of a disaster scene. "

FALLBACK_PROMPTS = {
    "earthquake": [
        "a photo of severe earthquake damage, collapsed concrete buildings, "
        "rubble-filled streets, dust clouds, realistic documentary photography",
        "aerial photograph of earthquake devastation, destroyed city blocks, "
        "crumbled infrastructure, realistic disaster photo",
        "ground-level photograph of earthquake aftermath, collapsed walls, "
        "emergency rescue teams, realistic disaster documentation",
    ],
    "fire": [
        "a photograph of a massive wildfire consuming a hillside, intense orange flames, "
        "thick smoke plumes, burning trees, realistic disaster photography",
        "a photo of a structural fire in an urban area, building engulfed in flames, "
        "fire trucks responding, realistic documentary photo",
        "aerial photograph of a wildfire burning through forest, smoke column rising, "
        "realistic disaster documentation",
    ],
    "flood": [
        "a photograph of extreme urban flooding, cars fully submerged, "
        "rescue boats on flooded streets, brown floodwater, realistic disaster photo",
        "a photo of a flooded residential neighborhood, houses partially underwater, "
        "floating debris, emergency response, realistic documentary photography",
        "aerial photograph of severe flooding, brown water covering an entire town, "
        "only rooftops visible, realistic disaster documentation",
    ],
    "hurricane": [
        "a photograph of major hurricane damage, flattened buildings, "
        "uprooted trees, debris-covered streets, realistic disaster photo",
        "a photo of hurricane aftermath, collapsed roofs, downed power lines, "
        "scattered wreckage, realistic documentary photography",
        "aerial photograph of hurricane destruction, widespread coastal devastation, "
        "realistic disaster documentation",
    ],
    "landslide": [
        "a photograph of a massive landslide, mud and rocks burying a mountain road, "
        "destroyed vehicles, realistic disaster photo",
        "a photo of a mudslide covering a hillside village, mud flow engulfing houses, "
        "rescue teams working, realistic documentary photography",
        "aerial photograph of a large landslide, brown mud scar on green hillside, "
        "debris flow covering roads and buildings, realistic disaster documentation",
    ],
}


def load_class_captions(captions_path: Path) -> dict[str, list[str]]:
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--captions_file", default="../data/captions.json")
    parser.add_argument("--lora_dir", default="../checkpoints/sd3_lora/final")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--n_images", type=int, default=500)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=256,
                        help="Resize generated images to this square size (default: 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    args = parser.parse_args()

    class_captions = load_class_captions(Path(args.captions_file))
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )

    print(f"Loading LoRA adapter from {args.lora_dir} ...")
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_dir)
    pipe.transformer = pipe.transformer.merge_and_unload()

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    out_root = Path(args.data_dir) / "fake" / "sd3_lora"

    for cls in args.classes:
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        pool = class_captions.get(cls) or FALLBACK_PROMPTS[cls]
        if class_captions.get(cls):
            print(f"\n[{cls}] generating {args.n_images} images (sampling from {len(pool)} VLM captions) ...")
        else:
            print(f"\n[{cls}] generating {args.n_images} images (no captions found, using fallback prompts) ...")

        using_vlm = bool(class_captions.get(cls))
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
                image = image.resize(
                    (args.output_size, args.output_size), Image.LANCZOS
                )
            image.save(out_dir / f"{i:05d}.png")

        print(f"  Saved {args.n_images} images -> {out_dir}")

    print("\nSD3-medium LoRA generation complete.")


if __name__ == "__main__":
    main()
