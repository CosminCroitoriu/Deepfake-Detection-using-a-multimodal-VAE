#!/usr/bin/env python3
"""
Generate fake disaster images using InstructBLIP-Vicuna-7B + Stable Diffusion v1.5.

Pipeline per image:
  1. InstructBLIP analyzes a real CrisisNLP image and produces a detailed scene description.
  2. SD v1.5 (base, no LoRA) generates a new fake image from that description.

Models are loaded and unloaded sequentially to avoid holding both in VRAM at once.

Input:  ../data/real_256/<class>/
Output: ../data/fake/instructblip/<class>/
"""
import argparse
import gc
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

INSTRUCTBLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]

PROMPT = (
    "This is a photograph from a {cls} disaster. "
    "Describe the scene in detail: the type and severity of damage, "
    "affected structures, environmental conditions, and visual atmosphere."
)


def load_instructblip(device: torch.device):
    print(f"Loading {INSTRUCTBLIP_MODEL_ID} ...")
    processor = InstructBlipProcessor.from_pretrained(INSTRUCTBLIP_MODEL_ID)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        INSTRUCTBLIP_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return processor, model


def generate_descriptions(
    processor,
    model,
    img_paths: list,
    cls: str,
    max_new_tokens: int = 100,
) -> list:
    descriptions = []
    prompt = PROMPT.format(cls=cls)
    for img_path in tqdm(img_paths, desc=f"  InstructBLIP [{cls}]"):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        description = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        descriptions.append(description)
    return descriptions


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_sd(device: torch.device):
    print(f"Loading {SD_MODEL_ID} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID, torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_images(pipe, descriptions: list, out_dir: Path, steps: int, guidance: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, desc in enumerate(tqdm(descriptions, desc=f"  SD [{out_dir.name}]")):
        with torch.inference_mode():
            image = pipe(
                desc,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=256,
                width=256,
            ).images[0]
        image.save(out_dir / f"{i:05d}.png")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--n_images", type=int, default=500,
                        help="Images to generate per class (default: 500)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    src_root = Path(args.data_dir) / "real_256"
    out_root = Path(args.data_dir) / "fake" / "instructblip"

    # --- Phase 1: generate all descriptions with InstructBLIP ---
    processor, iblip = load_instructblip(device)
    all_descriptions = {}

    for cls in args.classes:
        src_dir = src_root / cls
        if not src_dir.exists():
            print(f"WARNING: {src_dir} not found, skipping {cls}")
            continue
        imgs = sorted(src_dir.glob("*.png"))
        selected = rng.sample(imgs, min(args.n_images, len(imgs)))
        descs = generate_descriptions(processor, iblip, selected, cls, args.max_new_tokens)
        all_descriptions[cls] = descs
        print(f"  {cls}: {len(descs)} descriptions generated")

    print("\nUnloading InstructBLIP ...")
    unload(iblip)
    del processor

    # --- Phase 2: generate images with SD v1.5 ---
    pipe = load_sd(device)

    for cls, descs in all_descriptions.items():
        out_dir = out_root / cls
        generate_images(pipe, descs, out_dir, args.steps, args.guidance_scale)
        print(f"  {cls}: {len(descs)} images saved -> {out_dir}")

    print("\nInstructBLIP + SD generation complete.")


if __name__ == "__main__":
    main()
