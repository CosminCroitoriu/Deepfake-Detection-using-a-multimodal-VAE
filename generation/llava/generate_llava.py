#!/usr/bin/env python3
"""
Generate fake disaster images using LLaVA-1.5-7B + Stable Diffusion v1.5.

Pipeline per image:
  1. LLaVA analyzes a real CrisisNLP image and produces a detailed scene description.
  2. SD v1.5 (base, no LoRA) generates a new fake image from that description.

Models are loaded and unloaded sequentially to avoid holding both in VRAM at once.

Input:  ../data/real_256/<class>/
Output: ../data/fake/llava/<class>/
"""
import argparse
import gc
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from diffusers import StableDiffusionPipeline

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]

# LLaVA 1.5 conversation format: USER: <image>\n{text} ASSISTANT:
LLAVA_PROMPT = (
    "USER: <image>\n"
    "This is a photograph from a {cls} disaster event. "
    "Describe the scene in detail: the visible damage, destroyed structures, "
    "environmental conditions, and overall atmosphere. Be specific and concise.\n"
    "ASSISTANT:"
)


def load_llava(device: torch.device):
    print(f"Loading {LLAVA_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
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
    prompt = LLAVA_PROMPT.format(cls=cls)
    for img_path in tqdm(img_paths, desc=f"  LLaVA [{cls}]"):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        description = processor.decode(generated, skip_special_tokens=True).strip()
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
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Max tokens for LLaVA description (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", nargs="+", default=DISASTER_CLASSES,
                        choices=DISASTER_CLASSES)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    src_root = Path(args.data_dir) / "real_256"
    out_root = Path(args.data_dir) / "fake" / "llava"

    # --- Phase 1: generate all descriptions with LLaVA ---
    processor, llava = load_llava(device)
    all_descriptions = {}

    for cls in args.classes:
        src_dir = src_root / cls
        if not src_dir.exists():
            print(f"WARNING: {src_dir} not found, skipping {cls}")
            continue
        imgs = sorted(src_dir.glob("*.png"))
        selected = rng.sample(imgs, min(args.n_images, len(imgs)))
        descs = generate_descriptions(
            processor, llava, selected, cls, args.max_new_tokens
        )
        all_descriptions[cls] = descs
        print(f"  {cls}: {len(descs)} descriptions generated")

    print("\nUnloading LLaVA ...")
    unload(llava)
    del processor

    # --- Phase 2: generate images with SD v1.5 ---
    pipe = load_sd(device)

    for cls, descs in all_descriptions.items():
        out_dir = out_root / cls
        generate_images(pipe, descs, out_dir, args.steps, args.guidance_scale)
        print(f"  {cls}: {len(descs)} images saved -> {out_dir}")

    print("\nLLaVA + SD generation complete.")


if __name__ == "__main__":
    main()
