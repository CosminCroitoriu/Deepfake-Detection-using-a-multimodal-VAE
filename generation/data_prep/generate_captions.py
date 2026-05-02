#!/usr/bin/env python3
"""
Generate VLM captions and disaster-type labels for all real CrisisNLP images.

Runs LLaVA-1.5-7B (or InstructBLIP-Vicuna-7B) over every image in
real_256/ and real_extra_256/ using a combined prompt that produces both
a scene description and a disaster type classification in one forward pass.

Outputs:
  data/captions.json   {"rel/path/img.png": "scene description ..."}
  data/labels.json     {"rel/path/img.png": "flood"}   (disaster type)

For Task-1 images (real_256/<class>/):
  - Caption: from VLM
  - Label:   ground truth from directory name (VLM prediction ignored)

For extra images (real_extra_256/):
  - Caption: from VLM
  - Label:   VLM prediction; "unclear" if the VLM can't confidently classify

The labels.json enables GAN training (StyleGAN2-ADA, ProjectedGAN) to use
extra images by providing the class information they require.

Usage:
  python generate_captions.py
  python generate_captions.py --model instructblip
  python generate_captions.py --resume    # skip already-captioned images
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]
VALID_TYPES = set(DISASTER_CLASSES) | {"unclear"}

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
INSTRUCTBLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"

# Combined prompt: one forward pass produces both caption and disaster type.
# The structured format makes parsing reliable.
LLAVA_PROMPT = (
    "USER: <image>\n"
    "Look at this disaster photograph and do two things:\n"
    "1. Describe the scene in detail: the type of disaster, visible damage, "
    "affected structures, environmental conditions, and overall atmosphere.\n"
    "2. Classify the disaster type as exactly one word from: "
    "earthquake, fire, flood, hurricane, landslide, unclear.\n\n"
    "Use this exact format:\n"
    "Description: <your description>\n"
    "Type: <one word>\n"
    "ASSISTANT:"
)

INSTRUCTBLIP_PROMPT = (
    "Look at this disaster photograph and do two things:\n"
    "1. Describe the scene in detail: the type of disaster, visible damage, "
    "affected structures, environmental conditions, and overall atmosphere.\n"
    "2. Classify the disaster type as exactly one word from: "
    "earthquake, fire, flood, hurricane, landslide, unclear.\n\n"
    "Use this exact format:\n"
    "Description: <your description>\n"
    "Type: <one word>"
)


def parse_response(text: str) -> tuple[str, str]:
    """
    Extract (caption, disaster_type) from the structured VLM response.
    Returns ("unclear", "unclear") if parsing fails.
    """
    caption = "unclear"
    disaster_type = "unclear"

    desc_match = re.search(r"Description:\s*(.+?)(?=\nType:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if desc_match:
        caption = desc_match.group(1).strip()

    type_match = re.search(r"Type:\s*(\w+)", text, re.IGNORECASE)
    if type_match:
        predicted = type_match.group(1).strip().lower()
        if predicted in VALID_TYPES:
            disaster_type = predicted

    return caption, disaster_type


def load_llava():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    print(f"Loading {LLAVA_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return processor, model


def load_instructblip():
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    print(f"Loading {INSTRUCTBLIP_MODEL_ID} ...")
    processor = InstructBlipProcessor.from_pretrained(INSTRUCTBLIP_MODEL_ID)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        INSTRUCTBLIP_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return processor, model


def run_llava(processor, model, img_path: Path, max_new_tokens: int) -> str:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=LLAVA_PROMPT, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def run_instructblip(processor, model, img_path: Path, max_new_tokens: int) -> str:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, text=INSTRUCTBLIP_PROMPT, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def collect_images(data_dir: Path) -> list[tuple[Path, str, str | None]]:
    """
    Return (abs_path, rel_key, ground_truth_label) for every image.
    ground_truth_label is the class name for Task-1 images, None for extra images.
    """
    images = []

    for cls in DISASTER_CLASSES:
        cls_dir = data_dir / "real_256" / cls
        if cls_dir.exists():
            for p in sorted(cls_dir.glob("*.png")):
                images.append((p, f"real_256/{cls}/{p.name}", cls))

    extra_dir = data_dir / "real_extra_256"
    if extra_dir.exists():
        for p in sorted(extra_dir.glob("*.png")):
            images.append((p, f"real_extra_256/{p.name}", None))

    return images


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--captions_out", default="../data/captions.json")
    parser.add_argument("--labels_out", default="../data/labels.json")
    parser.add_argument(
        "--model", choices=["llava", "instructblip"], default="llava",
        help="VLM to use (default: llava)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Max tokens per response (default: 150)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip images already present in the output JSON",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    captions_path = Path(args.captions_out)
    labels_path = Path(args.labels_out)

    captions: dict[str, str] = {}
    labels: dict[str, str] = {}

    if args.resume:
        if captions_path.exists():
            captions = json.loads(captions_path.read_text())
        if labels_path.exists():
            labels = json.loads(labels_path.read_text())
        print(f"Resuming: {len(captions):,} captions already saved")

    images = collect_images(data_dir)
    print(f"Found {len(images):,} images total "
          f"({sum(1 for _, _, gt in images if gt)} Task-1, "
          f"{sum(1 for _, _, gt in images if not gt)} extra)")

    if args.resume:
        images = [(p, k, gt) for p, k, gt in images if k not in captions]
        print(f"  {len(images):,} remaining")

    if not images:
        print("Nothing to do.")
        return

    if args.model == "llava":
        processor, model = load_llava()
        run_fn = lambda p: run_llava(processor, model, p, args.max_new_tokens)
    else:
        processor, model = load_instructblip()
        run_fn = lambda p: run_instructblip(processor, model, p, args.max_new_tokens)

    captions_path.parent.mkdir(parents=True, exist_ok=True)
    unclear_count = 0

    for img_path, rel_key, ground_truth in tqdm(images, desc="Captioning"):
        try:
            raw = run_fn(img_path)
            caption, vlm_type = parse_response(raw)
        except Exception as e:
            print(f"  SKIP {img_path.name}: {e}")
            continue

        captions[rel_key] = caption

        # Ground truth always wins for Task-1; VLM prediction used for extras
        if ground_truth is not None:
            labels[rel_key] = ground_truth
        else:
            labels[rel_key] = vlm_type
            if vlm_type == "unclear":
                unclear_count += 1

        if len(captions) % 100 == 0:
            captions_path.write_text(json.dumps(captions, indent=2))
            labels_path.write_text(json.dumps(labels, indent=2))

    captions_path.write_text(json.dumps(captions, indent=2))
    labels_path.write_text(json.dumps(labels, indent=2))

    extra_total = sum(1 for k in labels if k.startswith("real_extra_256/"))
    print(f"\nDone.")
    print(f"  {len(captions):,} captions -> {captions_path}")
    print(f"  {len(labels):,} labels   -> {labels_path}")

    if extra_total:
        from collections import Counter
        extra_dist = Counter(
            v for k, v in labels.items() if k.startswith("real_extra_256/")
        )
        print(f"\nExtra image label distribution ({extra_total} total):")
        for cls in DISASTER_CLASSES + ["unclear"]:
            n = extra_dist.get(cls, 0)
            if n:
                print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
