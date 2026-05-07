#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion v1.5 with LoRA on CrisisNLP disaster images.

Each image is conditioned on its VLM-generated caption (from generate_captions.py).
Falls back to a generic class-level prompt for any image without a caption.

Training uses images from all tasks:
  - real_512/<class>/       Task 1 (disaster types, 5 classes)
  - real_extra_512/         Tasks 2/3/4 (extra disaster images)

Targets A100/H100/H200 cluster nodes (40-141 GB VRAM).
Default batch size 32 fits comfortably on a single A100 40GB.

Prerequisites:
  Run data_prep/prepare_dataset.py
  Run data_prep/preprocess.py --include_extra
  Run data_prep/generate_captions.py

Usage:
  python train_sd_lora.py
  python train_sd_lora.py --epochs 10 --batch_size 64
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

MODEL_ID = "runwayml/stable-diffusion-v1-5"

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]

CLASS_PROMPTS = {
    "earthquake": "a photograph of earthquake damage",
    "fire":       "a photograph of a wildfire disaster",
    "flood":      "a photograph of a flood disaster",
    "hurricane":  "a photograph of hurricane damage",
    "landslide":  "a photograph of a landslide disaster",
}

FALLBACK_PROMPT = "a photograph of a natural disaster"

SCRIPT_DIR = Path(__file__).resolve().parent


class CrisisDataset(Dataset):
    def __init__(self, data_dir: Path, captions: dict[str, str], size: int = 512):
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.samples = []  # (path, caption)
        missing = 0

        # Task 1: per-class images
        for cls in DISASTER_CLASSES:
            fallback = CLASS_PROMPTS[cls]
            for p in sorted((data_dir / "real_512" / cls).glob("*.png")):
                rel = f"real_512/{cls}/{p.name}"
                rel_legacy = f"real_256/{cls}/{p.name}"
                caption = captions.get(rel) or captions.get(rel_legacy, fallback)
                self.samples.append((p, caption))
                if rel not in captions and rel_legacy not in captions:
                    missing += 1

        # Tasks 2/3/4: extra images — only include if a caption exists
        extra_dir = data_dir / "real_extra_512"
        if extra_dir.exists():
            for p in sorted(extra_dir.glob("*.png")):
                rel = f"real_extra_512/{p.name}"
                rel_legacy = f"real_extra_256/{p.name}"
                caption = captions.get(rel) or captions.get(rel_legacy)
                if caption:
                    self.samples.append((p, caption))

        if missing:
            print(f"  WARNING: {missing} Task-1 images have no caption, using class fallback")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), caption


def find_latest_checkpoint(out_dir: Path, epochs: int) -> tuple[Path | None, int]:
    """Return (checkpoint_path, completed_epochs) for the latest saved epoch."""
    for e in range(epochs, 0, -1):
        candidate = out_dir / f"epoch_{e:03d}"
        if (candidate / "training_state.pt").exists():
            return candidate, e
    return None, 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--captions_file", default=str(SCRIPT_DIR / "../../data/captions.json"))
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "../../checkpoints/sd_lora"))
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_ckpt, start_epoch = find_latest_checkpoint(out_dir, args.epochs)
    if resume_ckpt:
        print(f"Resuming from epoch {start_epoch} ({resume_ckpt})")
    else:
        print("Starting from scratch")

    captions_path = Path(args.captions_file)
    if captions_path.exists():
        captions = json.loads(captions_path.read_text())
        print(f"Loaded {len(captions):,} captions from {captions_path}")
    else:
        print(f"WARNING: {captions_path} not found — falling back to class prompts for all images")
        captions = {}

    print(f"Loading {args.model_id} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, safety_checker=None
    )
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_base = pipe.unet
    unet_base.requires_grad_(False)
    if args.gradient_checkpointing:
        unet_base.enable_gradient_checkpointing()

    if resume_ckpt:
        unet = PeftModel.from_pretrained(unet_base, str(resume_ckpt), is_trainable=True).to(device)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
        )
        unet = get_peft_model(unet_base, lora_config).to(device)

    unet.print_trainable_parameters()

    dataset = CrisisDataset(Path(args.data_dir), captions, args.resolution)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b]),
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(loader)
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    if resume_ckpt:
        state = torch.load(resume_ckpt / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        scaler.load_state_dict(state["scaler"])

    for epoch in range(start_epoch, args.epochs):
        unet.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for pixel_values, prompts in pbar:
            pixel_values = pixel_values.to(device, dtype=torch.float16)
            bsz = pixel_values.shape[0]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                ids = tokenizer(
                    prompts, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(device)
                encoder_hidden = text_encoder(ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast():
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt = out_dir / f"epoch_{epoch + 1:03d}"
            unet.save_pretrained(ckpt)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                ckpt / "training_state.pt",
            )
            print(f"  Saved adapter + training state -> {ckpt}")

    unet.save_pretrained(out_dir / "final")
    print(f"\nDone. Final adapter: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
