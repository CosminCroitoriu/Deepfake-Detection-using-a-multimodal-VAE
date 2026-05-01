#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion v1.5 with LoRA on CrisisNLP disaster images.

Each image is conditioned on its class-level prompt (e.g. "a photograph of
a flood disaster") so training and inference use the same text conditioning.

Targets A100/H100/H200 cluster nodes (40-141 GB VRAM).
Default batch size 32 fits comfortably on a single A100 40GB.

Run data_prep/preprocess.py first (needs ../data/real_256/).

Usage:
  python train_sd_lora.py
  python train_sd_lora.py --epochs 10 --batch_size 64
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
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


class CrisisDataset(Dataset):
    def __init__(self, data_root: Path, size: int = 512):
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.samples = []  # (path, prompt)
        for cls in DISASTER_CLASSES:
            prompt = CLASS_PROMPTS[cls]
            for p in (data_root / cls).glob("*.png"):
                self.samples.append((p, prompt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, prompt = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), prompt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../checkpoints/sd_lora")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (only needed on low-VRAM GPUs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, safety_checker=None
    )
    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Precompute one embedding per class — reused each batch
    with torch.no_grad():
        class_embeds = {}
        for cls, prompt in CLASS_PROMPTS.items():
            ids = tokenizer(
                prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            class_embeds[cls] = text_encoder(ids)[0]  # (1, seq_len, 768)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    dataset = CrisisDataset(Path(args.data_dir) / "real_256", args.resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True,
                        collate_fn=lambda b: (
                            torch.stack([x[0] for x in b]),
                            [x[1] for x in b],
                        ))
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(loader)
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        unet.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for pixel_values, prompts in pbar:
            pixel_values = pixel_values.to(device, dtype=torch.float16)
            bsz = pixel_values.shape[0]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Look up precomputed embedding for each prompt in the batch
            encoder_hidden = torch.cat(
                [class_embeds[p] for p in prompts], dim=0
            )

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

        print(f"Epoch {epoch + 1}: avg_loss={total_loss / len(loader):.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt = out_dir / f"epoch_{epoch + 1:03d}"
            unet.save_pretrained(ckpt)
            print(f"  Saved adapter -> {ckpt}")

    unet.save_pretrained(out_dir / "final")
    print(f"\nDone. Final adapter: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
