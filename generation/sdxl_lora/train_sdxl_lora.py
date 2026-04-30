#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion XL with LoRA on CrisisNLP disaster images.

Trains unconditionally (empty prompts) — the LoRA adapter learns the
CrisisNLP image distribution without any text conditioning.
SDXL's UNet still requires the time-conditioning embeddings (image size/crop).
Requires ~20 GB VRAM with fp16 + gradient checkpointing.

Run data_prep/preprocess.py first (needs ../data/real_256/).

Usage:
  python train_sdxl_lora.py
  python train_sdxl_lora.py --epochs 5 --batch_size 2 --resolution 1024
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

DISASTER_CLASSES = ["earthquake", "fire", "flood", "hurricane", "landslide"]


class CrisisDataset(Dataset):
    def __init__(self, data_root: Path, size: int):
        self.transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.paths = []
        for cls in DISASTER_CLASSES:
            self.paths.extend((data_root / cls).glob("*.png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../checkpoints/sdxl_lora")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--resolution", type=int, default=1024,
                        help="SDXL native resolution (default 1024; use 512 if VRAM limited)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    # Precompute empty-prompt embeddings for both SDXL text encoders
    with torch.no_grad():
        empty_ids_1 = tokenizer(
            "", padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        empty_ids_2 = tokenizer_2(
            "", padding="max_length", max_length=tokenizer_2.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)

        enc1 = text_encoder(empty_ids_1, output_hidden_states=True)
        enc2 = text_encoder_2(empty_ids_2, output_hidden_states=True)
        empty_hidden = torch.cat(
            [enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1
        )  # (1, seq_len, 2048)
        empty_pooled = enc2[0]  # (1, 1280)

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
                        num_workers=4, drop_last=True, pin_memory=True)
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # Time IDs: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        dtype=torch.float32, device=device,
    )

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

        for pixel_values in pbar:
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

            added_cond = {
                "text_embeds": empty_pooled.to(dtype=torch.float16).expand(bsz, -1),
                "time_ids": time_ids.to(dtype=torch.float16).expand(bsz, -1),
            }

            with torch.cuda.amp.autocast():
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=empty_hidden.to(dtype=torch.float16).expand(bsz, -1, -1),
                    added_cond_kwargs=added_cond,
                ).sample
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
