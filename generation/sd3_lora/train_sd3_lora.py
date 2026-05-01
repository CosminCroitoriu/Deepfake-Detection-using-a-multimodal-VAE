#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion 3 Medium with LoRA on CrisisNLP disaster images.

Uses SD3's flow-matching objective. Text encoders (CLIP-L, CLIP-G, T5-XXL) are
precomputed once and offloaded to CPU — only the MMDiT transformer is kept on GPU
during training. This keeps peak VRAM well within a single A100 40 GB.

Architecturally distinct from SD v1.5 (MMDiT transformer vs. U-Net), producing a
different artifact profile that complements the SD v1.5 LoRA model.

Targets A100/H100/H200 cluster nodes (40-141 GB VRAM).
Default batch size 16 fits comfortably on a single A100 40 GB.

Prerequisites:
  Accept the model license at huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
  Run: huggingface-cli login

Run data_prep/preprocess.py first (needs ../data/real_256/).

Usage:
  python train_sd3_lora.py
  python train_sd3_lora.py --epochs 10 --batch_size 32
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

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
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.samples = []
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
    parser.add_argument("--output_dir", default="../checkpoints/sd3_lora")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
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
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )

    # Precompute one embedding per class, then offload text encoders to CPU
    print("Precomputing class embeddings ...")
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.text_encoder_3.to(device)

    class_embeds = {}
    with torch.no_grad():
        for cls, prompt in CLASS_PROMPTS.items():
            prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            class_embeds[cls] = (prompt_embeds.cpu(), pooled_embeds.cpu())
    print(f"  prompt_embeds: {prompt_embeds.shape}, pooled: {pooled_embeds.shape}")

    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.text_encoder_3.to("cpu")
    torch.cuda.empty_cache()

    vae = pipe.vae.to(device)
    transformer = pipe.transformer.to(device)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "add_q_proj", "add_k_proj", "add_v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    dataset = CrisisDataset(Path(args.data_dir) / "real_256", args.resolution)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            [x[1] for x in b],
        ),
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr, weight_decay=1e-2)
    total_steps = args.epochs * len(loader)
    lr_scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    T = noise_scheduler.config.num_train_timesteps  # 1000

    for epoch in range(args.epochs):
        transformer.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for pixel_values, prompts in pbar:
            pixel_values = pixel_values.to(device, dtype=torch.float16)
            bsz = pixel_values.shape[0]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, T, (bsz,), device=device).long()

            # Flow matching: x_t = sigma*noise + (1-sigma)*x_0, target = noise - x_0
            sigmas = (timesteps.float() / T).view(-1, 1, 1, 1).to(latents.dtype)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            target = noise - latents

            prompt_embeds_batch = torch.cat(
                [class_embeds[p][0] for p in prompts], dim=0
            ).to(device)
            pooled_embeds_batch = torch.cat(
                [class_embeds[p][1] for p in prompts], dim=0
            ).to(device)

            with torch.cuda.amp.autocast():
                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds_batch,
                    pooled_projections=pooled_embeds_batch,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(noise_pred.float(), target.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}: avg_loss={total_loss / len(loader):.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt = out_dir / f"epoch_{epoch + 1:03d}"
            transformer.save_pretrained(ckpt)
            print(f"  Saved adapter -> {ckpt}")

    transformer.save_pretrained(out_dir / "final")
    print(f"\nDone. Final adapter: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
