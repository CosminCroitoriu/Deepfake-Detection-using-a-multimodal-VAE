#!/usr/bin/env python3
"""
Flood-only fine-tune of Stable Diffusion 3 Medium with LoRA (single-disaster ablation).

Identical to train_sd3_lora.py except the dataset is restricted to the flood
class in real_512/flood and the extra non-disaster images (real_extra_512) are
not used. Each image is conditioned on its VLM-generated caption, falling back
to a generic flood prompt when none exists.

Motivation: the all-classes run pointed a rank-16 adapter at ~40k images
spanning five disaster types plus generic damage/non-disaster photos, which
exceeds the adapter's capacity and yields generic samples. Here the same adapter
covers one coherent mode (flood, ~3.2k images), which should produce more
faithful fakes. Uses SD3's flow-matching objective. Writes to a separate
checkpoint directory so the all-classes run is untouched.

Prerequisites:
  Accept the model license at huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
  Run: huggingface-cli login
  Run data_prep/prepare_dataset.py
  Run data_prep/preprocess.py
  Run data_prep/generate_captions.py

Usage:
  python train_sd3_lora_flood.py
  python train_sd3_lora_flood.py --epochs 8 --batch_size 16
"""
import argparse
import json
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

FLOOD_CLASS = "flood"
FLOOD_PROMPT = "a photograph of a flood disaster"

SCRIPT_DIR = Path(__file__).resolve().parent


def find_latest_checkpoint(out_dir: Path, epochs: int) -> tuple[Path | None, int]:
    for e in range(epochs, 0, -1):
        candidate = out_dir / f"epoch_{e:03d}"
        if (candidate / "training_state.pt").exists():
            return candidate, e
    return None, 0


class FloodDataset(Dataset):
    """Flood-only: real_512/flood, with per-image captions. No extra images."""

    def __init__(self, data_dir: Path, captions: dict[str, str], size: int = 512):
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.samples = []  # (path, caption)
        missing = 0

        flood_dir = data_dir / "real_512" / FLOOD_CLASS
        for p in sorted(flood_dir.glob("*.png")):
            rel = f"real_512/{FLOOD_CLASS}/{p.name}"
            rel_legacy = f"real_256/{FLOOD_CLASS}/{p.name}"
            caption = captions.get(rel) or captions.get(rel_legacy, FLOOD_PROMPT)
            self.samples.append((p, caption))
            if rel not in captions and rel_legacy not in captions:
                missing += 1

        if not self.samples:
            raise RuntimeError(f"No flood images found under {flood_dir}")
        if missing:
            print(f"  WARNING: {missing} flood images have no caption, using flood prompt")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), caption


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "../../data"))
    parser.add_argument("--captions_file", default=str(SCRIPT_DIR / "../../data/captions.json"))
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "../../checkpoints/sd3_lora_flood"))
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=1)
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
        print(f"WARNING: {captions_path} not found — falling back to flood prompt for all images")
        captions = {}

    print(f"Loading {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )

    # Keep text encoders on GPU — they encode per-batch captions during training
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.text_encoder_3.to(device)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder_3.requires_grad_(False)

    vae = pipe.vae.to(device)
    vae.requires_grad_(False)

    transformer = pipe.transformer.to(device)
    transformer.requires_grad_(False)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    if resume_ckpt:
        from peft import PeftModel
        transformer = PeftModel.from_pretrained(transformer, str(resume_ckpt), is_trainable=True).to(device)
    else:
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

    dataset = FloodDataset(Path(args.data_dir), captions, args.resolution)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b]),
    )
    print(f"Dataset: {len(dataset)} flood images, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr, weight_decay=1e-2)
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

    T = noise_scheduler.config.num_train_timesteps

    for epoch in range(start_epoch, args.epochs):
        transformer.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for pixel_values, prompts in pbar:
            pixel_values = pixel_values.to(device, dtype=torch.float16)
            bsz = pixel_values.shape[0]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=prompts,
                    prompt_3=prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, T, (bsz,), device=device).long()

            sigmas = (timesteps.float() / T).view(-1, 1, 1, 1).to(latents.dtype)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            target = noise - latents

            with torch.cuda.amp.autocast():
                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
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
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                ckpt / "training_state.pt",
            )
            print(f"  Saved adapter + training state -> {ckpt}")

    transformer.save_pretrained(out_dir / "final")
    print(f"\nDone. Final adapter: {out_dir / 'final'}")


if __name__ == "__main__":
    main()
