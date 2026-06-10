#!/bin/bash
#SBATCH --job-name=generate_sd_lora_flood
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/generate_sd_lora_flood_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/generate_sd_lora_flood_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Starting flood-only SD v1.5 LoRA image generation"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

python generation/sd_lora/generate_sd_lora_flood.py \
    --n_images 1000

echo ""
echo "Done at $(date)"
