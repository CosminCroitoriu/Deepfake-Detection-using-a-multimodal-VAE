#!/bin/bash
#SBATCH --job-name=train_svd_vae_flood
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_svd_vae_flood_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_svd_vae_flood_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Starting flood-only SVD U-Net VAE training"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

RESUME_FLAG=""
CKPT="$PROJECT_DIR/checkpoints/svd_unet_vae_flood/latest.pth"
if [ -f "$CKPT" ]; then
    echo "  Auto-resuming from $CKPT"
    RESUME_FLAG="--resume $CKPT"
fi

python -m detection.svd_unet_vae.train_flood \
    --epochs 100 \
    --batch 32 \
    --lr 1e-3 \
    --latent_dim 256 \
    --num_workers 8 \
    $RESUME_FLAG

echo ""
echo "Done at $(date)"
