#!/bin/bash
#SBATCH --job-name=train_gated_vae
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_gated_vae_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_gated_vae_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Starting Gated Multimodal VAE training (Step 4)"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

RESUME_FLAG=""
CKPT="$PROJECT_DIR/checkpoints/gated_vae/latest.pth"
if [ -f "$CKPT" ]; then
    echo "  Auto-resuming from $CKPT"
    RESUME_FLAG="--resume $CKPT"
fi

python -m detection.gated_vae.train \
    --epochs 100 \
    --batch 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --latent_dim 256 \
    --num_workers 8 \
    $RESUME_FLAG

echo ""
echo "Done at $(date)"
