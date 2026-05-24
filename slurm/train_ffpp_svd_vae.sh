#!/bin/bash
#SBATCH --job-name=train_ffpp_svd
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_ffpp_svd_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_ffpp_svd_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Starting FF++ SVD U-Net VAE training (paper validation)"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

RESUME_FLAG=""
CKPT="$PROJECT_DIR/checkpoints/ffpp_svd_vae/latest.pth"
if [ -f "$CKPT" ]; then
    echo "  Auto-resuming from $CKPT"
    RESUME_FLAG="--resume $CKPT"
fi

python -m detection.ffpp_svd_vae.train \
    --epochs 15 \
    --batch 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --latent_dim 256 \
    --num_workers 8 \
    --save_every_epoch \
    $RESUME_FLAG

echo ""
echo "Done at $(date)"
