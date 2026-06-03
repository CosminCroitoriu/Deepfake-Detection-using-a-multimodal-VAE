#!/bin/bash
#SBATCH --job-name=train_vit_clf
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_vit_classifier_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_vit_classifier_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"
source "$PROJECT_DIR/venv/bin/activate"

echo "Starting supervised ViT classifier training (comparison baseline)"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

RESUME_FLAG=""
CKPT="$PROJECT_DIR/checkpoints/vit_classifier/latest.pth"
if [ -f "$CKPT" ]; then
    echo "  Auto-resuming from $CKPT"
    RESUME_FLAG="--resume $CKPT"
fi

python -m detection.vit_classifier.train \
    --epochs 30 \
    --batch 64 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --img_size 224 \
    --patch_size 16 \
    --embed_dim 256 \
    --depth 6 \
    --num_heads 8 \
    --dropout 0.1 \
    --num_workers 8 \
    $RESUME_FLAG

echo ""
echo "Done at $(date)"
