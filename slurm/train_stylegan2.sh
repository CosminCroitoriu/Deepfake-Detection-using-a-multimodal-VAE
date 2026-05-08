#!/bin/bash
#SBATCH --job-name=train_stylegan2
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_stylegan2_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/train_stylegan2_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Usage:
#   sbatch train_stylegan2.sh                          # trains all 5 classes sequentially
#   sbatch --export=ALL,CLASSES="flood fire" train_stylegan2.sh   # trains specific classes

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Starting StyleGAN2-ADA training"
echo "  Project dir : $PROJECT_DIR"
echo "  Classes     : ${CLASSES:-all}"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

if [ -n "$CLASSES" ]; then
    python generation/stylegan2/train_stylegan2.py \
        --kimg 5000 \
        --classes $CLASSES
else
    python generation/stylegan2/train_stylegan2.py \
        --kimg 5000
fi

echo ""
echo "Done at $(date)"
