#!/bin/bash
#SBATCH --job-name=eval_ffpp_svd
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/eval_ffpp_svd_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/eval_ffpp_svd_%j.err
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

echo "Starting FF++ SVD U-Net VAE evaluation"
echo "  Project dir : $PROJECT_DIR"
echo "  Time        : $(date)"
echo ""

python -m detection.ffpp_svd_vae.evaluate \
    --batch 64 \
    --num_workers 8

echo ""
echo "Done at $(date)"
