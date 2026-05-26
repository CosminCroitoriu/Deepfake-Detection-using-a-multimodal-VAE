#!/bin/bash
#SBATCH --job-name=eval_gated_variants
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/eval_gated_variants_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/eval_gated_variants_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"
source "$PROJECT_DIR/venv/bin/activate"

echo "Gated VAE evaluation — 6 scoring strategies (no retraining)"
echo "  Time: $(date)"
echo ""

python -m detection.gated_vae.evaluate_score_variants \
    --batch 16 \
    --num_workers 8

echo ""
echo "Done at $(date)"
