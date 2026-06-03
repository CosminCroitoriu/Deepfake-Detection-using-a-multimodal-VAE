#!/bin/bash
#SBATCH --job-name=eval_resnet50
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/eval_resnet50_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/eval_resnet50_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"
source "$PROJECT_DIR/venv/bin/activate"

echo "Starting ResNet-50 evaluation"
echo "  Time: $(date)"
echo ""

python -m detection.pretrained_cnn.evaluate \
    --arch resnet50 \
    --batch 64 \
    --num_workers 8

echo ""
echo "Done at $(date)"
