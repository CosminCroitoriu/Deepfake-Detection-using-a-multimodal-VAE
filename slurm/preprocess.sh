#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/preprocess_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/preprocess_%j.err
#SBATCH --partition=haswell
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Starting preprocessing (512x512)"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  Time        : $(date)"
echo ""

python generation/data_prep/preprocess.py --include_extra

echo ""
echo "Done at $(date)"
