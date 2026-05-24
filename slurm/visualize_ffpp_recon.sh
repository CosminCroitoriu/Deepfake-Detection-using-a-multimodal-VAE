#!/bin/bash
#SBATCH --job-name=viz_ffpp_recon
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/viz_ffpp_recon_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/viz_ffpp_recon_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Generating reconstruction visualisation"
echo "  Time: $(date)"
echo ""

python -m detection.ffpp_svd_vae.visualize_reconstruction \
    --manipulation Deepfakes \
    --n_samples 4

echo ""
echo "Done at $(date)"
