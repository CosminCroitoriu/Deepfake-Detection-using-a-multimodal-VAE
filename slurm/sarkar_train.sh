#!/bin/bash
#SBATCH --job-name=sarkar_train
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/sarkar_train_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/sarkar_train_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
SARKAR_DIR="$PROJECT_DIR/sarkar_repro/Deepfake-detection"

cd "$SARKAR_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"
source "$PROJECT_DIR/venv/bin/activate"

echo "Running Sarkar's deepfake_train.py UNMODIFIED on our FF++ data"
echo "  Working dir : $(pwd)"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""
echo "Data folder check:"
for d in \
    "DeepFake/FaceForensics++/original_sequences_split/youtube/train/c23/frames" \
    "DeepFake/FaceForensics++/manipulated_sequences_split/FaceSwap/train/c23/frames" \
; do
    n=$(ls "$d" 2>/dev/null | wc -l)
    printf "  %-90s %6d files\n" "$d" "$n"
done
echo ""

# Run their training script with absolutely no modifications.
# They `cd` to their repo dir so the relative paths like
# 'DeepFake/FaceForensics++/...' and 'DeepFake/models/best_unet_vae.pth' resolve.
python deepfake_train.py

echo ""
echo "Training finished at $(date)"
echo ""

# Sanity check — what model files did the script produce?
echo "Files in DeepFake/models/ after training:"
ls -lh DeepFake/models/ 2>/dev/null || echo "  (empty)"

# Also check the repo root for stray .pth saves
echo ""
echo "Any .pth files in the repo root:"
ls -lh *.pth 2>/dev/null || echo "  (none)"
