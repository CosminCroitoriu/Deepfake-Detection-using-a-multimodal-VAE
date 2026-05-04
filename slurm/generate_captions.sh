#!/bin/bash
#SBATCH --job-name=gen_captions
#SBATCH --output=slurm/logs/gen_captions_%j.out
#SBATCH --error=slurm/logs/gen_captions_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p slurm/logs

source venv/bin/activate

echo "Starting VLM captioning"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

python generation/data_prep/generate_captions.py \
    --data_dir ./data \
    --captions_out ./data/captions.json \
    --labels_out ./data/labels.json \
    --model llava \
    --resume

echo ""
echo "Done at $(date)"
