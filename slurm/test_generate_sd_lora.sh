#!/bin/bash
#SBATCH --job-name=test_gen_sd_lora
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/test_gen_sd_lora_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/test_gen_sd_lora_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"

source "$PROJECT_DIR/venv/bin/activate"

echo "Starting SD LoRA test generation (3 images/class)"
echo "  Project dir : $PROJECT_DIR"
echo "  Python      : $(which python)"
echo "  GPU         : $CUDA_VISIBLE_DEVICES"
echo "  Time        : $(date)"
echo ""

python generation/sd_lora/generate_sd_lora.py \
    --data_dir ./data \
    --captions_file ./data/captions.json \
    --adapter_path ./checkpoints/sd_lora/final \
    --n_images 3 \
    --steps 30 \
    --guidance_scale 7.5 \
    --seed 42

echo ""
echo "Done at $(date)"
echo "Output: data/fake/sd_lora/"
