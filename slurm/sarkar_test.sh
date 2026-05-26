#!/bin/bash
#SBATCH --job-name=sarkar_test
#SBATCH --output=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/sarkar_test_%j.out
#SBATCH --error=/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE/slurm/logs/sarkar_test_%j.err
#SBATCH --partition=dgxa100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
SARKAR_DIR="$PROJECT_DIR/sarkar_repro/Deepfake-detection"

cd "$SARKAR_DIR"

mkdir -p "$PROJECT_DIR/slurm/logs"
source "$PROJECT_DIR/venv/bin/activate"

echo "Running Sarkar's deepfake_test.py UNMODIFIED"
echo "  Working dir : $(pwd)"
echo "  Time        : $(date)"
echo ""

# The test script loads 'DeepFake/models/best_unet_vae.pth'.
# Verify it exists before running so we get a clean error if training didn't
# produce that exact filename.
if [ ! -f "DeepFake/models/best_unet_vae.pth" ]; then
    echo "ERROR: DeepFake/models/best_unet_vae.pth does not exist."
    echo "If the training script saved the model under a different name, look at:"
    echo "  ls -lh DeepFake/models/"
    echo "  ls -lh *.pth"
    echo "Then symlink/copy it to DeepFake/models/best_unet_vae.pth before re-running."
    exit 1
fi

# DFDCP folder check — the test script reads from these
echo "DFDCP folder check (we symlinked our FF++ test data here):"
for d in \
    "DeepFake/DFDCP/validation/real" \
    "DeepFake/DFDCP/validation/fake" \
    "DeepFake/DFDCP/test/real" \
    "DeepFake/DFDCP/test/fake" \
; do
    n=$(ls "$d" 2>/dev/null | wc -l)
    printf "  %-50s %6d files\n" "$d" "$n"
done
echo ""

# Run their test script unmodified.
python deepfake_test.py

echo ""
echo "Test finished at $(date)"
