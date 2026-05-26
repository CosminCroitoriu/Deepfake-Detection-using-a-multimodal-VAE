#!/bin/bash
#
# Set up Sarkar's reference implementation for a faithful (and unmodified) run
# on our preprocessed FF++ data.
#
# We do NOT touch their code. We create the directory tree they expect and
# symlink our preprocessed face crops into it. Frame-level 80/20 random split
# matches the methodology in their public code.
#
# Run this ONCE before submitting the training slurm job.

set -e

PROJECT_DIR="/export/home/acs/stud/c/cosmin.croitoriu/Deepfake-Detection-using-a-multimodal-VAE"
SARKAR_DIR="$PROJECT_DIR/sarkar_repro"
REPO_DIR="$SARKAR_DIR/Deepfake-detection"

echo "Setting up Sarkar reproduction at $SARKAR_DIR"

# ---- 1. Clone the repo -------------------------------------------------------
mkdir -p "$SARKAR_DIR"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Sarkar's repo ..."
    git clone https://github.com/syamantak-sarkar/Deepfake-detection "$REPO_DIR"
else
    echo "Repo already cloned, skipping."
fi

# ---- 2. Create the directory tree their code reads from ----------------------
FF="$REPO_DIR/DeepFake/FaceForensics++"
DFDCP="$REPO_DIR/DeepFake/DFDCP"
MODELS="$REPO_DIR/DeepFake/models"

mkdir -p "$FF/original_sequences_split/youtube/train/c23/frames"
mkdir -p "$FF/original_sequences_split/youtube/test/c23/frames"
mkdir -p "$FF/manipulated_sequences_split/FaceSwap/train/c23/frames"
mkdir -p "$FF/manipulated_sequences_split/FaceSwap/test/c23/frames"
mkdir -p "$DFDCP/validation/real"
mkdir -p "$DFDCP/validation/fake"
mkdir -p "$DFDCP/test/real"
mkdir -p "$DFDCP/test/fake"
mkdir -p "$MODELS"

# ---- 3. Symlink our preprocessed FF++ frames into the expected paths --------
echo "Splitting frames 80/20 random (frame-level, matches their methodology) ..."

python3 <<PYTHON
import random
from pathlib import Path

PROJECT = Path("$PROJECT_DIR")
ORIGINAL = PROJECT / "data" / "ffpp" / "original"
FAKESWAP = PROJECT / "data" / "ffpp" / "FaceSwap"

FF = Path("$FF")
DFDCP = Path("$DFDCP")

if not ORIGINAL.is_dir() or not FAKESWAP.is_dir():
    raise SystemExit(
        f"Expected preprocessed FF++ data at {ORIGINAL} and {FAKESWAP}.\n"
        f"Run slurm/preprocess_ffpp.sh first if you haven't already."
    )

def link_subset(srcs, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in srcs:
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        # Use ABSOLUTE source path so the symlink remains valid regardless of
        # the working directory their code runs from.
        dst.symlink_to(src.resolve())

random.seed(42)  # frame-level random — exactly the leaky methodology

# Originals
all_real = sorted(ORIGINAL.glob("*.png"))
random.shuffle(all_real)
n_train = int(0.8 * len(all_real))
train_real, test_real = all_real[:n_train], all_real[n_train:]
print(f"  Originals: {len(train_real)} train / {len(test_real)} test")
link_subset(train_real, FF / "original_sequences_split/youtube/train/c23/frames")
link_subset(test_real,  FF / "original_sequences_split/youtube/test/c23/frames")

# FaceSwap
all_fake = sorted(FAKESWAP.glob("*.png"))
random.shuffle(all_fake)
n_train_f = int(0.8 * len(all_fake))
train_fake, test_fake = all_fake[:n_train_f], all_fake[n_train_f:]
print(f"  FaceSwap : {len(train_fake)} train / {len(test_fake)} test")
link_subset(train_fake, FF / "manipulated_sequences_split/FaceSwap/train/c23/frames")
link_subset(test_fake,  FF / "manipulated_sequences_split/FaceSwap/test/c23/frames")

# Their TEST script reads from DFDCP/validation and DFDCP/test
# We don't have DFDCP. To run their unmodified test script we point those
# paths at our FF++ data instead — the script doesn't know the difference.
# Use a SUBSET of the test_real / test_fake pool for validation (threshold
# selection) and the rest for test (final AUC).
n_val_split = max(1, len(test_real) // 5)  # 20% of test → validation
val_real, test_real_eval = test_real[:n_val_split], test_real[n_val_split:]
val_fake, test_fake_eval = test_fake[:n_val_split], test_fake[n_val_split:]

link_subset(val_real,        DFDCP / "validation/real")
link_subset(val_fake,        DFDCP / "validation/fake")
link_subset(test_real_eval,  DFDCP / "test/real")
link_subset(test_fake_eval,  DFDCP / "test/fake")
print(f"  DFDCP val : {len(val_real)} real / {len(val_fake)} fake")
print(f"  DFDCP test: {len(test_real_eval)} real / {len(test_fake_eval)} fake")
print("Symlinks created.")
PYTHON

# ---- 4. Verify --------------------------------------------------------------
echo ""
echo "Verification — file counts per target directory:"
for d in \
    "$FF/original_sequences_split/youtube/train/c23/frames" \
    "$FF/original_sequences_split/youtube/test/c23/frames" \
    "$FF/manipulated_sequences_split/FaceSwap/train/c23/frames" \
    "$FF/manipulated_sequences_split/FaceSwap/test/c23/frames" \
    "$DFDCP/validation/real" \
    "$DFDCP/validation/fake" \
    "$DFDCP/test/real" \
    "$DFDCP/test/fake" \
; do
    n=$(ls "$d" 2>/dev/null | wc -l)
    printf "  %-90s %6d files\n" "${d##$REPO_DIR/}" "$n"
done

echo ""
echo "Setup complete. To run their training + test:"
echo "  sbatch slurm/sarkar_train.sh"
echo "  # ...wait for completion..."
echo "  sbatch slurm/sarkar_test.sh"
