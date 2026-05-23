#!/usr/bin/env python3
"""
Preprocess FaceForensics++ c23 videos into face crops, matching the Sarkar
paper protocol: MTCNN face detection, +50% padding, resize to target size,
saved as PNG.

Output structure:
  data/ffpp/<manipulation>/<vid_id>_<frame_idx>.png

  manipulation in: original, Deepfakes, Face2Face, FaceSwap,
                   NeuralTextures, FaceShifter, DeepFakeDetection

Usage:
  python preprocess_ffpp.py
  python preprocess_ffpp.py --manipulations original Deepfakes --frames_per_video 16
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


def extract_frames(video_path: Path, num_frames: int):
    """Sample `num_frames` evenly-spaced frames from a video. Returns list of (frame_idx, rgb_array)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    if total < num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            frames.append((int(idx), cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def crop_face(frame_rgb: np.ndarray, mtcnn: MTCNN, padding_ratio: float, target_size: int):
    """Return PIL.Image of the largest face cropped with `padding_ratio` extra context, resized to target_size."""
    pil = Image.fromarray(frame_rgb)
    boxes, _ = mtcnn.detect(pil)
    if boxes is None or len(boxes) == 0:
        return None

    # largest face by area
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * padding_ratio, h * padding_ratio

    H, W = frame_rgb.shape[:2]
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(W, int(x2 + pad_w))
    y2 = min(H, int(y2 + pad_h))

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return Image.fromarray(crop).resize((target_size, target_size), Image.LANCZOS)


def process_manipulation(
    raw_dir: Path, out_dir: Path, manipulation: str,
    mtcnn: MTCNN, frames_per_video: int, target_size: int, padding_ratio: float,
):
    in_folder = raw_dir / manipulation
    out_folder = out_dir / manipulation
    out_folder.mkdir(parents=True, exist_ok=True)

    videos = sorted(in_folder.glob("*.mp4"))
    if not videos:
        print(f"  WARNING: no mp4s in {in_folder}, skipping")
        return

    n_saved = 0
    n_failed = 0
    for vid in tqdm(videos, desc=manipulation):
        vid_id = vid.stem
        frames = extract_frames(vid, frames_per_video)
        for frame_idx, frame_rgb in frames:
            out_path = out_folder / f"{vid_id}_{frame_idx:04d}.png"
            if out_path.exists():
                continue
            crop = crop_face(frame_rgb, mtcnn, padding_ratio, target_size)
            if crop is None:
                n_failed += 1
                continue
            crop.save(out_path)
            n_saved += 1
    print(f"  {manipulation}: saved {n_saved}, failed-detect {n_failed}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw_dir", default=str(SCRIPT_DIR / "../../data/ffpp_raw"))
    parser.add_argument("--out_dir", default=str(SCRIPT_DIR / "../../data/ffpp"))
    parser.add_argument("--frames_per_video", type=int, default=32)
    parser.add_argument("--target_size", type=int, default=256,
                        help="Output crop size. Default 256 matches Sarkar's paper; "
                             "do not raise unless faces in your videos are actually ≥this size")
    parser.add_argument("--padding_ratio", type=float, default=0.5)
    parser.add_argument(
        "--manipulations", nargs="+",
        default=["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    mtcnn = MTCNN(device=device, keep_all=False, post_process=False)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    for manip in args.manipulations:
        print(f"\nProcessing {manip} ...")
        process_manipulation(
            raw_dir, out_dir, manip, mtcnn,
            args.frames_per_video, args.target_size, args.padding_ratio,
        )

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()
