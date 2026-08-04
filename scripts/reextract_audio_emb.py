#!/usr/bin/env python3
"""
Re-extract audio embeddings for all TalkVid videos using Hallo3's AudioProcessor.

Problem: convert_talkvid_to_hallo3.py copied TalkVid's short_clip_aud_embeds directly.
         Those are (T, 1, 768) — single wav2vec layer.
         Hallo3's audio_proj expects (T, 12, 768) — all 12 hidden layers.

Fix: re-extract from each video's audio track using AudioProcessor(only_last_features=False).
     Overwrites audio_emb/*.pt in-place. talkvid.json does NOT need to be regenerated.

Usage (single process):
    python scripts/reextract_audio_emb.py --data_dir /scratch/li.qianyi/hallo3_data

Usage (4-way parallel, run rank 0-3 on separate GPUs/jobs):
    python scripts/reextract_audio_emb.py --data_dir /scratch/li.qianyi/hallo3_data \
        --parallelism 4 --rank 0
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

# Run from the hallo3 project root
sys.path.insert(0, str(Path(__file__).parent.parent / "hallo3"))
from sgm.utils.audio_processor import AudioProcessor
from sgm.utils.util import extract_audio_from_videos, get_fps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def get_video_paths(data_dir: Path, parallelism: int, rank: int) -> List[Path]:
    video_dir = data_dir / "videos"
    all_videos = sorted(p for p in video_dir.iterdir() if p.suffix == ".mp4")
    return [all_videos[i] for i in range(len(all_videos)) if i % parallelism == rank]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Root data dir (must contain videos/ and audio_emb/ subdirectories)"
    )
    parser.add_argument(
        "--wav2vec_model_path", type=str,
        default="pretrained_models/wav2vec/wav2vec2-base-960h",
        help="Path to wav2vec2-base-960h (relative to project root or absolute)"
    )
    parser.add_argument("--parallelism", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--rank", type=int, default=0,
                        help="Index of this worker (0-indexed)")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip files that already have shape (T, 12, 768)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for wav2vec inference: 'cuda:0', 'cpu', etc. "
             "Default: cuda:0 if available, else cpu"
    )
    args = parser.parse_args()

    logger = logging.getLogger()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {args.device}")

    audio_emb_dir = args.data_dir / "audio_emb"
    audio_emb_dir.mkdir(exist_ok=True)

    # Initialize AudioProcessor — no vocal separator (TalkVid clips are already speech-only)
    audio_processor = AudioProcessor(
        sample_rate=16000,
        wav2vec_model_path=args.wav2vec_model_path,
        only_last_features=False,       # all 12 hidden layers → (T, 12, 768)
        audio_separator_model_path=None,
        audio_separator_model_name=None,
        device=args.device,
    )

    video_paths = get_video_paths(args.data_dir, args.parallelism, args.rank)
    logger.info(f"rank={args.rank}/{args.parallelism}: {len(video_paths)} videos to process")

    ok = skip = fail_no_audio = fail_other = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for video_path in tqdm(video_paths, desc=f"rank{args.rank}"):
            out_path = audio_emb_dir / f"{video_path.stem}.pt"

            # Optionally skip already-correct files
            if args.skip_existing and out_path.exists():
                try:
                    emb = torch.load(out_path, map_location="cpu", weights_only=True)
                    if emb.ndim == 3 and emb.shape[1] == 12:
                        skip += 1
                        continue
                except Exception:
                    pass  # corrupted → re-extract

            try:
                fps = get_fps(video_path)

                # Extract audio track from the video to a temp .wav
                tmp_wav = Path(tmp_dir) / f"{video_path.stem}.wav"
                result = extract_audio_from_videos(str(video_path), str(tmp_wav))

                if result is None or not tmp_wav.exists():
                    # Video has no audio track — save a zero-tensor placeholder
                    # Stage2_SFTDataset will still run; audio conditioning will be zeroed out
                    logger.warning(f"No audio track: {video_path.name} — saving zeros")
                    torch.save(torch.zeros(1, 12, 768), out_path)
                    fail_no_audio += 1
                    continue

                # Extract all 12 hidden layers → shape (T, 12, 768)
                audio_emb, _ = audio_processor.preprocess(str(tmp_wav), fps=fps)
                assert audio_emb.ndim == 3 and audio_emb.shape[1] == 12, \
                    f"Unexpected shape {audio_emb.shape}"

                torch.save(audio_emb, out_path)
                ok += 1

            except Exception as e:
                logger.error(f"Failed {video_path.name}: {e}")
                fail_other += 1

    logger.info(
        f"rank={args.rank} done — "
        f"ok={ok}, skipped={skip}, no_audio={fail_no_audio}, errors={fail_other}"
    )


if __name__ == "__main__":
    main()
