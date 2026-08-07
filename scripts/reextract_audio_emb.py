#!/usr/bin/env python3
"""
Re-extract audio embeddings using .m4a files from TalkVid clips_flat/audios/.

Problem: convert_talkvid_to_hallo3.py stored (T, 1, 768) single-layer embeddings.
         Hallo3's audio_proj expects (T, 12, 768) — all 12 wav2vec hidden layers.
         The converted videos have no audio track, but .m4a files are available.

Fix: for each video in hallo3_data/videos/, find the matching .m4a in clips_flat/audios/,
     extract 12-layer wav2vec embeddings, overwrite audio_emb/*.pt in-place.
     talkvid.json does NOT need to be regenerated (paths unchanged).

Usage:
    python scripts/reextract_audio_emb.py \
        --data_dir /scratch/li.qianyi/hallo3_data \
        --audio_dir /scratch/li.qianyi/TalkVid/clips_flat/audios \
        --parallelism 4 --rank 0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "hallo3"))
from sgm.utils.audio_processor import AudioProcessor
from sgm.utils.util import get_fps

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
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="hallo3_data root (contains videos/, audio_emb/)")
    parser.add_argument("--audio_dir", type=Path, required=True,
                        help="clips_flat/audios/ containing .m4a files")
    parser.add_argument("--wav2vec_model_path", type=str,
                        default="pretrained_models/wav2vec/wav2vec2-base-960h")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files that already have shape (T, 12, 768)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda:0 / cpu (default: cuda:0 if available)")
    args = parser.parse_args()

    logger = logging.getLogger()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={args.device}, rank={args.rank}/{args.parallelism}")

    audio_processor = AudioProcessor(
        sample_rate=16000,
        wav2vec_model_path=args.wav2vec_model_path,
        only_last_features=False,   # all 12 hidden layers → (T, 12, 768)
        audio_separator_model_path=None,
        audio_separator_model_name=None,
        device=args.device,
    )

    video_paths = get_video_paths(args.data_dir, args.parallelism, args.rank)
    logger.info(f"rank={args.rank}: {len(video_paths)} videos to process")

    ok = skip = fail_no_m4a = fail_other = 0

    for video_path in tqdm(video_paths, desc=f"rank{args.rank}"):
        out_path = args.data_dir / "audio_emb" / f"{video_path.stem}.pt"

        if args.skip_existing and out_path.exists():
            try:
                emb = torch.load(out_path, map_location="cpu", weights_only=True)
                if emb.ndim == 3 and emb.shape[1] == 12:
                    skip += 1
                    continue
            except Exception:
                pass

        m4a_path = args.audio_dir / f"{video_path.stem}.m4a"
        if not m4a_path.exists():
            logger.warning(f"No m4a: {video_path.stem}")
            fail_no_m4a += 1
            continue

        try:
            fps = get_fps(video_path)
            audio_emb, _ = audio_processor.preprocess(str(m4a_path), fps=fps)
            # audio_emb shape: (T, 12, 768)
            assert audio_emb.ndim == 3 and audio_emb.shape[1] == 12, \
                f"Unexpected shape {audio_emb.shape}"
            torch.save(audio_emb, out_path)
            ok += 1

        except Exception as e:
            logger.error(f"Failed {video_path.stem}: {e}")
            fail_other += 1

    logger.info(
        f"rank={args.rank} done — ok={ok}, skipped={skip}, "
        f"no_m4a={fail_no_m4a}, errors={fail_other}"
    )


if __name__ == "__main__":
    main()
