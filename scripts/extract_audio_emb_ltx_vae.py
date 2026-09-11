#!/usr/bin/env python3
"""
Extract audio embeddings using LTX-2 Audio VAE encoder.

Output shape: (T, 128) where T ≈ duration × 25fps
Replaces wav2vec embeddings (T, 12, 768) with language-agnostic VAE latents.

Usage:
    python scripts/extract_audio_emb_ltx_vae.py \
        --data_dir /scratch/li_qiany_neu/hallo3_data \
        --audio_dir /scratch/li_qiany_neu/TalkVid/clips_flat/audios \
        --checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
from einops import rearrange
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "hallo3"))
from sgm.models.ltx_audio_vae import AudioEncoder, AudioProcessor, Audio, AudioLatentShape
from sgm.models.ltx_audio_vae.normalization import NormType
from sgm.models.ltx_audio_vae.causality_axis import CausalityAxis
from sgm.utils.util import get_fps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def load_audio_encoder(checkpoint_path: str, device: str = "cuda") -> AudioEncoder:
    """Load AudioEncoder from safetensors checkpoint."""
    encoder = AudioEncoder(
        ch=128,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions={8, 16, 32},
        resolution=256,
        z_channels=8,
        double_z=True,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=2,
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        sample_rate=16000,
        mel_hop_length=160,
        n_fft=1024,
        is_causal=True,
        mel_bins=64,
        mid_block_add_attention=False,
    )

    all_weights = load_file(checkpoint_path)
    encoder_state_dict = {}
    for key, value in all_weights.items():
        if key.startswith("audio_vae.encoder."):
            encoder_state_dict[key.replace("audio_vae.encoder.", "")] = value
        elif key.startswith("audio_vae.per_channel_statistics."):
            encoder_state_dict[key.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")] = value

    encoder.load_state_dict(encoder_state_dict, strict=True)
    encoder = encoder.to(device=device, dtype=torch.bfloat16)
    encoder.eval()
    return encoder


def extract_embedding(
    audio_path: str,
    encoder: AudioEncoder,
    processor: AudioProcessor,
    device: str,
) -> torch.Tensor:
    """Extract audio embedding from audio file.

    Returns:
        Tensor of shape (T, 128) where T ≈ duration × 25fps
    """
    # Load audio as mono, then duplicate to stereo
    speech_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    # (samples,) → (1, 2, samples) stereo
    waveform = torch.from_numpy(speech_array).float().unsqueeze(0).unsqueeze(0)
    waveform = waveform.repeat(1, 2, 1)  # mono → stereo
    waveform = waveform.to(device=device)

    audio = Audio(waveform=waveform, sampling_rate=16000)

    with torch.no_grad():
        mel = processor.waveform_to_mel(audio)
        latent = encoder(mel.to(dtype=torch.bfloat16))
        # latent: (1, 8, T_latent, 16)
        tokens = rearrange(latent, "b c t f -> b t (c f)")
        # tokens: (1, T_latent, 128)

    audio_emb = tokens.squeeze(0).float().cpu()
    # (T, 128)
    return audio_emb


def get_video_paths(data_dir: Path, parallelism: int, rank: int) -> List[Path]:
    video_dir = data_dir / "videos"
    all_videos = sorted(p for p in video_dir.iterdir() if p.suffix == ".mp4")
    return [all_videos[i] for i in range(len(all_videos)) if i % parallelism == rank]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="hallo3_data root (contains videos/, audio_emb/)")
    parser.add_argument("--audio_dir", type=Path, required=True,
                        help="Directory containing .m4a or .wav audio files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ltx-2.5-audio-vae-bf16.safetensors")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logger = logging.getLogger()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={args.device}, rank={args.rank}/{args.parallelism}")

    # Load encoder
    logger.info(f"Loading AudioEncoder from {args.checkpoint}")
    encoder = load_audio_encoder(args.checkpoint, device=args.device)

    processor = AudioProcessor(
        target_sample_rate=16000,
        mel_bins=64,
        mel_hop_length=160,
        n_fft=1024,
    ).to(device=args.device)

    # Get video list
    video_paths = get_video_paths(args.data_dir, args.parallelism, args.rank)
    logger.info(f"rank={args.rank}: {len(video_paths)} videos to process")

    out_dir = args.data_dir / "audio_emb"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = fail_no_audio = fail_other = 0

    for video_path in tqdm(video_paths, desc=f"rank{args.rank}"):
        out_path = out_dir / f"{video_path.stem}.pt"

        # Find matching audio file (.m4a or .wav)
        audio_path = None
        for ext in [".m4a", ".wav", ".mp3", ".flac"]:
            candidate = args.audio_dir / f"{video_path.stem}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if audio_path is None:
            logger.warning(f"No audio found: {video_path.stem}")
            fail_no_audio += 1
            continue

        try:
            audio_emb = extract_embedding(str(audio_path), encoder, processor, args.device)
            # audio_emb shape: (T, 128)
            assert audio_emb.ndim == 2 and audio_emb.shape[1] == 128, \
                f"Unexpected shape {audio_emb.shape}"
            torch.save(audio_emb, out_path)
            ok += 1
        except Exception as e:
            logger.error(f"Failed {video_path.stem}: {e}")
            fail_other += 1

    logger.info(
        f"rank={args.rank} done — ok={ok}, "
        f"no_audio={fail_no_audio}, errors={fail_other}"
    )


if __name__ == "__main__":
    main()
