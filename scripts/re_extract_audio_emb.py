"""Re-extract audio embeddings using a new wav2vec2 encoder (e.g. Omnilingual).

Reads the training JSON, extracts audio embeddings with the new encoder,
saves them to a new directory, and writes an updated JSON.

Usage:
    conda activate hallo3
    python scripts/re_extract_audio_emb.py \
        --json data/talkvid_4k_clean.json \
        --encoder /scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m \
        --output-dir /scratch/li_qiany_neu/talkvid_4k/audio_emb_omnilingual \
        --output-json data/talkvid_4k_omnilingual.json
"""
import argparse
import json
import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from sgm.models.wav2vec import Wav2VecModel


def extract_one(audio_path, encoder, feature_extractor, device, sample_rate=16000, fps=25.0):
    """Extract audio embedding for a single audio file.

    Returns:
        audio_emb: (T, 12, dim) tensor, where dim is the encoder's hidden size
    """
    speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    audio_feature = np.squeeze(
        feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    seq_len = math.ceil(len(audio_feature) / sample_rate * fps)

    audio_feature = torch.from_numpy(audio_feature).float().to(device).unsqueeze(0)

    with torch.no_grad():
        embeddings = encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)

    # Take last 12 of N transformer layers (skip embedding layer at index 0)
    all_hidden = embeddings.hidden_states[1:]
    num_layers = len(all_hidden)
    use_layers = 12
    selected = all_hidden[num_layers - use_layers:]
    audio_emb = torch.stack(selected, dim=1).squeeze(0)  # (T, 12, dim)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")   # (T, 12, dim)

    return audio_emb.cpu().detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Input training JSON")
    parser.add_argument("--encoder", required=True, help="Path to HF wav2vec2 model")
    parser.add_argument("--output-dir", required=True, help="Directory to save new .pt embeddings")
    parser.add_argument("--output-json", required=True, help="Path to write updated JSON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load encoder
    print(f"Loading encoder from {args.encoder} ...")
    encoder = Wav2VecModel.from_pretrained(args.encoder, local_files_only=True).to(args.device)
    encoder.feature_extractor._freeze_parameters()
    encoder.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.encoder, local_files_only=True)

    num_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    # Check hidden size from a dummy forward or config
    hidden_size = encoder.config.hidden_size
    num_layers = encoder.config.num_hidden_layers
    print(f"  {num_params:.1f}M params, {num_layers} layers, {hidden_size} dim")
    print(f"  Will take last 12 of {num_layers} layers")

    # Load JSON
    with open(args.json) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {args.json}")

    # Extract
    updated_data = []
    skipped = 0
    for item in tqdm(data, desc="Extracting audio embeddings"):
        video_path = item["video_path"]
        stem = os.path.splitext(os.path.basename(video_path))[0]

        # Find audio file
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "audios")
        audio_candidates = [
            os.path.join(audio_dir, f"{stem}.{ext}")
            for ext in ["wav", "m4a", "mp3", "flac"]
        ]
        audio_path = None
        for c in audio_candidates:
            if os.path.exists(c):
                audio_path = c
                break

        if audio_path is None:
            # Try extracting from video
            tqdm.write(f"  SKIP {stem}: no audio file found in {audio_dir}")
            skipped += 1
            continue

        out_path = os.path.join(args.output_dir, f"{stem}.pt")

        if os.path.exists(out_path):
            # Already extracted, just update JSON
            new_item = dict(item)
            new_item["vocals_emb_base_all"] = out_path
            updated_data.append(new_item)
            continue

        try:
            audio_emb = extract_one(
                audio_path, encoder, feature_extractor,
                args.device, args.sample_rate, args.fps
            )
            torch.save(audio_emb, out_path)

            new_item = dict(item)
            new_item["vocals_emb_base_all"] = out_path
            updated_data.append(new_item)
        except Exception as e:
            tqdm.write(f"  ERROR {stem}: {e}")
            skipped += 1

    # Save updated JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Extracted {len(updated_data)} / {len(data)} (skipped {skipped})")
    print(f"  Embeddings: {args.output_dir}")
    print(f"  Updated JSON: {args.output_json}")
    if updated_data:
        sample = torch.load(os.path.join(args.output_dir,
                            os.path.basename(updated_data[0]["vocals_emb_base_all"])),
                            map_location="cpu")
        print(f"  Sample shape: {sample.shape} (expected: T, 12, {hidden_size})")


if __name__ == "__main__":
    main()
