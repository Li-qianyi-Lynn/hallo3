"""Fit adapter initialization by linear regression between two wav2vec2 models.

Extracts features from both wav2vec2-base and Omnilingual on the same audio files,
then fits a linear regression: Omnilingual(1024) → wav2vec2-base(768).
The resulting W, b are saved as adapter_init.pt for use in training.

Usage:
    cd ~/hallo3/hallo3
    PYTHONPATH=. python ../scripts/fit_adapter_init.py \
        --base-model /scratch/li_qiany_neu/pretrained_models/wav2vec/wav2vec2-base-960h \
        --omni-model /scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m \
        --json ../data/talkvid_train_2700.json \
        --n 50 --seed 0 \
        --output ../configs/adapter_init.pt
"""
import argparse
import json
import math
import os
import random

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from sgm.models.wav2vec import Wav2VecModel


def extract_features(audio_path, encoder, feature_extractor, device, sample_rate=16000, fps=25.0):
    """Extract last-12-layer features from a wav2vec2 model."""
    speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    audio_feature = np.squeeze(
        feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    seq_len = math.ceil(len(audio_feature) / sample_rate * fps)
    audio_feature = torch.from_numpy(audio_feature).float().to(device).unsqueeze(0)

    with torch.no_grad():
        embeddings = encoder(audio_feature, seq_len=seq_len, output_hidden_states=True)

    all_hidden = embeddings.hidden_states[1:]  # skip embedding layer
    num_layers = len(all_hidden)
    use_layers = min(12, num_layers)
    selected = all_hidden[num_layers - use_layers:]
    # (1, T, 12, dim)
    feats = torch.stack(selected, dim=2).squeeze(0)  # (T, 12, dim)
    return feats.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Path to wav2vec2-base-960h")
    parser.add_argument("--omni-model", required=True, help="Path to omnilingual-w2v2-300m")
    parser.add_argument("--json", required=True, help="Training JSON to sample audio from")
    parser.add_argument("--n", type=int, default=50, help="Number of audio files to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="Output path for adapter_init.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    # Load training JSON to find audio files
    with open(args.json) as f:
        data = json.load(f)

    rng = random.Random(args.seed)
    rng.shuffle(data)

    # Find audio files
    audio_paths = []
    for item in data:
        video_path = item["video_path"]
        stem = os.path.splitext(os.path.basename(video_path))[0]
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "audios")
        for ext in ["wav", "m4a", "mp3", "flac"]:
            ap = os.path.join(audio_dir, f"{stem}.{ext}")
            if os.path.exists(ap):
                audio_paths.append(ap)
                break
        if len(audio_paths) >= args.n:
            break

    print(f"Found {len(audio_paths)} audio files")

    # Load both models
    print(f"Loading wav2vec2-base: {args.base_model}")
    base_encoder = Wav2VecModel.from_pretrained(args.base_model, local_files_only=True).to(args.device)
    base_encoder.eval()
    base_encoder.feature_extractor._freeze_parameters()
    base_fe = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model, local_files_only=True)
    print(f"  hidden_size={base_encoder.config.hidden_size}, layers={base_encoder.config.num_hidden_layers}")

    print(f"Loading Omnilingual: {args.omni_model}")
    omni_encoder = Wav2VecModel.from_pretrained(args.omni_model, local_files_only=True).to(args.device)
    omni_encoder.eval()
    omni_encoder.feature_extractor._freeze_parameters()
    omni_fe = Wav2Vec2FeatureExtractor.from_pretrained(args.omni_model, local_files_only=True)
    print(f"  hidden_size={omni_encoder.config.hidden_size}, layers={omni_encoder.config.num_hidden_layers}")

    # Extract features from both models
    all_base = []
    all_omni = []
    for ap in tqdm(audio_paths, desc="Extracting features"):
        try:
            base_feats = extract_features(ap, base_encoder, base_fe, args.device, args.sample_rate)
            omni_feats = extract_features(ap, omni_encoder, omni_fe, args.device, args.sample_rate)

            # Align lengths (may differ slightly due to different CNN strides)
            min_t = min(base_feats.shape[0], omni_feats.shape[0])
            base_feats = base_feats[:min_t]  # (T, 12, 768)
            omni_feats = omni_feats[:min_t]  # (T, 12, 1024)

            # Flatten time and layers: (T*12, dim)
            all_base.append(base_feats.reshape(-1, base_feats.shape[-1]))
            all_omni.append(omni_feats.reshape(-1, omni_feats.shape[-1]))
        except Exception as e:
            tqdm.write(f"  Skip {ap}: {e}")

    # Concatenate all samples
    X = torch.cat(all_omni, dim=0).float()   # (N, 1024)  — input
    Y = torch.cat(all_base, dim=0).float()    # (N, 768)   — target

    print(f"\nRegression data: X {X.shape}, Y {Y.shape}")
    print(f"  Total frames×layers: {X.shape[0]}")

    # Linear regression: Y = X @ W.T + b
    # Solve via least squares: [X, 1] @ [W.T; b.T] = Y
    ones = torch.ones(X.shape[0], 1)
    X_aug = torch.cat([X, ones], dim=1)  # (N, 1025)

    # lstsq: X_aug @ params = Y  →  params = (X_aug.T @ X_aug)^-1 @ X_aug.T @ Y
    print("Fitting linear regression...")
    result = torch.linalg.lstsq(X_aug, Y)
    params = result.solution  # (1025, 768)

    W = params[:1024, :].T    # (768, 1024) — Linear weight shape
    b = params[1024, :]        # (768,)      — Linear bias shape

    # Evaluate fit quality
    Y_pred = X @ W.T + b
    residual = (Y - Y_pred).norm() / Y.norm()
    r2 = 1 - ((Y - Y_pred) ** 2).sum() / ((Y - Y.mean(dim=0)) ** 2).sum()
    print(f"  Relative residual: {residual:.4f}")
    print(f"  R² score: {r2:.4f}")
    print(f"  W shape: {W.shape}, b shape: {b.shape}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"weight": W, "bias": b}, args.output)
    print(f"\nSaved adapter init to {args.output}")
    print(f"  Load with:")
    print(f"    init = torch.load('{args.output}')")
    print(f"    model.audio_proj.adapter.weight.data = init['weight']")
    print(f"    model.audio_proj.adapter.bias.data = init['bias']")


if __name__ == "__main__":
    main()
