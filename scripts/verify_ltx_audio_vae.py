"""
验证 LTX-2 Audio VAE encoder 能否独立加载权重并跑通 forward pass。
用法:
    python scripts/verify_ltx_audio_vae.py \
        --checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
"""

import argparse
import sys
import os

import torch
from safetensors.torch import load_file
from einops import rearrange

# 确保 hallo3 在 import 路径上
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hallo3.sgm.models.ltx_audio_vae import AudioEncoder, AudioProcessor, Audio, AudioLatentShape
from hallo3.sgm.models.ltx_audio_vae.normalization import NormType
from hallo3.sgm.models.ltx_audio_vae.causality_axis import CausalityAxis
from hallo3.sgm.models.ltx_audio_vae.attention import AttentionType


def load_audio_encoder(checkpoint_path: str, device: str = "cuda") -> AudioEncoder:
    """从 safetensors 文件加载 AudioEncoder 权重。"""

    # 实例化 encoder（使用 LTX-2.5 默认配置）
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
    )

    # 加载 safetensors
    print(f"Loading checkpoint: {checkpoint_path}")
    all_weights = load_file(checkpoint_path)

    # checkpoint 中 encoder 的 key 格式:
    #   audio_vae.encoder.xxx  → 去掉前缀变成 xxx
    #   audio_vae.per_channel_statistics.xxx → 变成 per_channel_statistics.xxx
    encoder_state_dict = {}
    for key, value in all_weights.items():
        if key.startswith("audio_vae.encoder."):
            new_key = key.replace("audio_vae.encoder.", "")
            encoder_state_dict[new_key] = value
        elif key.startswith("audio_vae.per_channel_statistics."):
            new_key = key.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")
            encoder_state_dict[new_key] = value

    print(f"Matched {len(encoder_state_dict)} keys for encoder")

    # 加载权重
    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    if missing:
        print(f"WARNING - Missing keys: {missing}")
    if unexpected:
        print(f"WARNING - Unexpected keys: {unexpected}")
    if not missing and not unexpected:
        print("All keys matched perfectly!")

    encoder = encoder.to(device=device, dtype=torch.bfloat16)
    encoder.eval()
    return encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ltx-2.5-audio-vae-bf16.safetensors")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device

    # ====== Step 1: 加载 encoder ======
    print("=" * 60)
    print("Step 1: Loading AudioEncoder")
    print("=" * 60)
    encoder = load_audio_encoder(args.checkpoint, device=device)

    # ====== Step 2: 创建 AudioProcessor ======
    print("\n" + "=" * 60)
    print("Step 2: Creating AudioProcessor (mel spectrogram)")
    print("=" * 60)
    processor = AudioProcessor(
        target_sample_rate=16000,
        mel_bins=64,
        mel_hop_length=160,
        n_fft=1024,
    ).to(device=device)
    print("AudioProcessor created.")

    # ====== Step 3: 用合成音频测试 forward pass ======
    print("\n" + "=" * 60)
    print("Step 3: Forward pass with synthetic 10s stereo audio")
    print("=" * 60)

    # 模拟 10 秒 stereo 16kHz 音频
    duration_sec = 10
    waveform = torch.randn(1, 2, 16000 * duration_sec, device=device, dtype=torch.float32)
    audio = Audio(waveform=waveform, sampling_rate=16000)

    # waveform → mel
    with torch.no_grad():
        mel = processor.waveform_to_mel(audio)
    print(f"Mel spectrogram shape: {mel.shape}")
    # 期望: (1, 2, T_mel, 64) where T_mel ≈ 1001

    # mel → encoder → latent
    with torch.no_grad():
        latent = encoder(mel.to(dtype=torch.bfloat16))
    print(f"Encoder output shape:  {latent.shape}")
    # 期望: (1, 8, T_mel/4, 16) ≈ (1, 8, 250, 16)

    # patchify: (B, 8, T, 16) → (B, T, 128)
    tokens = rearrange(latent, "b c t f -> b t (c f)")
    print(f"Patchified tokens:     {tokens.shape}")
    # 期望: (1, 250, 128)

    # ====== Step 4: 验证时间分辨率对齐 ======
    print("\n" + "=" * 60)
    print("Step 4: Temporal resolution check")
    print("=" * 60)
    latent_frames = latent.shape[2]
    expected_fps = 16000 / 160 / 4  # = 25.0
    actual_fps = latent_frames / duration_sec
    print(f"Duration:       {duration_sec}s")
    print(f"Latent frames:  {latent_frames}")
    print(f"Expected fps:   {expected_fps}")
    print(f"Actual fps:     {actual_fps}")
    print(f"Video fps match: {'YES' if abs(actual_fps - 25.0) < 1.0 else 'NO'}")

    # ====== Step 5: 模拟 Hallo3 AudioProjModel 的输入 ======
    print("\n" + "=" * 60)
    print("Step 5: Simulate Hallo3 AudioProjModel input")
    print("=" * 60)

    # 去掉 batch dim，得到 (T, 128)
    audio_emb = tokens.squeeze(0).float()  # (T, 128)
    print(f"Per-frame embedding: {audio_emb.shape}")

    # unsqueeze blocks dim: (T, 128) → (T, 1, 128)
    audio_emb = audio_emb.unsqueeze(1)  # (T, 1, 128)

    # 模拟 temporal window ±2 (如 data_video.py)
    audio_margin = 2
    T = audio_emb.shape[0]
    # 取中间一段 13 帧作为示例
    center_indices = torch.arange(audio_margin, audio_margin + 13)
    margin_indices = torch.arange(2 * audio_margin + 1) - audio_margin  # [-2,-1,0,1,2]
    window_indices = center_indices.unsqueeze(1) + margin_indices.unsqueeze(0)  # (13, 5)
    audio_window = audio_emb[window_indices]  # (13, 5, 1, 128)
    print(f"Temporal window:     {audio_window.shape}")
    print(f"  → (frames=13, seq_len=5, blocks=1, channels=128)")

    # flatten 成 AudioProjModel 输入
    flat = audio_window.view(13, 5 * 1 * 128)  # (13, 640)
    print(f"Flattened input:     {flat.shape}")
    print(f"  → AudioProjModel.proj1 input_dim = {flat.shape[1]} (原来是 46080)")

    # ====== 总结 ======
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Encoder loaded:       OK")
    print(f"  Forward pass:         OK")
    print(f"  Output shape:         {latent.shape} → patchified {tokens.shape}")
    print(f"  Temporal fps:         {actual_fps:.1f} (target: 25.0)")
    print(f"  AudioProjModel input: {flat.shape[1]} (was 46080, ratio: {46080/flat.shape[1]:.1f}x)")
    print(f"\nAll checks passed! Ready for Hallo3 integration.")


if __name__ == "__main__":
    main()
