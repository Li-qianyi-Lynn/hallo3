"""
LTX-2 VAE Audio Processor for inference.
Replaces the wav2vec-based AudioProcessor for the new encoder pipeline.
"""
import math
import os

import librosa
import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file

from sgm.models.ltx_audio_vae import AudioEncoder, Audio
from sgm.models.ltx_audio_vae.ops import AudioProcessor as MelProcessor
from sgm.models.ltx_audio_vae.normalization import NormType
from sgm.models.ltx_audio_vae.causality_axis import CausalityAxis


class LTXAudioProcessor:
    """
    Audio processor using LTX-2 VAE encoder for inference.
    Drop-in replacement for the wav2vec AudioProcessor — same preprocess() interface.
    """

    def __init__(self, vae_checkpoint: str, device="cuda:0"):
        self.device = device
        self.sample_rate = 16000

        # Build encoder
        self.encoder = AudioEncoder(
            ch=128, ch_mult=(1, 2, 4), num_res_blocks=2,
            attn_resolutions={8, 16, 32}, resolution=256, z_channels=8,
            double_z=True, dropout=0.0, resamp_with_conv=True, in_channels=2,
            norm_type=NormType.PIXEL, causality_axis=CausalityAxis.HEIGHT,
            sample_rate=16000, mel_hop_length=160, n_fft=1024,
            is_causal=True, mel_bins=64, mid_block_add_attention=False,
        )

        # Load weights
        all_weights = load_file(vae_checkpoint)
        encoder_sd = {}
        for k, v in all_weights.items():
            if k.startswith("audio_vae.encoder."):
                encoder_sd[k.replace("audio_vae.encoder.", "")] = v
            elif k.startswith("audio_vae.per_channel_statistics."):
                encoder_sd[k.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")] = v
        self.encoder.load_state_dict(encoder_sd, strict=True)
        self.encoder = self.encoder.to(device=device, dtype=torch.bfloat16).eval()

        # Mel processor
        self.mel_processor = MelProcessor(
            target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
        ).to(device=device)

    def preprocess(self, wav_file: str, clip_length: int = -1, fps: float = 25.0):
        """
        Preprocess audio file → LTX-2 VAE latent embedding.
        Returns (audio_emb, audio_length) with same interface as wav2vec AudioProcessor.

        audio_emb: (T, 128) tensor, T matched to fps-based frame count
        audio_length: number of audio frames before padding
        """
        # Load audio
        speech, sr = librosa.load(wav_file, sr=self.sample_rate, mono=True)
        audio_duration = len(speech) / self.sample_rate
        audio_length = math.ceil(audio_duration * fps)

        # Encode with LTX-2 VAE
        waveform = torch.from_numpy(speech).float().unsqueeze(0).unsqueeze(0)
        waveform = waveform.repeat(1, 2, 1).to(device=self.device)
        audio = Audio(waveform=waveform, sampling_rate=self.sample_rate)

        with torch.no_grad():
            mel = self.mel_processor.waveform_to_mel(audio)
            latent = self.encoder(mel.to(dtype=torch.bfloat16))
            tokens = rearrange(latent, "b c t f -> b t (c f)")

        audio_emb = tokens.squeeze(0).float().cpu()  # (T_vae, 128)

        # Interpolate to match fps-based frame count
        # audio_emb: (T_vae, 128) → (audio_length, 128)
        if audio_emb.shape[0] != audio_length:
            audio_emb = audio_emb.unsqueeze(0).permute(0, 2, 1)  # (1, 128, T_vae)
            audio_emb = F.interpolate(audio_emb, size=audio_length, mode="linear", align_corners=False)
            audio_emb = audio_emb.permute(0, 2, 1).squeeze(0)  # (audio_length, 128)

        # Pad to multiple of clip_length (same logic as old processor)
        seq_len = audio_length
        if clip_length > 0 and seq_len % clip_length != 0:
            pad_amount = clip_length - seq_len % clip_length
            padding = audio_emb[-1:].repeat(pad_amount, 1)
            audio_emb = torch.cat([audio_emb, padding], dim=0)

        return audio_emb, audio_length

    def close(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
