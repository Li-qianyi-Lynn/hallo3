from typing import Set, Tuple

import torch

from .patchifier import AudioPatchifier
from .attention import AttentionType, make_attn
from .causal_conv_2d import make_conv2d
from .causality_axis import CausalityAxis
from .downsample import build_downsampling_path
from .ops import AudioProcessor, PerChannelStatistics
from .resnet import ResnetBlock
from .normalization import NormType, build_normalization_layer
from .types import Audio, AudioLatentShape

LATENT_DOWNSAMPLE_FACTOR = 4


def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> torch.nn.Module:
    """Build the middle block with two ResNet blocks and optional attention."""
    mid = torch.nn.Module()
    mid.block_1 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid.attn_1 = make_attn(channels, attn_type=attn_type, norm_type=norm_type) if add_attention else torch.nn.Identity()
    mid.block_2 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run features through the middle block."""
    features = mid.block_1(features, temb=None)
    features = mid.attn_1(features)
    return mid.block_2(features, temb=None)


class AudioEncoder(torch.nn.Module):
    """
    Encoder that compresses audio spectrograms into latent representations.
    The encoder uses a series of downsampling blocks with residual connections,
    attention mechanisms, and configurable causal convolutions.
    """

    def __init__(
        self,
        *,
        ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        attn_type: AttentionType = AttentionType.VANILLA,
        mid_block_add_attention: bool = True,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        is_causal: bool = True,
        mel_bins: int = 64,
        **_ignore_kwargs,
    ) -> None:
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.double_z = double_z
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        # downsampling
        self.conv_in = make_conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.non_linearity = torch.nn.SiLU()

        self.down, block_in = build_downsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
        )

        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        self.norm_out = build_normalization_layer(block_in, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Encode audio spectrogram into latent representations.
        Args:
            spectrogram: Input spectrogram of shape (batch, channels, time, frequency)
        Returns:
            Encoded latent representation of shape (batch, channels, frames, mel_bins)
        """
        print(f"[AUDIO_DEBUG] AudioEncoder.forward() 输入 spectrogram: shape={spectrogram.shape}, dtype={spectrogram.dtype}")

        h = self.conv_in(spectrogram)
        print(f"[AUDIO_DEBUG]   conv_in 后: shape={h.shape}  (Conv2d {self.in_channels}→{self.ch}, k=3, s=1)")

        h = self._run_downsampling_path(h)
        print(f"[AUDIO_DEBUG]   downsampling_path 后: shape={h.shape}")

        h = run_mid_block(self.mid, h)
        print(f"[AUDIO_DEBUG]   mid_block 后: shape={h.shape}")

        h = self._finalize_output(h)
        print(f"[AUDIO_DEBUG]   finalize_output 后 (norm→SiLU→conv_out): shape={h.shape}  (double_z={self.double_z}, z_ch={self.z_channels})")

        result = self._normalize_latents(h)
        print(f"[AUDIO_DEBUG]   normalize_latents 后 (最终输出): shape={result.shape}")
        return result

    def _run_downsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage.block[block_idx](h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)
                print(f"[AUDIO_DEBUG]     downsample level={level} block={block_idx}: shape={h.shape}")

            if level != self.num_resolutions - 1:
                h = stage.downsample(h)
                print(f"[AUDIO_DEBUG]     downsample level={level} 下采样后: shape={h.shape}")

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm_out(h)
        h = self.non_linearity(h)
        return self.conv_out(h)

    def _normalize_latents(self, latent_output: torch.Tensor) -> torch.Tensor:
        means = torch.chunk(latent_output, 2, dim=1)[0]
        print(f"[AUDIO_DEBUG]     _normalize: chunk 取 means: shape={means.shape}  (从 {latent_output.shape} 沿 dim=1 拆半)")
        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[1],
            frames=means.shape[2],
            mel_bins=means.shape[3],
        )
        print(f"[AUDIO_DEBUG]     _normalize: AudioLatentShape=(b={latent_shape.batch}, c={latent_shape.channels}, t={latent_shape.frames}, f={latent_shape.mel_bins})")
        latent_patched = self.patchifier.patchify(means)
        print(f"[AUDIO_DEBUG]     _normalize: patchify 后: shape={latent_patched.shape}  (b, t, c*f)")
        latent_normalized = self.per_channel_statistics.normalize(latent_patched)
        print(f"[AUDIO_DEBUG]     _normalize: normalize 后: shape={latent_normalized.shape}")
        result = self.patchifier.unpatchify(latent_normalized, latent_shape)
        print(f"[AUDIO_DEBUG]     _normalize: unpatchify 后: shape={result.shape}  (b, c, t, f)")
        return result


def encode_audio(
    audio: Audio,
    audio_encoder: AudioEncoder,
    audio_processor: AudioProcessor | None = None,
) -> torch.Tensor:
    """Encode audio waveform into latent representation.
    Args:
        audio: Audio container with waveform tensor of shape (batch, channels, samples) and sampling rate.
        audio_encoder: Audio encoder model
        audio_processor: Audio processor model (optional, if not provided, it will be created from the audio encoder)
    """
    dtype = next(audio_encoder.parameters()).dtype
    device = next(audio_encoder.parameters()).device

    if audio_processor is None:
        audio_processor = AudioProcessor(
            target_sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        ).to(device=device)

    print(f"[AUDIO_DEBUG] encode_audio(): 输入 audio.waveform shape={audio.waveform.shape}, sr={audio.sampling_rate}")
    mel_spectrogram = audio_processor.waveform_to_mel(audio.to(device=device))
    print(f"[AUDIO_DEBUG] encode_audio(): mel_spectrogram shape={mel_spectrogram.shape}, dtype={mel_spectrogram.dtype}")

    latent = audio_encoder(mel_spectrogram.to(dtype=dtype))
    print(f"[AUDIO_DEBUG] encode_audio(): 最终 latent shape={latent.shape}, dtype={latent.dtype}")
    return latent
