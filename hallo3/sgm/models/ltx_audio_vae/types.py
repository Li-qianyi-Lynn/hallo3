from dataclasses import dataclass, replace
from typing import NamedTuple

import torch


class AudioLatentShape(NamedTuple):
    """
    Shape of audio in VAE latent space: (batch, channels, frames, mel_bins).
    mel_bins is the number of frequency bins from the mel-spectrogram encoding.
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    def token_count(self) -> int:
        """Number of tokens after patchification."""
        return self.frames

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            mel_bins=shape[3],
        )

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)

        return AudioLatentShape(
            batch=batch,
            channels=channels,
            frames=round(duration * latents_per_second),
            mel_bins=mel_bins,
        )


@dataclass(frozen=True)
class Audio:
    """
    Container for decoded audio samples and metadata.
    Attributes:
        waveform: Audio waveform tensor.
        sampling_rate: Sampling rate (Hz) of the waveform.
    """

    waveform: torch.Tensor
    sampling_rate: int

    def to(self, **kwargs: object) -> "Audio":
        return replace(self, waveform=self.waveform.to(**kwargs))
