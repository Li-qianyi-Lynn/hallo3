from typing import Optional, Tuple

import einops
import torch

from .types import AudioLatentShape


class AudioPatchifier:
    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift
        self._patch_size = (1, patch_size, patch_size)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def get_token_count(self, tgt_shape: AudioLatentShape) -> int:
        return tgt_shape.frames

    def patchify(
        self,
        audio_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flattens the audio latent tensor along time.
        Args:
            audio_latents: Latent tensor to patchify, shape (b, c, t, f).
        Returns:
            Flattened patch tokens tensor, shape (b, t, c*f).
        """
        audio_latents = einops.rearrange(
            audio_latents,
            "b c t f -> b t (c f)",
        )

        return audio_latents

    def unpatchify(
        self,
        audio_latents: torch.Tensor,
        output_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """
        Restores the (B, C, T, F) spectrogram tensor from flattened patches.
        Args:
            audio_latents: Latent tensor to unpatchify, shape (b, t, c*f).
            output_shape: Shape of the unpatched output tensor.
        Returns:
            Unpatched latent tensor, shape (b, c, t, f).
        """
        audio_latents = einops.rearrange(
            audio_latents,
            "b t (c f) -> b c t f",
            c=output_shape.channels,
            f=output_shape.mel_bins,
        )

        return audio_latents
