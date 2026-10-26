"""Convert Omnilingual wav2vec2 300M (fairseq2 format) to HuggingFace Transformers format.

Usage (on cluster login node):
    conda activate hallo3
    python scripts/convert_omnilingual_to_hf.py \
        --input /scratch/li_qiany_neu/pretrained_models/omniASR-W2V-300M.pt \
        --output /scratch/li_qiany_neu/pretrained_models/omnilingual-w2v2-300m

After conversion, load with:
    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained("/scratch/.../omnilingual-w2v2-300m")
"""
import argparse
import re
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor


# fairseq2 key -> HuggingFace Wav2Vec2Model key
# Derived from actual checkpoint keys and HF's SeamlessM4T converter
FAIRSEQ2_TO_HF = [
    # Feature extractor conv layers (7 layers, layer_norm for all)
    (r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.conv\.",
     r"wav2vec2.feature_extractor.conv_layers.\1.conv."),
    (r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.layer_norm\.",
     r"wav2vec2.feature_extractor.conv_layers.\1.layer_norm."),
    # Feature projection
    (r"^encoder_frontend\.post_extract_layer_norm\.",
     r"wav2vec2.feature_projection.layer_norm."),
    (r"^encoder_frontend\.model_dim_proj\.",
     r"wav2vec2.feature_projection.projection."),
    # Positional conv embedding (weight-normalized)
    (r"^encoder_frontend\.pos_encoder\.conv\.",
     r"wav2vec2.encoder.pos_conv_embed.conv."),
    # Encoder layer norm
    (r"^encoder\.layer_norm\.",
     r"wav2vec2.encoder.layer_norm."),
    # Self-attention
    (r"^encoder\.layers\.(\d+)\.self_attn\.k_proj\.",
     r"wav2vec2.encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.v_proj\.",
     r"wav2vec2.encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.q_proj\.",
     r"wav2vec2.encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.output_proj\.",
     r"wav2vec2.encoder.layers.\1.attention.out_proj."),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.",
     r"wav2vec2.encoder.layers.\1.layer_norm."),
    # FFN
    (r"^encoder\.layers\.(\d+)\.ffn\.inner_proj\.",
     r"wav2vec2.encoder.layers.\1.feed_forward.intermediate_dense."),
    (r"^encoder\.layers\.(\d+)\.ffn\.output_proj\.",
     r"wav2vec2.encoder.layers.\1.feed_forward.output_dense."),
    (r"^encoder\.layers\.(\d+)\.ffn_layer_norm\.",
     r"wav2vec2.encoder.layers.\1.final_layer_norm."),
    # Masking
    (r"^masker\.temporal_mask_embed$",
     r"wav2vec2.masked_spec_embed"),
]

# These keys are for pre-training (quantizer, projections) -- not needed for
# feature extraction. We skip them.
SKIP_PREFIXES = ["quantizer.", "final_proj.", "final_target_proj."]


def convert_key(fairseq2_key):
    """Map a single fairseq2 key to HuggingFace format. Returns None if unmapped."""
    for prefix in SKIP_PREFIXES:
        if fairseq2_key.startswith(prefix) or fairseq2_key == prefix.rstrip("."):
            return None

    for pattern, replacement in FAIRSEQ2_TO_HF:
        new_key = re.sub(pattern, replacement, fairseq2_key)
        if new_key != fairseq2_key:
            return new_key

    return "__UNMAPPED__"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to omniASR-W2V-300M.pt")
    parser.add_argument("--output", required=True, help="Output directory for HF model")
    args = parser.parse_args()

    print(f"Loading fairseq2 checkpoint: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    print(f"  Total keys in checkpoint: {len(sd)}")

    # Convert keys
    hf_sd = {}
    skipped = []
    unmapped = []
    for key, value in sd.items():
        new_key = convert_key(key)
        if new_key is None:
            skipped.append(key)
        elif new_key == "__UNMAPPED__":
            unmapped.append(key)
        else:
            hf_sd[new_key] = value

    print(f"  Mapped:   {len(hf_sd)}")
    print(f"  Skipped (pre-training only): {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    {s}")
    if unmapped:
        print(f"  UNMAPPED: {len(unmapped)}")
        for u in unmapped:
            print(f"    {u}  {sd[u].shape}")

    # Create HF config matching Omnilingual 300M architecture
    # Same as facebook/mms-300m: 24 layers, 1024 hidden, 16 heads
    config = Wav2Vec2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        # 7 CNN layers for feature extraction
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        # Layer norm type
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        # Positional encoding
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
    )

    print(f"\nCreating Wav2Vec2Model with config:")
    print(f"  layers={config.num_hidden_layers}, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}")

    model = Wav2Vec2Model(config)

    # Load state dict
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)

    if missing:
        print(f"\nMissing keys ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    if unexpected:
        print(f"\nUnexpected keys ({len(unexpected)}):")
        for u in unexpected:
            print(f"  {u}")

    # Sanity check: most keys should load
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(hf_sd.keys()) & model_keys
    print(f"\nLoaded {len(loaded_keys)}/{len(model_keys)} model keys")

    if len(loaded_keys) < len(model_keys) * 0.9:
        print("WARNING: Less than 90% of keys loaded. Check the mapping!")
        return

    # Save
    model.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")

    # Also save a feature extractor config (needed for AudioProcessor)
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    fe.save_pretrained(args.output)
    print(f"FeatureExtractor config saved to {args.output}")

    # Verify
    print(f"\nVerification: loading back...")
    loaded = Wav2Vec2Model.from_pretrained(args.output)
    print(f"  Success! Model has {sum(p.numel() for p in loaded.parameters())/1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
