#!/usr/bin/env python3
"""
一站式处理 4k clips: 视频 → face_emb + face_mask + images + caption + audio_emb(VAE) + json

输入:
  /scratch/li_qiany_neu/talkvid_4k/
  ├── videos/    ← 2719 个 .mp4
  └── audios/    ← 2719 个 .m4a

输出:
  /scratch/li_qiany_neu/talkvid_4k/
  ├── videos/
  ├── audios/
  ├── images/         ← 每个视频的抽帧
  ├── face_emb/       ← (512,) insightface embedding
  ├── face_mask/      ← 人脸区域 mask png
  ├── audio_emb/      ← (T, 128) LTX-2 VAE embedding
  ├── caption/        ← "A person talking."
  └── ~/hallo3/data/talkvid_4k.json  ← 训练索引

用法:
  # 在 GPU 节点上跑
  python scripts/process_4k_data.py \
      --data_dir /scratch/li_qiany_neu/talkvid_4k \
      --face_model_path /scratch/li_qiany_neu/pretrained_models/face_analysis \
      --vae_checkpoint /scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def step1_extract_frames(data_dir: Path):
    """从每个视频提取帧到 images/ 目录。"""
    videos_dir = data_dir / "videos"
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"\n[Step 1] 提取视频帧: {len(video_files)} 个视频")

    ok = skip = fail = 0
    for vf in tqdm(video_files, desc="Extracting frames"):
        out_dir = images_dir / vf.stem
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
            skip += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(vf))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)
            idx += 1
        cap.release()

        if idx > 0:
            ok += 1
        else:
            fail += 1

    print(f"  完成: ok={ok}, skip={skip}, fail={fail}")


def step2_extract_face(data_dir: Path, face_model_path: str):
    """用 insightface 提取 face_emb 和 face_mask。"""
    import insightface
    from insightface.app import FaceAnalysis

    videos_dir = data_dir / "videos"
    face_emb_dir = data_dir / "face_emb"
    face_mask_dir = data_dir / "face_mask"
    face_emb_dir.mkdir(exist_ok=True)
    face_mask_dir.mkdir(exist_ok=True)

    # 初始化人脸模型
    app = FaceAnalysis(
        name="buffalo_l",
        root=face_model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"\n[Step 2] 提取人脸 embedding + mask: {len(video_files)} 个视频")

    ok = skip = fail = 0
    for vf in tqdm(video_files, desc="Face detection"):
        emb_path = face_emb_dir / f"{vf.stem}.pt"
        mask_path = face_mask_dir / f"{vf.stem}.png"

        if emb_path.exists() and mask_path.exists():
            skip += 1
            continue

        cap = cv2.VideoCapture(str(vf))
        embeddings = []
        all_bboxes = []

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 每 5 帧检测一次（加速）
            if frame_count % 5 == 0:
                faces = app.get(frame)
                if faces:
                    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    embeddings.append(torch.tensor(best.embedding))
                    all_bboxes.append(best.bbox)
            frame_count += 1
        cap.release()

        if not embeddings:
            fail += 1
            continue

        # face_emb: 所有帧的平均 embedding
        face_emb = torch.stack(embeddings).mean(dim=0)
        torch.save(face_emb, emb_path)

        # face_mask: 用最大 bbox 生成 union mask
        cap2 = cv2.VideoCapture(str(vf))
        ret, first_frame = cap2.read()
        cap2.release()
        if ret:
            h, w = first_frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for bbox in all_bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                mask[y1:y2, x1:x2] = 255
            cv2.imwrite(str(mask_path), mask)

        ok += 1

    print(f"  完成: ok={ok}, skip={skip}, fail={fail}")


def step3_extract_audio_emb(data_dir: Path, vae_checkpoint: str, device: str):
    """用 LTX-2 Audio VAE 提取 audio embedding。"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "hallo3"))
    from sgm.models.ltx_audio_vae import AudioEncoder, AudioProcessor, Audio
    from sgm.models.ltx_audio_vae.normalization import NormType
    from sgm.models.ltx_audio_vae.causality_axis import CausalityAxis
    import librosa
    from safetensors.torch import load_file
    from einops import rearrange

    audios_dir = data_dir / "audios"
    audio_emb_dir = data_dir / "audio_emb"
    audio_emb_dir.mkdir(exist_ok=True)

    # 加载 encoder
    print(f"\n[Step 3] 加载 Audio VAE encoder...")
    encoder = AudioEncoder(
        ch=128, ch_mult=(1, 2, 4), num_res_blocks=2,
        attn_resolutions={8, 16, 32}, resolution=256, z_channels=8,
        double_z=True, dropout=0.0, resamp_with_conv=True, in_channels=2,
        norm_type=NormType.PIXEL, causality_axis=CausalityAxis.HEIGHT,
        sample_rate=16000, mel_hop_length=160, n_fft=1024,
        is_causal=True, mel_bins=64, mid_block_add_attention=False,
    )
    all_weights = load_file(vae_checkpoint)
    encoder_sd = {}
    for k, v in all_weights.items():
        if k.startswith("audio_vae.encoder."):
            encoder_sd[k.replace("audio_vae.encoder.", "")] = v
        elif k.startswith("audio_vae.per_channel_statistics."):
            encoder_sd[k.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")] = v
    encoder.load_state_dict(encoder_sd, strict=True)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()

    processor = AudioProcessor(
        target_sample_rate=16000, mel_bins=64, mel_hop_length=160, n_fft=1024,
    ).to(device=device)

    audio_files = sorted(audios_dir.glob("*"))
    # 用视频文件名来匹配（音频和视频同名）
    video_stems = {p.stem for p in (data_dir / "videos").glob("*.mp4")}
    audio_files = [f for f in audio_files if f.stem in video_stems]

    print(f"  提取 audio embedding: {len(audio_files)} 个音频")

    ok = skip = fail = 0
    for af in tqdm(audio_files, desc="Audio VAE"):
        out_path = audio_emb_dir / f"{af.stem}.pt"
        if out_path.exists():
            skip += 1
            continue

        try:
            speech, sr = librosa.load(str(af), sr=16000, mono=True)
            waveform = torch.from_numpy(speech).float().unsqueeze(0).unsqueeze(0)
            waveform = waveform.repeat(1, 2, 1).to(device=device)
            audio = Audio(waveform=waveform, sampling_rate=16000)

            with torch.no_grad():
                mel = processor.waveform_to_mel(audio)
                latent = encoder(mel.to(dtype=torch.bfloat16))
                tokens = rearrange(latent, "b c t f -> b t (c f)")

            audio_emb = tokens.squeeze(0).float().cpu()
            assert audio_emb.ndim == 2 and audio_emb.shape[1] == 128
            torch.save(audio_emb, out_path)
            ok += 1
        except Exception as e:
            print(f"  Failed {af.stem}: {e}")
            fail += 1

    print(f"  完成: ok={ok}, skip={skip}, fail={fail}")


def step4_generate_caption(data_dir: Path):
    """为每个视频生成简单的 caption。"""
    videos_dir = data_dir / "videos"
    caption_dir = data_dir / "caption"
    caption_dir.mkdir(exist_ok=True)

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"\n[Step 4] 生成 caption: {len(video_files)} 个")

    for vf in video_files:
        cap_path = caption_dir / f"{vf.stem}.txt"
        if not cap_path.exists():
            cap_path.write_text("A person talking.")

    print(f"  完成")


def step5_generate_json(data_dir: Path, json_name: str):
    """生成训练用的 json 索引文件。"""
    images_dir = data_dir / "images"
    output_dir = Path("./data")
    output_dir.mkdir(exist_ok=True)

    print(f"\n[Step 5] 生成训练 json...")

    meta_infos = []
    for frames_dir in sorted(images_dir.iterdir()):
        if not frames_dir.is_dir():
            continue
        stem = frames_dir.name

        video_path = str(data_dir / "videos" / f"{stem}.mp4")
        mask_path = str(data_dir / "face_mask" / f"{stem}.png")
        face_emb_path = str(data_dir / "face_emb" / f"{stem}.pt")
        audio_emb_path = str(data_dir / "audio_emb" / f"{stem}.pt")
        caption_path = data_dir / "caption" / f"{stem}.txt"

        # 检查所有文件存在
        missing = False
        for p in [video_path, mask_path, face_emb_path, audio_emb_path]:
            if not Path(p).exists():
                missing = True
                break
        if missing:
            continue

        caption = caption_path.read_text().strip() if caption_path.exists() else "A person talking."

        meta_infos.append({
            "video_path": video_path,
            "face_mask_union_path": mask_path,
            "face_emb_path": face_emb_path,
            "vocals_emb_base_all": audio_emb_path,
            "caption": caption,
        })

    output_file = output_dir / f"{json_name}.json"
    with open(output_file, "w") as f:
        json.dump(meta_infos, f, indent=2)

    print(f"  生成 {len(meta_infos)} 条 → {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--face_model_path", type=str,
                        default="/scratch/li_qiany_neu/pretrained_models/face_analysis")
    parser.add_argument("--vae_checkpoint", type=str,
                        default="/scratch/li_qiany_neu/pretrained_models/ltx2.5/vae/ltx-2.5-audio-vae-bf16.safetensors")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json_name", type=str, default="talkvid_4k")
    parser.add_argument("--skip_steps", type=str, default="",
                        help="跳过的步骤，逗号分隔，如 '1,2' 跳过帧提取和人脸")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    skip = set(args.skip_steps.split(",")) if args.skip_steps else set()

    print(f"数据目录: {args.data_dir}")
    print(f"设备: {args.device}")
    print(f"视频数: {len(list((args.data_dir / 'videos').glob('*.mp4')))}")
    print(f"音频数: {len(list((args.data_dir / 'audios').glob('*')))}")

    if "1" not in skip:
        step1_extract_frames(args.data_dir)
    if "2" not in skip:
        step2_extract_face(args.data_dir, args.face_model_path)
    if "3" not in skip:
        step3_extract_audio_emb(args.data_dir, args.vae_checkpoint, args.device)
    if "4" not in skip:
        step4_generate_caption(args.data_dir)
    if "5" not in skip:
        step5_generate_json(args.data_dir, args.json_name)

    print("\n===== 全部完成! =====")
    print(f"下一步: 开始训练")
    print(f"  torchrun --standalone --nproc_per_node=8 hallo3/train_video.py \\")
    print(f"      --base configs/cogvideox_5b_i2v_s2.yaml configs/sft_s2_en_multigpu.yaml")


if __name__ == "__main__":
    main()
