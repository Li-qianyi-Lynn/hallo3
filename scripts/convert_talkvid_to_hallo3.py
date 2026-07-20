#!/usr/bin/env python3
"""
Convert TalkVid clips_flat data to Hallo3 training format.

TalkVid structure (clips_flat/):
    videos-crop/*.mp4
    new_face_info/*.pt       -- list[frames] of list[faces], each face has bbox/landmarks/embedding
    short_clip_aud_embeds/*.pt -- dict{'global_embeds': Tensor[T, 1, 768]}

Hallo3 structure (output_dir/):
    videos/*.mp4
    images/*/                -- extracted frames (*.jpg)
    face_emb/*.pt            -- Tensor[512]
    face_mask/*.png          -- binary mask image
    audio_emb/*.pt           -- Tensor[T, 1, 768]
    caption/*.txt

Usage:
    python scripts/convert_talkvid_to_hallo3.py \
        --clips_flat /path/to/TalkVid/clips_flat \
        --output /path/to/hallo3_data \
        --dataset_name talkvid

After conversion, run:
    python hallo3/extract_meta_info.py -r /path/to/hallo3_data -n talkvid
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(output_dir / f"{frame_idx:06d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    return frame_idx


def generate_face_mask(face_info: list, video_path: Path, output_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for frame_faces in face_info:
        if frame_faces:
            bbox = frame_faces[0]['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            mask[y1:y2, x1:x2] = 255
            break

    cv2.imwrite(str(output_path), mask)
    return True


def convert_face_emb(face_info: list):
    embeddings = []
    for frame_faces in face_info:
        if frame_faces:
            emb = frame_faces[0].get('embedding')
            if emb is not None:
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                embeddings.append(emb)
    if not embeddings:
        return None
    return torch.stack(embeddings).mean(dim=0)


def convert_audio_emb(audio_data):
    if isinstance(audio_data, dict):
        return audio_data['global_embeds']
    return audio_data


def process_video(stem: str, clips_flat_dir: Path, output_dir: Path) -> bool:
    video_path      = clips_flat_dir / "videos-crop"          / f"{stem}.mp4"
    face_info_path  = clips_flat_dir / "new_face_info"         / f"{stem}.pt"
    audio_emb_path  = clips_flat_dir / "short_clip_aud_embeds" / f"{stem}.pt"

    for p in [video_path, face_info_path, audio_emb_path]:
        if not p.exists():
            print(f"  [skip] missing: {p.name}")
            return False

    # 1. Copy video
    dst_video = output_dir / "videos" / f"{stem}.mp4"
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, dst_video)

    # 2. Extract frames
    images_dir = output_dir / "images" / stem
    n_frames = extract_frames(video_path, images_dir)
    if n_frames == 0:
        print(f"  [skip] no frames extracted: {stem}")
        return False

    # 3. Face embedding
    face_info = torch.load(face_info_path, weights_only=False, map_location='cpu')
    face_emb = convert_face_emb(face_info)
    if face_emb is None:
        print(f"  [skip] no face embedding: {stem}")
        return False
    dst_face_emb = output_dir / "face_emb" / f"{stem}.pt"
    dst_face_emb.parent.mkdir(parents=True, exist_ok=True)
    torch.save(face_emb, dst_face_emb)

    # 4. Face mask
    dst_face_mask = output_dir / "face_mask" / f"{stem}.png"
    dst_face_mask.parent.mkdir(parents=True, exist_ok=True)
    if not generate_face_mask(face_info, video_path, dst_face_mask):
        print(f"  [skip] mask generation failed: {stem}")
        return False

    # 5. Audio embedding
    audio_data = torch.load(audio_emb_path, weights_only=False, map_location='cpu')
    audio_emb = convert_audio_emb(audio_data)
    dst_audio_emb = output_dir / "audio_emb" / f"{stem}.pt"
    dst_audio_emb.parent.mkdir(parents=True, exist_ok=True)
    torch.save(audio_emb, dst_audio_emb)

    # 6. Caption
    dst_caption = output_dir / "caption" / f"{stem}.txt"
    dst_caption.parent.mkdir(parents=True, exist_ok=True)
    dst_caption.write_text("A person talking.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert TalkVid data to Hallo3 format")
    parser.add_argument("--clips_flat",    type=Path, required=True, help="Path to TalkVid clips_flat/")
    parser.add_argument("--output",        type=Path, required=True, help="Output directory for Hallo3 data")
    parser.add_argument("--dataset_name",  type=str, default="talkvid", help="Dataset name for meta JSON")
    args = parser.parse_args()

    clips_flat_dir = args.clips_flat
    output_dir     = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stems = sorted(
        p.stem for p in (clips_flat_dir / "videos-crop").iterdir()
        if p.suffix == '.mp4'
    )
    print(f"Found {len(video_stems)} videos in videos-crop/")

    success = 0
    for stem in tqdm(video_stems, desc="Converting"):
        if process_video(stem, clips_flat_dir, output_dir):
            success += 1

    print(f"\nDone: {success}/{len(video_stems)} videos converted")
    print(f"Output: {output_dir}")
    print(f"\nNext step:")
    print(f"  python hallo3/extract_meta_info.py -r {output_dir} -n {args.dataset_name}")


if __name__ == "__main__":
    main()
