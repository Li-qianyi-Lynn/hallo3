"""从 8K quality 集中剔除 val+eval，随机抽 N 条，生成训练 JSON。

生成的 JSON 格式与 talkvid_4k_clean.json 兼容，路径指向 talkvid_8k/ 目录。

用法:
    python scripts/make_train_json_from_8k.py \
        --splits-8k splits/talkvid_en_8k_quality.json \
        --splits-val splits/talkvid_val.json \
        --splits-eval splits/talkvid_eval.json \
        --data-dir /scratch/li_qiany_neu/talkvid_8k \
        --n 2700 --seed 42 \
        --output data/talkvid_train_2700.json
"""
import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-8k", required=True)
    parser.add_argument("--splits-val", required=True)
    parser.add_argument("--splits-eval", required=True)
    parser.add_argument("--data-dir", required=True, help="e.g. /scratch/.../talkvid_8k")
    parser.add_argument("--n", type=int, default=2700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load splits
    with open(args.splits_8k) as f:
        all_8k = json.load(f)
    with open(args.splits_val) as f:
        val = json.load(f)
    with open(args.splits_eval) as f:
        ev = json.load(f)

    # Exclude val + eval
    exclude_ids = {item["id"] for item in val} | {item["id"] for item in ev}
    trainable = [item for item in all_8k if item["id"] not in exclude_ids]
    print(f"8K total: {len(all_8k)}, exclude val+eval: {len(exclude_ids)}, trainable: {len(trainable)}")

    # Random sample
    rng = random.Random(args.seed)
    rng.shuffle(trainable)

    # Verify files exist and build JSON entries
    data_dir = args.data_dir
    results = []
    skipped = 0
    for item in trainable:
        if len(results) >= args.n:
            break

        stem = item["id"]
        video_path = os.path.join(data_dir, "videos", f"{stem}.mp4")
        face_emb_path = os.path.join(data_dir, "face_emb", f"{stem}.pt")
        face_mask_path = os.path.join(data_dir, "face_mask", f"{stem}.png")

        # Check video exists (audio can be .wav, .m4a, etc.)
        if not os.path.exists(video_path):
            skipped += 1
            continue

        # audio_emb will be re-extracted by re_extract_audio_emb.py
        audio_emb_path = os.path.join(data_dir, "audio_emb_omnilingual", f"{stem}.pt")

        caption = item.get("description", "A person talking.")
        if not caption or caption.strip() == "":
            caption = "A person talking."

        results.append({
            "video_path": video_path,
            "face_mask_union_path": face_mask_path,
            "face_emb_path": face_emb_path,
            "vocals_emb_base_all": audio_emb_path,
            "caption": caption,
        })

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} entries to {args.output} (skipped {skipped} missing files)")
    print(f"Audio embeddings path: {data_dir}/audio_emb_omnilingual/ (need to run re_extract_audio_emb.py)")


if __name__ == "__main__":
    main()
