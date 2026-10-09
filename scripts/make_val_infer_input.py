"""从 val/eval 的 json 里随机挑 N 条，生成 sample_video.py 的输入文件。

输入行格式:  caption@@第一帧图片@@音频
每条的对应关系（clip 名、真值视频路径）另存一份 <out>.map.tsv，便于之后对比真值、算 SyncNet。

目录约定（来自 scripts/process_4k_data.py）:
    <data_dir>/videos/<stem>.mp4
    <data_dir>/images/<stem>/000000.jpg      <- 第一帧
    <data_dir>/audios/<stem>.<m4a|wav|...>

用法:
    python scripts/make_val_infer_input.py --json data/talkvid_val.json --n 5 --out my_inference/input_val5.txt
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="固定种子，保证每个 checkpoint 用同一批样本")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json) as f:
        items = json.load(f)
    items = items if isinstance(items, list) else list(items.values())

    rng = random.Random(args.seed)
    order = list(range(len(items)))
    rng.shuffle(order)

    lines, rows, skipped = [], [], 0
    for i in order:
        if len(lines) >= args.n:
            break
        it = items[i]
        video = Path(it["video_path"])
        stem = video.stem
        data_dir = video.parent.parent

        image = data_dir / "images" / stem / "000000.jpg"
        audios = sorted((data_dir / "audios").glob(stem + ".*"))
        if not image.exists() or not audios:
            skipped += 1
            continue

        caption = str(it.get("caption", "A person talking.")).replace("\n", " ").replace("@@", " ")
        lines.append(f"{caption}@@{image}@@{audios[0]}")
        rows.append(f"{stem}\t{video}\t{image}\t{audios[0]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(str(out) + ".map.tsv").write_text(
        "stem\tgt_video\timage\taudio\n" + "\n".join(rows) + "\n", encoding="utf-8")

    print(f"写入 {len(lines)} 条 -> {out}（跳过缺文件 {skipped} 条）")
    print(f"对应关系 -> {out}.map.tsv")


if __name__ == "__main__":
    main()
