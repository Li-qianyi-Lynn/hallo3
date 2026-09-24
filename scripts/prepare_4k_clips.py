#!/usr/bin/env python3
"""
从 T7 硬盘的 scene 文件中切出 4000 条 5 秒 clip。

硬盘结构:
  /Volumes/T7/TalkVid/{batch}/{youtube_id}/
    ├── 0000_video{youtube_id}.txt          ← scene 时间信息
    ├── video{youtube_id}_{scene}.mp4       ← 完整 scene 视频
    └── video{youtube_id}_{scene}.m4a       ← 完整 scene 音频

JSON 里的 start-time/end-time 是相对于原始 YouTube 视频的绝对时间。
需要: 读 txt 获取 scene 起始时间 → 算出 scene 文件内的偏移 → ffmpeg 截取 5 秒。

用法:
  python scripts/prepare_4k_clips.py \
      --output /Volumes/T7/TalkVid_4k_clips

  # 然后传到集群
  rsync -avP /Volumes/T7/TalkVid_4k_clips/ li_qiany_neu@aicr:/scratch/li_qiany_neu/talkvid_4k/
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


TALKVID_BASE = Path("/Volumes/T7/TalkVid")

# 缓存: youtube_id → {scene_name: start_seconds}
_scene_times_cache = {}


def parse_scene_times(txt_path: Path) -> dict:
    """解析 0000_video{id}.txt，返回 {scene_name: start_seconds}。"""
    times = {}
    with open(txt_path, "r") as f:
        for line in f:
            # 格式: "scene 5 infos: start_time 3:0, end_time 3:14"
            m = re.match(r"scene\s+(\d+)\s+infos:\s+start_time\s+(\d+):(\d+)", line)
            if m:
                scene_num = int(m.group(1))
                minutes = int(m.group(2))
                seconds = int(m.group(3))
                times[f"scene{scene_num}"] = minutes * 60 + seconds
    return times


def find_youtube_dir(youtube_id: str) -> Path | None:
    """在硬盘各 batch 目录中找到 youtube_id 对应的文件夹。"""
    for batch_dir in TALKVID_BASE.iterdir():
        if not batch_dir.is_dir() or batch_dir.name.endswith("_av1"):
            continue
        yt_dir = batch_dir / youtube_id
        if yt_dir.exists():
            return yt_dir
    return None


def get_scene_start_time(youtube_id: str, scene_name: str) -> float | None:
    """获取 scene 在原始视频中的起始时间（秒）。"""
    if youtube_id not in _scene_times_cache:
        yt_dir = find_youtube_dir(youtube_id)
        if yt_dir is None:
            return None
        # 找 txt 文件
        txt_files = list(yt_dir.glob("0000_*.txt"))
        if not txt_files:
            return None
        _scene_times_cache[youtube_id] = parse_scene_times(txt_files[0])

    return _scene_times_cache.get(youtube_id, {}).get(scene_name)


def parse_clip_id(clip_id: str):
    """
    解析 clip ID → (youtube_id, parent_scene, sub_scene)
    例: videovideo_xoXaK1WdGzc-scene5-scene3
      → youtube_id = _xoXaK1WdGzc, parent_scene = scene5, sub_scene = scene3
    """
    raw = clip_id
    if raw.startswith("videovideo"):
        raw = raw[len("video"):]  # → video_xoXaK1WdGzc-scene5-scene3

    # 从右边拆 scene 信息
    parts = raw.rsplit("-", 2)
    if len(parts) == 3:
        video_part, parent_scene, sub_scene = parts
    elif len(parts) == 2:
        video_part, parent_scene = parts
        sub_scene = None
    else:
        return None, None, None

    youtube_id = video_part[len("video"):] if video_part.startswith("video") else video_part
    return youtube_id, parent_scene, sub_scene


def process_clip(item: dict, output_dir: Path) -> dict:
    """处理一条 clip: 找到源 scene → 算偏移 → ffmpeg 截取。"""
    clip_id = item["id"]
    abs_start = item["start-time"]       # 相对于原始 YouTube 视频的绝对时间
    abs_end = item["end-time"]
    duration = abs_end - abs_start

    youtube_id, parent_scene, sub_scene = parse_clip_id(clip_id)
    if youtube_id is None:
        return {"id": clip_id, "status": "parse_error"}

    # 找 youtube 目录
    yt_dir = find_youtube_dir(youtube_id)
    if yt_dir is None:
        return {"id": clip_id, "status": "yt_dir_not_found", "yt": youtube_id}

    # 找 scene 视频和音频
    scene_video = yt_dir / f"video{youtube_id}_{parent_scene}.mp4"
    if not scene_video.exists():
        return {"id": clip_id, "status": "video_not_found", "path": str(scene_video)}

    # 获取 scene 在原始视频中的起始时间
    scene_start = get_scene_start_time(youtube_id, parent_scene)
    if scene_start is None:
        return {"id": clip_id, "status": "scene_time_not_found", "yt": youtube_id, "scene": parent_scene}

    # 计算在 scene 文件内的偏移
    offset_in_scene = abs_start - scene_start

    # 安全检查
    if offset_in_scene < 0:
        offset_in_scene = 0

    # 输出路径
    safe_id = clip_id.replace("/", "_")
    out_video = output_dir / "videos" / f"{safe_id}.mp4"
    out_audio = output_dir / "audios" / f"{safe_id}.m4a"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_audio.parent.mkdir(parents=True, exist_ok=True)

    if out_video.exists() and out_audio.exists():
        return {"id": clip_id, "status": "exists"}

    try:
        # 切视频
        cmd_video = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(offset_in_scene),
            "-i", str(scene_video),
            "-t", str(duration),
            "-c:v", "copy",
            "-c:a", "copy",
            str(out_video),
        ]
        subprocess.run(cmd_video, check=True, timeout=30)

        # 切音频 (优先用独立 m4a，没有就从视频提取)
        scene_audio = None
        for ext in [".m4a", ".wav", ".mp3"]:
            p = yt_dir / f"video{youtube_id}_{parent_scene}{ext}"
            if p.exists():
                scene_audio = p
                break

        if scene_audio:
            cmd_audio = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(offset_in_scene),
                "-i", str(scene_audio),
                "-t", str(duration),
                "-c:a", "copy",
                str(out_audio),
            ]
        else:
            cmd_audio = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(offset_in_scene),
                "-i", str(scene_video),
                "-t", str(duration),
                "-vn", "-c:a", "aac",
                str(out_audio),
            ]
        subprocess.run(cmd_audio, check=True, timeout=30)

        return {"id": clip_id, "status": "ok"}

    except subprocess.TimeoutExpired:
        return {"id": clip_id, "status": "timeout"}
    except subprocess.CalledProcessError as e:
        return {"id": clip_id, "status": f"ffmpeg_error"}
    except Exception as e:
        return {"id": clip_id, "status": f"error: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="splits/talkvid_en_4k.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    data = json.load(open(args.json))
    print(f"待处理: {len(data)} 条 clips")
    print(f"输出到: {args.output}")

    # 测试前 5 条
    print("\n--- 前 5 条测试 ---")
    for item in data[:5]:
        yt_id, scene, sub = parse_clip_id(item["id"])
        yt_dir = find_youtube_dir(yt_id)
        scene_start = get_scene_start_time(yt_id, scene) if yt_dir else None
        offset = item["start-time"] - scene_start if scene_start is not None else None

        print(f"  {item['id']}")
        print(f"    youtube_id={yt_id}, scene={scene}")
        print(f"    yt_dir: {'FOUND' if yt_dir else 'NOT FOUND'}")
        if scene_start is not None:
            print(f"    scene_start={scene_start:.0f}s, clip_abs_start={item['start-time']:.1f}s, offset_in_scene={offset:.2f}s")
            print(f"    → 从 {scene}.mp4 的 {offset:.2f}s 截取 {item['end-time']-item['start-time']:.1f}s")
        else:
            print(f"    scene_start: NOT FOUND")

    if os.isatty(0):
        input("\n按回车开始处理 (Ctrl+C 取消)... ")
    else:
        print("\n非交互模式，直接开始处理...")

    worker = partial(process_clip, output_dir=args.output)

    # 单进程版本（因为 _scene_times_cache 不能跨进程共享，先用单进程）
    # 如果太慢可以改成预加载 cache 后多进程
    results = []
    for item in tqdm(data, desc="Cutting clips"):
        results.append(worker(item))

    # 统计
    from collections import Counter
    stats = Counter(r["status"] for r in results)
    print(f"\n=== 完成 ===")
    for status, count in stats.most_common():
        print(f"  {status}: {count}")

    failed = [r for r in results if r["status"] not in ("ok", "exists")]
    if failed:
        fail_path = args.output / "failed.json"
        json.dump(failed, open(fail_path, "w"), indent=2)
        print(f"\n失败列表 ({len(failed)} 条): {fail_path}")

    ok_count = stats.get("ok", 0) + stats.get("exists", 0)
    print(f"\n成功: {ok_count}/{len(data)}")
    print(f"\n下一步: rsync {args.output}/ li_qiany_neu@aicr:/scratch/li_qiany_neu/talkvid_4k/")


if __name__ == "__main__":
    main()
