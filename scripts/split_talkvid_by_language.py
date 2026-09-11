#!/usr/bin/env python3
"""
Split TalkVid metadata JSON into English and non-English subsets.

Reads filtered_video_clips_with_captions.json and outputs:
  - talkvid_english.json       (English clips only)
  - talkvid_non_english.json   (all other languages)
  - talkvid_split_stats.json   (summary statistics)

Usage:
    python scripts/split_talkvid_by_language.py \
        --input /path/to/filtered_video_clips_with_captions.json \
        --output_dir /path/to/output/

After splitting, use each JSON to drive the TalkVid → Hallo3 pipeline separately:
    # English subset
    python scripts/convert_talkvid_to_hallo3.py \
        --clips_flat /path/to/clips_flat \
        --output /path/to/hallo3_data_en \
        --clip_ids talkvid_english.json

    # Non-English subset
    python scripts/convert_talkvid_to_hallo3.py \
        --clips_flat /path/to/clips_flat \
        --output /path/to/hallo3_data_non_en \
        --clip_ids talkvid_non_english.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_duration(dur_str: str) -> float:
    """Parse duration string like '5.042s' to seconds."""
    return float(dur_str.replace("s", ""))


def extract_youtube_id(clip_id: str) -> str:
    """Extract YouTube video ID from TalkVid clip ID.

    'videovideoTr6MMsoWAog-scene1-scene1' -> 'Tr6MMsoWAog'
    """
    cleaned = clip_id.replace("videovideo", "")
    return cleaned.split("-scene")[0]


def main():
    parser = argparse.ArgumentParser(
        description="Split TalkVid metadata by language"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to filtered_video_clips_with_captions.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--include_unknown_in_nonenglish",
        action="store_true",
        help="Include 'Unknown' language clips in non-English set (default: exclude)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {args.input} ...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Total clips: {len(data)}")

    # Split by language
    english_clips = []
    non_english_clips = []
    excluded_clips = []  # Unknown language clips (if not included)

    lang_stats = defaultdict(lambda: {"clips": 0, "duration_s": 0.0, "videos": set()})

    for clip in data:
        lang = clip.get("info", {}).get("Language", "Unknown")
        dur = parse_duration(clip.get("durations", "0s"))
        yt_id = extract_youtube_id(clip["id"])

        lang_stats[lang]["clips"] += 1
        lang_stats[lang]["duration_s"] += dur
        lang_stats[lang]["videos"].add(yt_id)

        if lang == "English":
            english_clips.append(clip)
        elif lang == "Unknown" and not args.include_unknown_in_nonenglish:
            excluded_clips.append(clip)
        else:
            non_english_clips.append(clip)

    # Save English subset
    en_path = output_dir / "talkvid_english.json"
    with open(en_path, "w") as f:
        json.dump(english_clips, f, ensure_ascii=False)
    print(f"\nEnglish subset: {len(english_clips)} clips -> {en_path}")

    # Save non-English subset
    non_en_path = output_dir / "talkvid_non_english.json"
    with open(non_en_path, "w") as f:
        json.dump(non_english_clips, f, ensure_ascii=False)
    print(f"Non-English subset: {len(non_english_clips)} clips -> {non_en_path}")

    if excluded_clips:
        excluded_path = output_dir / "talkvid_unknown_lang.json"
        with open(excluded_path, "w") as f:
            json.dump(excluded_clips, f, ensure_ascii=False)
        print(f"Excluded (Unknown): {len(excluded_clips)} clips -> {excluded_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Language Distribution")
    print("=" * 70)
    print(f"{'Language':<15} {'Clips':>8} {'Duration':>10} {'Videos':>8} {'Subset':<12}")
    print("-" * 70)

    for lang in sorted(lang_stats.keys(), key=lambda l: -lang_stats[l]["duration_s"]):
        s = lang_stats[lang]
        hours = s["duration_s"] / 3600
        n_videos = len(s["videos"])
        if lang == "English":
            subset = "english"
        elif lang == "Unknown" and not args.include_unknown_in_nonenglish:
            subset = "excluded"
        else:
            subset = "non_english"
        print(f"{lang:<15} {s['clips']:>8} {hours:>9.1f}h {n_videos:>8} {subset:<12}")

    en_hours = sum(
        parse_duration(c.get("durations", "0s")) for c in english_clips
    ) / 3600
    non_en_hours = sum(
        parse_duration(c.get("durations", "0s")) for c in non_english_clips
    ) / 3600

    print("-" * 70)
    print(f"{'English total':<15} {len(english_clips):>8} {en_hours:>9.1f}h")
    print(f"{'Non-English':<15} {len(non_english_clips):>8} {non_en_hours:>9.1f}h")
    if excluded_clips:
        ex_hours = sum(
            parse_duration(c.get("durations", "0s")) for c in excluded_clips
        ) / 3600
        print(f"{'Excluded':<15} {len(excluded_clips):>8} {ex_hours:>9.1f}h")

    # Save stats
    stats = {
        "total_clips": len(data),
        "english_clips": len(english_clips),
        "non_english_clips": len(non_english_clips),
        "excluded_clips": len(excluded_clips),
        "english_hours": round(en_hours, 1),
        "non_english_hours": round(non_en_hours, 1),
        "languages": {
            lang: {
                "clips": s["clips"],
                "hours": round(s["duration_s"] / 3600, 1),
                "unique_videos": len(s["videos"]),
            }
            for lang, s in lang_stats.items()
        },
        "non_english_languages": [
            lang for lang in lang_stats if lang not in ("English", "Unknown")
        ],
    }
    stats_path = output_dir / "talkvid_split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats saved: {stats_path}")

    # Generate YouTube video ID lists (useful for filtering on T7 / cluster)
    en_yt_ids = sorted({extract_youtube_id(c["id"]) for c in english_clips})
    non_en_yt_ids = sorted({extract_youtube_id(c["id"]) for c in non_english_clips})

    en_ids_path = output_dir / "youtube_ids_english.txt"
    with open(en_ids_path, "w") as f:
        f.write("\n".join(en_yt_ids) + "\n")
    print(f"English YouTube IDs: {len(en_yt_ids)} -> {en_ids_path}")

    non_en_ids_path = output_dir / "youtube_ids_non_english.txt"
    with open(non_en_ids_path, "w") as f:
        f.write("\n".join(non_en_yt_ids) + "\n")
    print(f"Non-English YouTube IDs: {len(non_en_yt_ids)} -> {non_en_ids_path}")


if __name__ == "__main__":
    main()
