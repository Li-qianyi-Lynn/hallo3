"""Remove videos that timed out during training from a training json.

Training logs contain lines like
    [Stage2_SFTDataset] skip index=936 path=/.../x.mp4: TimeoutError: sample read exceeded 30s (decord hang)
This collects those paths from one or more logs and writes a copy of the json without them.

Usage:
    python scripts/exclude_videos_from_log.py --json data/talkvid_4k.json \
        --logs /scratch/li_qiany_neu/hallo3_logs/train_4k_*.log \
        --out data/talkvid_4k_clean.json
"""
import argparse
import glob
import json
import re

PATTERN = re.compile(r"skip index=\d+ path=(.+?\.mp4): TimeoutError")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--logs", nargs="+", required=True, help="log files or globs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bad = set()
    for pattern in args.logs:
        for path in glob.glob(pattern):
            with open(path, errors="ignore") as f:
                for line in f:
                    m = PATTERN.search(line)
                    if m:
                        bad.add(m.group(1))

    with open(args.json) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else list(data.values())
    kept = [it for it in items if it["video_path"] not in bad]
    out = kept if isinstance(data, list) else {str(i): it for i, it in enumerate(kept)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    matched = len(items) - len(kept)
    print(f"timed-out videos found in logs: {len(bad)}")
    print(f"removed from json: {matched}  kept: {len(kept)} / {len(items)}")
    unmatched = bad - {it["video_path"] for it in items}
    if unmatched:
        print(f"warning: {len(unmatched)} paths not present in json, e.g. {sorted(unmatched)[:2]}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
