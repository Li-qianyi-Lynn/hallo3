"""Scan training videos for decord problems (corrupt frames / hanging seeks).

Each video is probed in its own subprocess with a hard timeout, because a hanging
decord seek spins in native code and cannot be interrupted from Python.

Usage:
    python scripts/scan_bad_videos.py --json data/talkvid_4k.json --workers 6
Outputs (next to --json unless --out-dir is given):
    <stem>_bad_videos.json    {video_path: reason}
    <stem>_clean.json         input json with bad videos removed
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def probe(path):
    from decord import VideoReader

    vr = VideoReader(path, num_threads=1)
    total = len(vr)
    decoded = 0
    try:
        while True:
            vr.next()
            decoded += 1
    except StopIteration:
        pass
    if decoded != total:
        print(f"LENMISMATCH len={total} decoded={decoded}")
        return

    vr = VideoReader(path, num_threads=1)
    fixed = [0, total // 4, total // 2, (3 * total) // 4, total - 31, total - 12, total - 2, total - 1]
    for i in sorted({min(max(i, 0), total - 1) for i in fixed}):
        vr[i]
    print("OK")


def run_probe(path, timeout):
    try:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--probe", path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return path, "HANG"
    out = r.stdout.strip()
    err = r.stderr
    if r.returncode != 0:
        return path, f"ERROR: {err.strip().splitlines()[-1] if err.strip() else r.returncode}"
    if out.startswith("LENMISMATCH"):
        return path, out
    if "corrupted" in err:
        return path, "CORRUPT_FRAMES"
    return path, "OK"


def extract_paths(data):
    items = data if isinstance(data, list) else list(data.values())
    return items, [it["video_path"] for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", help="internal: probe a single video")
    ap.add_argument("--json", help="training json (list of dicts with video_path)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.probe:
        probe(args.probe)
        return

    with open(args.json) as f:
        data = json.load(f)
    items, paths = extract_paths(data)

    bad = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_probe, p, args.timeout) for p in paths]
        for fut in as_completed(futs):
            path, status = fut.result()
            done += 1
            if status != "OK":
                bad[path] = status
                print(f"[{done}/{len(paths)}] BAD {status}: {path}", flush=True)
            elif done % 100 == 0:
                print(f"[{done}/{len(paths)}] scanned, bad so far: {len(bad)}", flush=True)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.json))
    stem = os.path.splitext(os.path.basename(args.json))[0]
    bad_path = os.path.join(out_dir, f"{stem}_bad_videos.json")
    clean_path = os.path.join(out_dir, f"{stem}_clean.json")
    with open(bad_path, "w") as f:
        json.dump(bad, f, indent=2, ensure_ascii=False)

    clean_items = [it for it in items if it["video_path"] not in bad]
    clean = clean_items if isinstance(data, list) else {str(i): it for i, it in enumerate(clean_items)}
    with open(clean_path, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print(f"total={len(paths)} bad={len(bad)} clean={len(clean_items)}")
    print(f"bad list : {bad_path}")
    print(f"clean json: {clean_path}")


if __name__ == "__main__":
    main()
