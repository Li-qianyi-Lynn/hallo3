"""Scan training videos for decord problems (corrupt frames / slow or hanging seeks).

Each video is probed in its own subprocess with a hard timeout, because a hanging
decord seek spins in native code and cannot be interrupted from Python.
Reads are timed the way training does them (get_batch window, single frames via a fresh reader).

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

def probe(path, slow):
    import time

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

    # 与训练一致：窗口用 get_batch 读，参考帧用新开的 reader 读单帧
    worst, worst_what = 0.0, ""

    def timed(what, fn):
        nonlocal worst, worst_what
        t = time.time()
        fn()
        dt = time.time() - t
        if dt > worst:
            worst, worst_what = dt, what

    n_win = 49
    last_start = max(total - n_win, 0)
    for s in sorted({0, last_start // 2, last_start}):
        window = list(range(s, min(s + n_win, total)))
        timed(f"window@{s}", lambda w=window: VideoReader(path, num_threads=1).get_batch(w))

    points = sorted({round(k * (total - 1) / 11) for k in range(12)} | {total - 1})
    for i in points:
        timed(f"frame@{i}", lambda i=i: VideoReader(path, num_threads=1)[i])

    if worst > slow:
        print(f"SLOW_SEEK {worst:.1f}s {worst_what}")
    else:
        print("OK")


def run_probe(path, timeout, slow):
    try:
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--probe", path, "--slow", str(slow)],
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
    reasons = []
    if out.startswith(("LENMISMATCH", "SLOW_SEEK")):
        reasons.append(out)
    if "corrupted" in err:
        reasons.append("CORRUPT_FRAMES")
    return path, "; ".join(reasons) or "OK"


def extract_paths(data):
    items = data if isinstance(data, list) else list(data.values())
    return items, [it["video_path"] for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", help="internal: probe a single video")
    ap.add_argument("--json", help="training json (list of dicts with video_path)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=180.0, help="per-video hard limit (s)")
    ap.add_argument("--slow", type=float, default=3.0, help="single seek slower than this (s) is flagged")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.probe:
        probe(args.probe, args.slow)
        return

    with open(args.json) as f:
        data = json.load(f)
    items, paths = extract_paths(data)

    bad = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_probe, p, args.timeout, args.slow) for p in paths]
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
