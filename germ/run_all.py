#!/usr/bin/env python3
"""One-command germination pipeline for one experiment folder.

Example:
  python run_all.py --frames-dir ../maizeCam --profile maize --prefix maize
  python run_all.py --frames-dir ../wheatCam --profile wheat --prefix wheat

Runs: calibrate -> process -> analyze -> report -> timelapse, writing
  <prefix>_seed_map.json, <prefix>_features.csv, <prefix>_per_seed.csv,
  <prefix>_per_frame.csv, germination_report_<prefix>.xlsx,
  germination_timelapse_<prefix>.mp4
"""
import argparse, subprocess, sys, re
from pathlib import Path

def seq_index(name):
    m = re.search(r"_(\d+)\.jpg$", name); return int(m.group(1)) if m else -1

def run(cmd):
    print(">>", " ".join(cmd)); subprocess.run([sys.executable] + cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ref-seq", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    p = a.prefix
    frames = sorted(Path(a.frames_dir).glob("*.jpg"), key=lambda x: seq_index(x.name))
    ref = next(x for x in frames if seq_index(x.name) == a.ref_seq)
    smap = f"{p}_seed_map.json"; feat = f"{p}_features.csv"
    pseed = f"{p}_per_seed.csv"; pframe = f"{p}_per_frame.csv"
    run(["calibrate.py", "--frame", str(ref), "--out", smap, "--profile", a.profile,
         "--scale", str(a.scale), "--annot", f"{p}_calib_check.jpg"])
    run(["process.py", "--frames-dir", a.frames_dir, "--map", smap, "--out", feat,
         "--profile", a.profile, "--ref-seq", str(a.ref_seq), "--scale", str(a.scale)])
    run(["analyze.py", "--features", feat, "--map", smap, "--profile", a.profile,
         "--out-seed", pseed, "--out-frame", pframe])
    run(["report.py", "--map", smap, "--per-seed", pseed, "--per-frame", pframe,
         "--out", f"germination_report_{p}.xlsx"])
    run(["make_video.py", "--frames-dir", a.frames_dir, "--map", smap, "--features", feat,
         "--per-seed", pseed, "--out", f"germination_timelapse_{p}.mp4", "--scale", "0.4"])
    print(f"\nDONE. Report: germination_report_{p}.xlsx  Timelapse: germination_timelapse_{p}.mp4")

if __name__ == "__main__":
    main()
