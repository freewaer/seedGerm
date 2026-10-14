#!/usr/bin/env python3
"""Stage 4b - annotated germination timelapse for QC (green=germinated, red=not)."""
import argparse, json, re, cv2, numpy as np, pandas as pd
from pathlib import Path
from calibrate import load

def seq_index(name):
    m = re.search(r"_(\d+)\.jpg$", name); return int(m.group(1)) if m else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--map", default="seed_map.json")
    ap.add_argument("--features", default="features4.csv")
    ap.add_argument("--per-seed", default="per_seed4.csv")
    ap.add_argument("--out", default="germination_timelapse.mp4")
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    cfg = json.loads(Path(args.map).read_text())
    feats = pd.read_csv(args.features)
    kx = feats.set_index(["seq", "seed_id"])["kx"].to_dict()
    ky = feats.set_index(["seq", "seed_id"])["ky"].to_dict()
    ps = pd.read_csv(args.per_seed).set_index("seed_id")
    germ_seq = {}
    for sid, r in ps.iterrows():
        v = r["germ_seq"]
        germ_seq[sid] = int(v) if str(v) not in ("", "nan") else None
    elapsed = feats.groupby("seq")["elapsed_h"].first().to_dict()
    sc_map = args.scale / 0.5  # features were computed at scale 0.5

    frames = sorted(Path(args.frames_dir).glob("*.jpg"), key=lambda p: seq_index(p.name))
    frames = [p for p in frames if seq_index(p.name) >= 0][::args.every]

    writer = None
    for p in frames:
        seq = seq_index(p.name)
        bgr = load(p, args.scale)
        if writer is None:
            h, w = bgr.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        ng = 0
        for dish in cfg["dishes"]:
            c = (int(dish["cx"]*sc_map), int(dish["cy"]*sc_map))
            cv2.circle(bgr, c, int(dish["r"]*sc_map), (0, 200, 0), 1)
            gc = 0
            for seed in dish["seeds"]:
                sid = seed["seed_id"]
                x = kx.get((seq, sid), seed["x"]); y = ky.get((seq, sid), seed["y"])
                x = int(x*sc_map); y = int(y*sc_map)
                gs = germ_seq.get(sid)
                is_g = gs is not None and gs <= seq
                cv2.circle(bgr, (x, y), 6, (0, 255, 0) if is_g else (0, 0, 255), 2)
                gc += int(is_g); ng += int(is_g)
            cv2.putText(bgr, f"{dish['dish_id']}:{gc}/{dish['n_seeds']}",
                        (c[0]-40, c[1]-int(dish['r']*sc_map)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(bgr, f"seq {seq}  ~{elapsed.get(seq,0):.0f}h  germinated {ng}/{cfg['n_seeds_total']}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(bgr)
    writer.release()
    print("wrote", args.out, "frames", len(frames))

if __name__ == "__main__":
    main()
