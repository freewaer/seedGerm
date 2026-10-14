#!/usr/bin/env python3
"""Rig rigidity check — does the camera/dish geometry actually hold still?

Everything downstream (fixed dish tiles, dish-anchored counting, slot tracking)
assumes the rig does not move. That assumption deserves to be MEASURED, not
believed, because if it silently fails the counts drift without any error.

How, and why not the obvious way: locating the petri rim per frame with
HoughCircles looks like the natural test but is actively misleading — it happily
locks onto a lid rim, a neighbouring dish or a reflection and reports 100 px of
"drift" that is not there (it did exactly that on wheatCam while the rig was in
fact rock-steady). Instead we phase-correlate STATIC parts of the scene — rig
bars, tray, posts — which contain no seeds and therefore look the same at t=0 and
at the end of the run, no matter how much has germinated. Several independent
regions are used so one bad region cannot fake a verdict.

Output: <out_dir>/dish_track.json, holding the per-frame dish circles (the
calibrated circles plus the measured rigid shift). Consumers — dish_tiles.py,
run_track.py, germinate_from_detections.py — pick it up automatically. If the rig
is rigid the file is near-constant and harmless; if it is not, it is what keeps
the tiles on their dishes.

    python dish_track.py --crop wheat
    python dish_track.py --crop wheat --plot ../out_wheat/rig_drift.png
"""
import argparse, json
from pathlib import Path

import numpy as np
import cv2

try:
    from process import seq_index
    import cropcfg
except Exception:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from process import seq_index
    import cropcfg

REDUCED = {2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
           8: cv2.IMREAD_REDUCED_GRAYSCALE_8}

# fraction-of-frame boxes (x0, y0, x1, y1) that hold no seeds in this rig layout
STATIC_REGIONS = {
    "top bar":    (0.15, 0.00, 0.75, 0.07),
    "tray":       (0.10, 0.64, 0.85, 0.70),
    "left post":  (0.00, 0.07, 0.09, 0.60),
    "lower rig":  (0.30, 0.72, 0.75, 0.92),
}


def load_gray_fast(path, scale):
    """Decode straight to a reduced-size greyscale image.

    libjpeg's reduced-scale decode is several times faster than decoding
    4056x3040 and resizing, which matters over a 200-frame run.
    """
    for div in (8, 4, 2):
        if abs(scale - 1.0 / div) < 1e-6:
            g = cv2.imread(str(path), REDUCED[div])
            if g is not None:
                return g
    div = max((d for d in (8, 4, 2) if 1.0 / d >= scale), default=None)
    if div:
        g = cv2.imread(str(path), REDUCED[div])
        if g is not None:
            f = scale * div
            return (g if abs(f - 1.0) < 1e-6 else
                    cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _boxes(shape):
    H, W = shape[:2]
    out = {}
    for k, (a, b, c, d) in STATIC_REGIONS.items():
        x0, y0, x1, y1 = int(a * W), int(b * H), int(c * W), int(d * H)
        if x1 - x0 >= 32 and y1 - y0 >= 32:
            out[k] = (x0, y0, x1, y1)
    return out


def frame_shift(ref_f32, cur_f32, boxes, min_resp=0.15):
    """Median rigid (dx, dy) over the static regions, plus the mean confidence."""
    shifts, resps = [], []
    for (x0, y0, x1, y1) in boxes.values():
        a = ref_f32[y0:y1, x0:x1].copy()
        b = cur_f32[y0:y1, x0:x1].copy()
        if a.size == 0 or b.size == 0:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(a, b)
        if resp >= min_resp:
            shifts.append((dx, dy))
            resps.append(resp)
    if not shifts:
        return 0.0, 0.0, 0.0
    s = np.array(shifts)
    return float(np.median(s[:, 0])), float(np.median(s[:, 1])), float(np.mean(resps))


def measure(frames, scale, verbose=True):
    """Returns (seqs, shifts Nx2, mean confidences N)."""
    ref = load_gray_fast(frames[0], scale).astype(np.float32)
    boxes = _boxes(ref.shape)
    if verbose:
        print(f">> static regions used: {', '.join(boxes)}")
    seqs, shifts, resps = [], [], []
    for i, p in enumerate(frames):
        g = load_gray_fast(p, scale).astype(np.float32)
        dx, dy, r = frame_shift(ref, g, boxes)
        seqs.append(seq_index(p.name))
        shifts.append((dx, dy))
        resps.append(r)
        if verbose:
            print(f"\r   {i+1}/{len(frames)} frames", end="", flush=True)
    if verbose:
        print()
    return seqs, np.array(shifts), np.array(resps)


def verdict(shifts, resps, tol_px=3.0):
    mag = np.hypot(shifts[:, 0], shifts[:, 1])
    ok = bool(np.nanmax(mag) <= tol_px)
    return ok, float(np.nanmax(mag)), float(np.nanmedian(resps))


def save(path, scale, seqs, shifts, base_dishes):
    """Store per-frame dish circles = calibrated circle + measured rigid shift."""
    frames = {}
    for s, (dx, dy) in zip(seqs, shifts):
        frames[str(int(s))] = [(cx + dx, cy + dy, r) for cx, cy, r in base_dishes]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scale": scale, "method": "rigid shift of static scene regions",
                                "frames": frames}), encoding="utf-8")
    print(f">> wrote {path}")


def load_track(path, scale=None):
    """Returns {seq: [(cx, cy, r), ...]} rescaled to `scale`, or None if absent."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    src = float(d.get("scale", 0.5))
    f = 1.0 if scale is None else float(scale) / src
    return {int(k): [(x * f, y * f, r * f) for x, y, r in v] for k, v in d["frames"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--tol-px", type=float, default=3.0,
                    help="drift above this (at `scale`) is reported as a moving rig")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    cfg = cropcfg.load(a.crop)
    scale = a.scale if a.scale is not None else cfg["scale"]
    frames_dir = Path(a.frames_dir) if a.frames_dir else cfg.path("frames_dir")
    out_dir = Path(a.out) if a.out else cfg.path("out_dir")

    m = json.loads(cfg.path("seed_map").read_text(encoding="utf-8"))
    sc = scale / float(m.get("scale", 0.5))
    base = [(d["cx"] * sc, d["cy"] * sc, d["r"] * sc) for d in m["dishes"]]

    frames = sorted([p for p in frames_dir.glob("*.jpg") if seq_index(p.name) >= 0],
                    key=lambda p: seq_index(p.name))
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        raise SystemExit(f"no frames in {frames_dir}")
    print(f">> checking rig rigidity over {len(frames)} frames (scale {scale})")

    seqs, shifts, resps = measure(frames, scale)
    ok, worst, conf = verdict(shifts, resps, a.tol_px)
    print(f">> max rigid shift {worst:.2f}px (tolerance {a.tol_px}px), "
          f"median confidence {conf:.2f}")
    if ok:
        print(">> RIG IS RIGID — fixed dish tiles and dish-anchored counting are valid.")
    else:
        print(">> RIG MOVED — the saved per-frame offsets will keep the tiles on their "
              "dishes, but check the footage: a bumped camera can also change focus "
              "and scale, which a translation cannot fix.")
        k = int(np.argmax(np.hypot(shifts[:, 0], shifts[:, 1])))
        print(f"   worst at seq {seqs[k]}: dx={shifts[k,0]:+.1f} dy={shifts[k,1]:+.1f}")

    save(out_dir / "dish_track.json", scale, seqs, shifts, base)

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(seqs, np.hypot(shifts[:, 0], shifts[:, 1]), label="|shift|")
            ax.axhline(a.tol_px, ls="--", c="r", label=f"tolerance {a.tol_px}px")
            ax.set_xlabel("frame"); ax.set_ylabel("rigid shift vs frame 1 (px)")
            ax.set_title(f"{a.crop}: rig stability"); ax.legend(); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(a.plot, dpi=110)
            print(f">> wrote {a.plot}")
        except ImportError:
            print(">> [skip] matplotlib not installed, no plot")


if __name__ == "__main__":
    main()
