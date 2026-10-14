#!/usr/bin/env python3
"""Stage 3 — germination scoring from detector tracks (no torch needed).

Reads <out_dir>/detections.csv (seq,dish,track_id,cx,cy from run_track.py) plus
the raw frames, measures the sprout signal for each seed against ITS OWN
baseline, latches germination when the growth is sustained, and writes
germ_cue_per_seed.csv + germ_cue_per_frame.csv.

Why a growth-latch and not a classifier: proven three ways on maize that the
limit is the SIGNAL (pale radicle on pale blotter), not the algorithm. A
classifier was confidently wrong on detector crops; the latch, which only ever
compares a seed to its own earlier self, is the robust choice.

Crop differences come from models.yaml + profiles.py:
  wheat  seedlings GREEN up -> strong cue, ref_mode 'pale', 20 seeds/dish
  maize  seedlings stay PALE and the kernels ROLL -> ref_mode 'bright', 6/dish

    python germinate_from_detections.py --crop wheat
    python germinate_from_detections.py --crop maize --delta 1.2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import pandas as pd

try:
    from calibrate import load, kernel_mask
    from process import build_signal, measure_seed, seq_index, pale_mask
    import profiles, cropcfg
except Exception:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibrate import load, kernel_mask
    from process import build_signal, measure_seed, seq_index, pale_mask
    import profiles, cropcfg


def build_ref_sub(ref_bgr, prof):
    """Static background to subtract, per the crop's ref_mode (see profiles.py)."""
    mode = prof.get("ref_mode", "none")
    dil = max(3, int(prof["ref_dilate"]))
    if mode == "pale":
        rm = pale_mask(ref_bgr)
    elif mode == "bright":
        V = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        rm = ((V >= int(prof["ref_bright"])).astype(np.uint8)) * 255
    else:
        rm = np.zeros(ref_bgr.shape[:2], np.uint8)
    return cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)))


class SlotTracker:
    """Nearest-neighbour slot assignment inside a dish.

    Detector track IDs break when a seed is briefly missed, so seeds are pinned to
    persistent SLOTS by proximity instead. `gate` caps how far a seed may have
    moved between frames — it must be generous enough for rolling kernels (maize)
    but tight enough that with 20 dense seeds (wheat) a slot does not steal its
    neighbour's detection.
    """

    def __init__(self, n_slots, gate):
        self.n_slots = int(n_slots)
        self.gate2 = float(gate) ** 2
        self.last = {}        # (dish, slot) -> (cx, cy)
        self.next_slot = {}   # dish -> counter
        self.steals = 0

    def assign(self, dish, pts):
        cur = [(sid, xy) for (d, sid), xy in self.last.items() if d == dish]
        out = [None] * len(pts)
        used = set()
        # greedy globally-nearest first, so confident matches claim their slot
        pairs = []
        for i, (cx, cy) in enumerate(pts):
            for sid, (sx, sy) in cur:
                d2 = (cx - sx) ** 2 + (cy - sy) ** 2
                if d2 <= self.gate2:
                    pairs.append((d2, i, sid))
        pairs.sort()
        for d2, i, sid in pairs:
            if out[i] is None and sid not in used:
                out[i] = sid
                used.add(sid)
        for i, (cx, cy) in enumerate(pts):
            if out[i] is not None:
                continue
            n = self.next_slot.get(dish, 0)
            if n < self.n_slots:
                out[i] = n
                self.next_slot[dish] = n + 1
            else:
                # more detections than slots: attach to the nearest free-ish slot
                self.steals += 1
                cand = [s for s in range(self.n_slots) if s not in used] or list(range(self.n_slots))
                out[i] = min(cand, key=lambda s: (cx - self.last.get((dish, s), (1e9, 1e9))[0]) ** 2
                             + (cy - self.last.get((dish, s), (1e9, 1e9))[1]) ** 2)
                used.add(out[i])
        for (cx, cy), sid in zip(pts, out):
            self.last[(dish, sid)] = (cx, cy)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True, help="key in models.yaml")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--med-r", type=float, default=None, help="override seed radius (px at scale)")
    ap.add_argument("--slots", type=int, default=None, help="override seeds per dish")
    ap.add_argument("--ref-seq", type=int, default=1)
    ap.add_argument("--delta", type=float, default=None, help="override profile delta")
    ap.add_argument("--k", type=int, default=None, help="override profile k (consecutive frames)")
    ap.add_argument("--abs-floor", type=float, default=None)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--baseline-n", type=int, default=3)
    ap.add_argument("--slot-gate-factor", type=float, default=None,
                    help="max seed movement between frames, in seed radii")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth frame (quick preview; coarsens germination timing)")
    ap.add_argument("--slow-load", action="store_true",
                    help="decode full-res then resize instead of libjpeg reduced decode")
    a = ap.parse_args()
    fast = not a.slow_load

    cfg = cropcfg.load(a.crop)
    prof = profiles.get(cfg["profile"])
    frames_dir = Path(a.frames_dir) if a.frames_dir else cfg.path("frames_dir")
    out_dir = Path(a.out) if a.out else cfg.path("out_dir")
    scale = a.scale if a.scale is not None else cfg["scale"]
    slots = a.slots or cfg["seeds_per_dish"]

    # seed radius: prefer the calibration map (authoritative), fall back to registry
    med_r = a.med_r
    if med_r is None:
        smap = cfg.path("seed_map")
        if smap and smap.exists():
            m = json.loads(smap.read_text(encoding="utf-8"))
            med_r = float(m["median_kernel_eq_r_px"]) * (scale / float(m.get("scale", 0.5)))
        else:
            med_r = float(cfg["med_r_px"])

    delta = a.delta if a.delta is not None else prof["delta"]
    K = a.k if a.k is not None else prof["k"]
    abs_floor = a.abs_floor if a.abs_floor is not None else prof["abs_floor"]
    # dense dishes need a tighter gate or slots steal each other's seeds
    gate_f = a.slot_gate_factor if a.slot_gate_factor is not None else (6.0 if slots <= 8 else 2.5)

    print(f">> crop={a.crop} profile={cfg['profile']} slots/dish={slots} med_r={med_r:.2f}px "
          f"scale={scale}")
    print(f">> latch: delta={delta} k={K} abs_floor={abs_floor} ref_mode={prof['ref_mode']} "
          f"green_cue={prof['use_green']} slot_gate={gate_f:.1f}r")

    det_path = out_dir / "detections.csv"
    if not det_path.exists():
        raise SystemExit(f"{det_path} not found — run run_track.py --crop {a.crop} first")

    frames = {seq_index(p.name): p for p in frames_dir.glob("*.jpg") if seq_index(p.name) >= 0}
    if not frames:
        raise SystemExit(f"no frames in {frames_dir}")

    ref_seq = a.ref_seq if a.ref_seq in frames else min(frames)
    ref_sub = build_ref_sub(load(frames[ref_seq], scale, fast), prof)

    det = pd.read_csv(det_path)
    # detections.csv is written at the same `scale` run_track.py used
    seqs = sorted(int(s) for s in det["seq"].unique() if int(s) in frames)
    if a.stride > 1:
        seqs = seqs[::a.stride]
        print(f">> stride {a.stride}: using {len(seqs)} frames "
              f"(effective step {a.stride * cfg['dt_hours']}h — preview only)")
    if not seqs:
        raise SystemExit("detections.csv and frames do not overlap")

    # Dish drift (measured: up to ~100px on wheatCam) must be removed before slot
    # matching, otherwise a shifting dish looks like all its seeds jumping at once.
    import dish_track
    from dish_tiles import DishTiler
    tmap = dish_track.load_track(out_dir / "dish_track.json", scale)
    tiler = None
    if tmap:
        tiler = DishTiler.from_map(cfg.path("seed_map"), scale=scale,
                                   tile_px=cfg["tile_px"], track=tmap)
        print(f">> dish track loaded ({len(tmap)} frames) — slots matched in "
              f"dish-relative coordinates")
        uncovered = [s for s in seqs if s not in tmap]
        if uncovered:
            print(f">> [warn] dish track covers only {len(seqs) - len(uncovered)}/{len(seqs)} "
                  f"of the frames used; the rest fall back to the calibrated dish "
                  f"positions (fine for a rigid rig). Re-run dish_track.py without --limit.")
    else:
        print(">> [warn] no dish_track.json; slot matching uses raw frame coordinates. "
              f"If dishes moved during the run, run: python dish_track.py --crop {a.crop}")

    khsv = prof["kernel_hsv"]
    tracker = SlotTracker(slots, gate_f * med_r)
    series = {}          # (dish, slot) -> {seq: sprout length in seed radii}

    for n, seq in enumerate(seqs, 1):
        bgr = load(frames[seq], scale, fast)
        km = kernel_mask(bgr, khsv)
        sig = build_signal(bgr, prof, ref_sub)
        sub = det[det["seq"] == seq]
        for dish, gd in sub.groupby("dish"):
            dish = int(dish)
            pts = [(float(r.cx), float(r.cy)) for r in gd.itertuples()]
            if tiler is not None:
                key_pts = [tiler.to_dish_frame(dish, cx, cy, seq) for cx, cy in pts]
            else:
                key_pts = pts
            sids = tracker.assign(dish, key_pts)
            for (cx, cy), sid in zip(pts, sids):
                _, _, ext, _, _ = measure_seed({"x": int(round(cx)), "y": int(round(cy))},
                                               km, sig, med_r)
                series.setdefault((dish, sid), {})[seq] = ext / med_r
        print(f"\r   {n}/{len(seqs)} frames  ({len(series)} slots)", end="", flush=True)
    print()
    if tracker.steals:
        print(f">> [note] {tracker.steals} detections exceeded the slot budget and were "
              f"force-attached (expected if a frame over-counts)")

    # ---- latch per slot ----
    seq_arr = np.array(seqs)
    elapsed = {int(s): round(cfg["dt_hours"] * (int(s) - 1), 2) for s in seq_arr}
    per_seed, state_by = [], {}
    for (dish, sid), d in series.items():
        lr = np.array([d.get(int(s), np.nan) for s in seq_arr], float)
        if np.all(np.isnan(lr)):
            continue
        lr = pd.Series(lr).interpolate(limit_direction="both").to_numpy()
        sm = pd.Series(lr).rolling(a.smooth, center=True, min_periods=1).median().to_numpy()
        base = float(np.median(sm[:a.baseline_n]))
        signal = sm - base
        gi, run = -1, 0
        for i, v in enumerate(signal):
            ok = (v >= delta) and (sm[i] >= abs_floor)
            run = run + 1 if ok else 0
            if run >= K:
                gi = i - K + 1
                break
        germ = gi >= 0
        st = np.zeros(len(seq_arr), int)
        if germ:
            st[gi:] = 1
        state_by[(dish, sid)] = st
        per_seed.append(dict(dish=dish, slot=sid, germinated=int(germ),
                             germ_h=elapsed[int(seq_arr[gi])] if germ else "",
                             baseline_len_r=round(base, 2),
                             max_len_r=round(float(np.nanmax(sm)), 2)))

    ps = pd.DataFrame(per_seed).sort_values(["dish", "slot"])
    dishes = sorted(int(x) for x in det["dish"].unique())
    total_seeds = cfg.total_seeds

    rows = []
    for j, s in enumerate(seq_arr):
        row = {"seq": int(s), "time_h": elapsed[int(s)]}
        tot = 0
        for dsh in dishes:
            c = sum(int(state_by[(d, sid)][j]) for (d, sid) in state_by if d == dsh)
            row[f"dish{dsh}_germ"] = c
            tot += c
        row["total_germinated"] = tot
        row["total_seeds"] = total_seeds
        row["germ_pct"] = round(100 * tot / total_seeds, 1)
        rows.append(row)
    pf = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    ps.to_csv(out_dir / "germ_cue_per_seed.csv", index=False)
    pf.to_csv(out_dir / "germ_cue_per_frame.csv", index=False)

    gfin = int(ps["germinated"].sum())
    gt = sorted(float(x) for x in ps["germ_h"] if str(x) != "")
    t50 = gt[int(np.ceil(0.5 * len(gt))) - 1] if gt else None
    mgt = round(float(np.mean(gt)), 2) if gt else None
    n_slots = len(ps)
    print(f"slots={n_slots} (expected {total_seeds})  germinated_final={gfin} "
          f"({round(100*gfin/max(n_slots,1),1)}% of slots, "
          f"{round(100*gfin/total_seeds,1)}% of sown)  T50={t50}h MGT={mgt}h")
    if n_slots != total_seeds:
        print(f">> [warn] slot count != sown count. Check run_track.py counts "
              f"(per_frame_counts.csv) before trusting the percentage.")
    print("per dish:")
    for d in dishes:
        sub = ps[ps.dish == d]
        base = sub["baseline_len_r"].mean()
        note = "  <-- baseline high: cue swamped by bright background" if base > 5 else ""
        print(f"  dish{d}: {int(sub['germinated'].sum())}/{len(sub)}  baseline {base:.2f}{note}")
    if gt and gt[0] <= 2 * cfg["dt_hours"]:
        print(">> [warn] earliest germination at t=%.1fh — suspiciously early, possible "
              "glare false-positive. Consider raising --abs-floor." % gt[0])
    print(f">> wrote {out_dir}/germ_cue_per_seed.csv, germ_cue_per_frame.csv")
    print(f"\nnext:\n  python build_report.py --out {out_dir.as_posix()} --crop {a.crop}")


if __name__ == "__main__":
    main()
