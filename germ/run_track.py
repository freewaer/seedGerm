#!/usr/bin/env python3
"""Stage 2 — detect + track + count seeds across a fixed-camera timelapse (Intel XPU).

The camera and the dishes are fixed; only seeds move (they roll as the radicle
pushes). So: detect every frame, track with BoT-SORT for persistent IDs, assign
each detection to a dish, cap at the number of seeds actually sown, and write
per-frame counts + a per-seed detection table (input to the germination stage).

Two detection modes, chosen per crop in models.yaml:

  tile_dishes: false  (maize — 6 large seeds/dish)
      Detect on the whole frame; dish centres come from k-means over all
      detections.

  tile_dishes: true   (wheat — 20 small, touching seeds/dish)
      Crop each dish and upscale it to a fixed tile, then detect on the tile.
      The seed lands at ~33 px at the network input instead of ~9 px, the dish
      identity is known from the tile (no clustering), and each tracker only has
      to disambiguate 20 seeds inside one dish instead of 100 across the frame.
      Tracking runs dish-by-dish over the whole run, resetting between dishes.

    python run_track.py --crop wheat            # everything from models.yaml
    python run_track.py --crop maize --conf 0.35

Outputs in <out_dir>: per_frame_counts.csv, detections.csv
(seq,dish,track_id,cx,cy,conf — cx/cy always in FRAME coordinates at `scale`).
"""
import argparse, csv, time
from pathlib import Path
import numpy as np
import cv2

try:
    from process import seq_index
    from calibrate import load
    from dish_tiles import DishTiler
    import cropcfg
except Exception:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from process import seq_index
    from calibrate import load
    from dish_tiles import DishTiler
    import cropcfg


def patch_xpu():
    # imported here, not at module scope, so --help and the config
    # guards work in an environment without torch installed
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        import ultralytics.utils.torch_utils as tu
        _orig = tu.select_device

        def sel(device="", *a, **k):
            if "xpu" in str(device).lower():
                return torch.device("xpu:0")
            return _orig(device, *a, **k)

        tu.select_device = sel
        for mod in ("ultralytics.engine.predictor",):
            try:
                import importlib
                importlib.import_module(mod).select_device = sel
            except Exception:
                pass
        return "xpu"
    print(">> [warn] no XPU, running on CPU")
    return "cpu"


def reset_tracker(model):
    """Drop tracker state so the next dish starts with fresh IDs."""
    pr = getattr(model, "predictor", None)
    if pr is None:
        return
    for t in getattr(pr, "trackers", []) or []:
        try:
            t.reset()
        except Exception:
            pass
    try:
        del model.predictor.trackers
    except Exception:
        pass
    model.predictor = None


def track_boxes(model, img, dev, a, persist=True):
    """Run one tracked inference; returns list of (cx, cy, conf, track_id)."""
    r = model.track(source=img, persist=persist, imgsz=a.imgsz, conf=a.conf,
                    iou=a.iou, agnostic_nms=True, tracker=a.tracker,
                    device=dev, verbose=False)
    b = r[0].boxes
    if b is None or len(b) == 0:
        return []
    xywh = b.xywh.cpu().numpy()
    cf = b.conf.cpu().numpy()
    ids = (b.id.cpu().numpy().astype(int) if b.id is not None
           else np.full(len(cf), -1, int))
    return [(float(xywh[k][0]), float(xywh[k][1]), float(cf[k]), int(ids[k]))
            for k in range(len(cf))]


def out_track_path(a):
    return Path(a.out) / "dish_track.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default=None, help="key in models.yaml — supplies every default")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--n-dishes", type=int, default=None)
    ap.add_argument("--max-per-dish", type=int, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="xpu", choices=["xpu", "cpu"])
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--dt-hours", type=float, default=None)
    ap.add_argument("--tile-dishes", dest="tile_dishes", action="store_true", default=None)
    ap.add_argument("--no-tile-dishes", dest="tile_dishes", action="store_false")
    ap.add_argument("--tile-px", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N frames (smoke test)")
    a = ap.parse_args()

    if a.crop:
        cfg = cropcfg.load(a.crop)
        a.weights = a.weights or str(cropcfg.require_weights(cfg))
        a.frames_dir = a.frames_dir or str(cfg.path("frames_dir"))
        a.out = a.out or str(cfg.path("out_dir"))
        a.map = a.map or str(cfg.path("seed_map"))
        a.n_dishes = a.n_dishes or cfg["n_dishes"]
        a.max_per_dish = a.max_per_dish or cfg["seeds_per_dish"]
        a.scale = a.scale if a.scale is not None else cfg["scale"]
        a.imgsz = a.imgsz or cfg["imgsz"]
        a.dt_hours = a.dt_hours if a.dt_hours is not None else cfg["dt_hours"]
        a.tile_dishes = cfg["tile_dishes"] if a.tile_dishes is None else a.tile_dishes
        a.tile_px = a.tile_px or cfg["tile_px"]
    missing = [k for k in ("weights", "frames_dir", "out") if not getattr(a, k)]
    if missing:
        raise SystemExit(f"missing {missing} — pass them or use --crop")
    a.n_dishes = a.n_dishes or 5
    a.max_per_dish = a.max_per_dish or 6
    a.scale = 0.5 if a.scale is None else a.scale
    a.imgsz = a.imgsz or 640
    a.dt_hours = 0.5 if a.dt_hours is None else a.dt_hours
    a.tile_dishes = bool(a.tile_dishes)
    a.tile_px = a.tile_px or 1024

    dev = patch_xpu() if a.device == "xpu" else "cpu"
    print(f">> device: {dev}  weights: {a.weights}")
    from ultralytics import YOLO
    model = YOLO(a.weights)

    frames = [p for p in sorted(Path(a.frames_dir).glob("*.jpg")) if seq_index(p.name) >= 0]
    frames.sort(key=lambda p: seq_index(p.name))
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        raise SystemExit(f"no frames in {a.frames_dir}")
    print(f">> {len(frames)} frames, expecting {a.n_dishes}x{a.max_per_dish} "
          f"= {a.n_dishes * a.max_per_dish} seeds")

    dets = []          # (seq, cx, cy, conf, tid, dish)  cx/cy in frame coords
    assign_dish = None
    t0 = time.time()

    if a.tile_dishes:
        # ---- per-dish tiles: dish identity is known, tracking is per dish ----
        import dish_track
        track_map = dish_track.load_track(out_track_path(a), a.scale)
        if track_map is None:
            print(">> [note] no dish_track.json — tiles use the CALIBRATED dish positions, "
                  "which is correct only if the rig never moved. Verify with:\n"
                  f"     python dish_track.py --crop {a.crop or '<crop>'}")
        else:
            print(f">> dish track: {len(track_map)} frames of measured dish positions")
        tiler = DishTiler.from_map(a.map, scale=a.scale, tile_px=a.tile_px, track=track_map)
        if len(tiler.dishes) != a.n_dishes:
            print(f">> [warn] seed map has {len(tiler.dishes)} dishes, expected {a.n_dishes}")
        print(f">> tiling: {a.tile_px}px tiles; seed diameter at network input "
              f"~{tiler.apparent_seed_px(a.imgsz):.0f}px  [target 15-40]")
        total_steps = len(tiler.dishes) * len(frames)
        step, dropped = 0, 0
        for di in range(len(tiler.dishes)):
            reset_tracker(model)
            for p in frames:
                seq = seq_index(p.name)
                tile = tiler.tile(load(p, a.scale), di, seq)
                for (tx, ty, cf, tid) in track_boxes(model, tile, dev, a):
                    # the tile corners show NEIGHBOURING dishes — their seeds must
                    # not be attributed to this dish
                    if not tiler.in_dish(di, tx, ty):
                        dropped += 1
                        continue
                    fx, fy = tiler.to_frame(di, tx, ty, seq)
                    dets.append((seq, fx, fy, cf, tid, di))
                step += 1
                el = time.time() - t0
                print(f"\r   dish {di+1}/{len(tiler.dishes)}  {step}/{total_steps} tiles  "
                      f"{el:.0f}s  ETA {el/step*(total_steps-step):.0f}s  ({len(dets)} dets)",
                      end="", flush=True)
        print()
        if dropped:
            print(f">> dropped {dropped} detections outside the dish circle "
                  f"(seeds of neighbouring dishes visible in the tile corners)")
    else:
        # ---- whole frame: cluster detections into dishes afterwards ----
        for i, p in enumerate(frames, 1):
            seq = seq_index(p.name)
            for (cx, cy, cf, tid) in track_boxes(model, str(p), dev, a):
                dets.append((seq, cx, cy, cf, tid, -1))
            el = time.time() - t0
            print(f"\r   {i}/{len(frames)} frames  {el:.0f}s  "
                  f"ETA {el/i*(len(frames)-i):.0f}s  ({len(dets)} dets)", end="", flush=True)
        print()
        if not dets:
            raise SystemExit("No detections.")
        cen = np.array([[d[1], d[2]] for d in dets], np.float32)
        K = min(a.n_dishes, len(cen))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, _, centers = cv2.kmeans(cen, K, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        print(">> dish centers:", np.round(centers).tolist())

        def assign_dish(cx, cy):
            return int(np.argmin((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2))

    if not dets:
        raise SystemExit("No detections.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    by_seq = {}
    for seq, cx, cy, cf, tid, dish in dets:
        d = dish if dish >= 0 else assign_dish(cx, cy)
        by_seq.setdefault(seq, []).append((cx, cy, cf, tid, d))

    K = a.n_dishes
    per_frame, kept_rows = [], []
    for seq in sorted(by_seq):
        perdish = {}
        for (cx, cy, cf, tid, dish) in by_seq[seq]:
            perdish.setdefault(dish, []).append((cf, cx, cy, tid))
        counts = [0] * K
        total = 0
        for dish, lst in perdish.items():
            lst.sort(reverse=True)                       # keep the most confident
            keep = lst[:a.max_per_dish]
            if 0 <= dish < K:
                counts[dish] = len(keep)
            total += len(keep)
            for (cf, cx, cy, tid) in keep:
                kept_rows.append((seq, dish, tid, round(cx, 1), round(cy, 1), round(cf, 3)))
        per_frame.append((seq, round(a.dt_hours * (seq - 1), 2), *counts, total))

    with (out / "per_frame_counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "time_h"] + [f"dish{i}" for i in range(K)] + ["total"])
        w.writerows(per_frame)
    with (out / "detections.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "dish", "track_id", "cx", "cy", "conf"])
        w.writerows(kept_rows)

    tot = np.array([r[-1] for r in per_frame])
    expect = a.n_dishes * a.max_per_dish
    exact = int((tot == expect).sum())
    print(f">> frames={len(per_frame)}  seeds/frame: min={tot.min()} max={tot.max()} "
          f"median={int(np.median(tot))}")
    print(f">> exactly {expect} seeds in {exact}/{len(tot)} frames ({100*exact/len(tot):.0f}%)")
    if exact < 0.8 * len(tot):
        under = [c for c in per_frame if c[-1] < expect]
        print(f">> [check] {len(under)} frames under-count. If they cluster at the END of the "
              f"run this is sprout occlusion (expected); if at the START, lower --conf or "
              f"re-check the detector.")
        if under:
            print(f"   first under-count at seq {under[0][0]} (t={under[0][1]}h), "
                  f"last at seq {under[-1][0]} (t={under[-1][1]}h)")
    print(f">> wrote {out}/per_frame_counts.csv and detections.csv")
    print(f"\nnext:\n  python germinate_from_detections.py --crop {a.crop or '<crop>'}")


if __name__ == "__main__":
    main()
