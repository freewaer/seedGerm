#!/usr/bin/env python3
"""
Stage 1 - one-time calibration + seed map (annotation-free).

Detects the 5 petri dishes once on a clean early frame, detects the wheat kernels
inside each dish (splitting touching kernels), assigns stable seed IDs, and writes
seed_map.json. This fixed map is reused for every frame in the run.

Usage:
    python calibrate.py --frame <early.jpg> --out seed_map.json [--scale 0.5]
"""
import argparse, json, cv2, numpy as np
from pathlib import Path
import profiles

# scipy is only needed by split_kernels() during calibration. Importing it lazily
# keeps load()/kernel_mask()/dish_mask() usable (they are imported by the whole
# detector chain) in environments without scipy.

# ---------- shared detection helpers (imported by process.py too) ----------

_REDUCED_COLOR = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                  8: cv2.IMREAD_REDUCED_COLOR_8}


def load(path, scale, fast=False):
    """Read a frame at `scale`.

    fast=True lets libjpeg do the downscale during decode (IMREAD_REDUCED_*),
    which is ~2x quicker on the 4056x3040 HQ-cam JPEGs — worth it over a 200-frame
    run. Measured difference vs decode-then-INTER_AREA is 0.25/255 mean absolute
    per pixel, far below any threshold in this pipeline. Default stays off so the
    already-validated maize numbers reproduce bit-for-bit.
    """
    if fast and scale in (0.5, 0.25, 0.125):
        img = cv2.imread(str(path), _REDUCED_COLOR[int(round(1 / scale))])
        if img is not None:
            return img
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def find_dishes(bgr, want=5):
    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 7)
    H, W = gray.shape
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(W*0.13),
                               param1=120, param2=35, minRadius=int(W*0.085),
                               maxRadius=int(W*0.17))
    cand = []
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            if y > 0.72*H:          # reject rig/tray below the dishes
                continue
            cand.append((int(x), int(y), int(r)))
    kept = []
    for x, y, r in cand:
        if all((x-kx)**2 + (y-ky)**2 > (0.6*max(r, kr))**2 for kx, ky, kr in kept):
            kept.append((x, y, r))
    dishes = kept[:want]
    dishes.sort(key=lambda c: (round(c[1] / (0.15*H)), c[0]))   # row-major order
    return dishes

def kernel_mask(bgr, hsv_range=(3, 30, 70, 45)):
    """Orange/yellow kernels via HSV. Stable across the run (kernels stay coloured)."""
    h_lo, h_hi, s_min, v_min = hsv_range
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hc, S, V = cv2.split(hsv)
    m = ((Hc >= h_lo) & (Hc <= h_hi) & (S >= s_min) & (V >= v_min)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    return m

def split_kernels(km, rdish):
    """Return one entry per kernel as (cx, cy, area, w, h).

    Kernels that are separate stay as-is; only a blob that is clearly several
    touching kernels (area >> a single kernel) is split with a distance-transform
    watershed. This avoids dropping small separate kernels (the old global-max
    maxima threshold did) — important for maize (few, large, well-separated seeds).
    """
    from scipy.ndimage import maximum_filter
    binm = (km > 0).astype(np.uint8)
    if binm.sum() == 0:
        return []
    n, lab, stats, cent = cv2.connectedComponentsWithStats(binm, 8)
    min_a = (rdish*rdish)*0.002
    max_a = (rdish*rdish)*0.12
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
             if min_a <= stats[i, cv2.CC_STAT_AREA] <= max_a]
    if not areas:
        return []
    single = float(np.median(areas))          # typical one-kernel area
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_a or area > max_a:
            continue
        n_est = max(1, int(round(area / single)))
        comp = (lab == i).astype(np.uint8)
        if n_est == 1:
            cx, cy = cent[i]
            x0 = stats[i, cv2.CC_STAT_LEFT]; y0 = stats[i, cv2.CC_STAT_TOP]
            w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
            out.append((int(cx), int(cy), area, w, h))
            continue
        # split touching kernels within this component
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        eq_r = np.sqrt(single/np.pi)
        min_d = max(6, int(eq_r*1.2))
        mx = (dist == maximum_filter(dist, size=min_d)) & (dist > 0.4*eq_r)
        ys, xs = np.where(mx)
        seeds = []
        for x, y in zip(xs, ys):
            if all((x-sx)**2 + (y-sy)**2 > (min_d*0.8)**2 for sx, sy in seeds):
                seeds.append((x, y))
        if len(seeds) <= 1:
            cx, cy = cent[i]
            out.append((int(cx), int(cy), area, int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])))
            continue
        markers = np.zeros(comp.shape, np.int32)
        for j, (x, y) in enumerate(seeds, 1):
            markers[y, x] = j
        markers = cv2.watershed(cv2.cvtColor(comp*255, cv2.COLOR_GRAY2BGR), markers)
        for j in range(1, len(seeds)+1):
            reg = (markers == j); a = int(reg.sum())
            if a < min_a:
                continue
            ys2, xs2 = np.where(reg)
            out.append((int(xs2.mean()), int(ys2.mean()), a,
                        int(xs2.max()-xs2.min()+1), int(ys2.max()-ys2.min()+1)))
    return out

def dish_mask(shape, x, y, r, shrink=0.93):
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (x, y), int(r*shrink), 255, -1)
    return m

# ---------- calibration entry point ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", default="seed_map.json")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--annot", default="")
    ap.add_argument("--profile", default="wheat")
    args = ap.parse_args()

    bgr = load(args.frame, args.scale)
    khsv = profiles.get(args.profile)["kernel_hsv"]
    dishes = find_dishes(bgr)
    kmask = kernel_mask(bgr, khsv)

    kernel_sizes = []
    dishes_out = []
    vis = bgr.copy()
    for di, (x, y, r) in enumerate(dishes):
        dm = dish_mask(bgr.shape, x, y, r)
        km = cv2.bitwise_and(kmask, kmask, mask=dm)
        kernels = split_kernels(km, r)
        seeds = []
        for si, (cx, cy, area, w, h) in enumerate(kernels):
            eq_r = float(np.sqrt(area/np.pi))
            kernel_sizes.append(eq_r)
            seeds.append(dict(seed_id=f"D{di+1}_{si+1:02d}", x=cx, y=cy,
                              area=area, w=w, h=h, eq_r=eq_r))
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.putText(vis, f"D{di+1}:{len(seeds)}", (x-40, y-r-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        dishes_out.append(dict(dish_id=f"D{di+1}", cx=x, cy=y, r=r, n_seeds=len(seeds), seeds=seeds))

    med_r = float(np.median(kernel_sizes)) if kernel_sizes else 0.0
    cfg = dict(
        source_frame=str(args.frame), scale=args.scale,
        image_w=bgr.shape[1], image_h=bgr.shape[0],
        median_kernel_eq_r_px=med_r,
        n_dishes=len(dishes_out),
        n_seeds_total=sum(d["n_seeds"] for d in dishes_out),
        dishes=dishes_out,
    )
    Path(args.out).write_text(json.dumps(cfg, indent=2))
    if args.annot:
        cv2.imwrite(args.annot, vis)
    print(f"dishes={cfg['n_dishes']} seeds={cfg['n_seeds_total']} "
          f"per={[d['n_seeds'] for d in dishes_out]} med_kernel_r={med_r:.1f}px -> {args.out}")

if __name__ == "__main__":
    main()
