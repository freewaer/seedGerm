#!/usr/bin/env python3
"""Stage 0 (wheat) — colour-based seed pre-annotation for CVAT. No GPU needed.

Why this exists instead of just tuning FastSAM: measured on the 150 wheat tiles,
FastSAM "segment everything" put only **5%** of tiles at the correct 20 seeds and
averaged 10.2 seeds/tile, collapsing to 1.4 by the end of the run. The failure is
not occlusion — at t=44h the seeds are plainly visible — it is that FastSAM
segments the whole SEEDLING: the grain and its coleoptile are one connected
object, so the returned mask is the shoot, or seed+shoot fused.

Wheat gives us a much stronger signal: the grain is brown/amber, the shoot is
green, the blotter is grey. The kernel HSV mask from profiles.py covers every
seed at every timepoint. So the seeds are found by colour, and the only real
problem left is separating grains that touch — handled by count-guided splitting.

Measured on wheatCam, 150 tiles:
    exactly 20 seeds     140/150 (93%)      [FastSAM: 7/150, 5%]
    mean seeds per tile  19.9               [FastSAM: 10.2]
    known seeds found    91/95 (96%), median centre error 2.5px (seed radius 29px)

IMPORTANT — what the count guidance does and does not prove: the splitter keeps
subdividing the largest blob until the tile holds the number of seeds actually
SOWN (`seeds_per_dish`). That is legitimate prior knowledge, but it means the
per-tile count is NOT independent evidence that 20 seeds are visible. A forced
split is flagged (`"split": true` in the COCO attributes, and counted in the QC
report) so review can start where the geometry was invented rather than measured.

    python pre_annotate_color.py --crop wheat --out-dir ../datasets/seed_wheat_sam

Output matches pre_annotate_sam.py exactly: COCO 1.0 + YOLO-seg labels + images +
prelabel_counts.csv, ready for CVAT and then build_dataset.py.
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import cv2

try:
    from calibrate import load, kernel_mask
    from process import seq_index
    from dish_tiles import DishTiler
    import profiles, cropcfg
except Exception:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from calibrate import load, kernel_mask
    from process import seq_index
    from dish_tiles import DishTiler
    import profiles, cropcfg


def seed_likeness(mask, a0):
    """High for compact, seed-sized blobs; low for elongated root/shoot fragments.

    Used to decide which candidates to drop when colour noise (browning roots late
    in the run) yields more pieces than seeds were sown.
    """
    a = float(mask.sum())
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return -1.0
    (_, (w, h), _) = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    elong = max(w, h) / max(1e-6, min(w, h))
    fill = a / max(1e-6, w * h)
    size = min(a, a0) / max(a, a0)          # 1.0 when the blob is exactly seed-sized
    return float(size * fill / elong)


def colour_blobs(bgr, khsv, a0, min_area_factor, close_px=5):
    km = (kernel_mask(bgr, khsv) > 0).astype(np.uint8)
    if close_px >= 3:
        km = cv2.morphologyEx(km, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(km, 8)
    return [(lab == i) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area_factor * a0]


def split_blob(mask, k, a0, min_area_factor):
    """Cut a fused blob into k parts by k-means over its pixels.

    k-means beats a distance-transform watershed here because touching wheat
    grains form a smooth dumbbell with no reliable distance minimum, while their
    pixel cloud is clearly bimodal.
    """
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(np.float32)
    if len(P) < k * 8 or k < 2:
        return [mask]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lb, _ = cv2.kmeans(P, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    out = []
    lb = lb.ravel()
    for v in range(k):
        m = np.zeros_like(mask)
        m[ys[lb == v], xs[lb == v]] = True
        if m.sum() >= min_area_factor * a0:
            out.append(m)
    return out or [mask]


def core_count(bgr, khsv, r, close_px=5):
    """Independent seed-count estimate, using NO knowledge of how many were sown.

    A wheat grain is thick (~2r across); its coleoptile and roots are thin. So an
    opening with a disk of ~0.55r erases the appendages and leaves one compact
    "core" per grain — including where the shoot is pale brown rather than green
    and the colour mask fuses grain and shoot into one blob (dish D5 here).

    This is the number that makes the count prior honest: where it independently
    lands on the sown count, the 20 is measured, not assumed. Measured on wheatCam
    it averages 20.3 over seq 1-104 (t<=52h) and only diverges past t~65h, when
    browning root mats fragment into extra cores.
    """
    a0 = np.pi * r * r
    km = (kernel_mask(bgr, khsv) > 0).astype(np.uint8)
    km = cv2.morphologyEx(km, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    k = int(max(3, round(0.55 * r)) | 1)
    core = cv2.morphologyEx(km, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    areas = [float(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 0.10 * a0]
    if not areas:
        return 0
    # Two grains that touch keep one fused core, so counting components alone
    # under-reports. Scale each core by the typical single-core area instead —
    # the median core is a single grain as long as most grains are separate.
    unit = float(np.median(areas))
    return int(sum(max(1, int(round(x / unit))) for x in areas))


def find_seeds(bgr, khsv, a0, expect, min_area_factor=0.22, close_px=5, max_iter=60):
    """Returns (masks, n_forced_splits)."""
    cand = colour_blobs(bgr, khsv, a0, min_area_factor, close_px)
    forced = 0
    for _ in range(max_iter):
        if not cand or len(cand) >= expect:
            break
        big = max(cand, key=lambda m: m.sum())
        if big.sum() < 1.5 * a0:            # nothing left that plausibly holds 2 seeds
            break
        k = max(2, int(round(big.sum() / a0)))
        parts = split_blob(big, min(k, expect - len(cand) + 1), a0, min_area_factor)
        if len(parts) < 2:
            break
        forced += len(parts) - 1
        cand = [c for c in cand if c is not big] + parts
    if len(cand) > expect:
        cand = sorted(cand, key=lambda m: seed_likeness(m, a0), reverse=True)[:expect]
    return cand, forced


def mask_to_poly(mask, eps_frac=0.012):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 8:
        return None
    ap = cv2.approxPolyDP(c, max(1.5, eps_frac * cv2.arcLength(c, True)), True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    return [float(v) for p in ap for v in p]


def poly_meta(seg):
    pts = np.array(seg, np.float32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    return ([float(x), float(y), float(w), float(h)],
            float(cv2.contourArea(pts.reshape(-1, 1, 2))))


def select_frames(frames, max_frames):
    frames = sorted(frames, key=lambda p: seq_index(p.name))
    if len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[i] for i in sorted(set(idx))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True, help="key in models.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--seq-min", type=int, default=0)
    ap.add_argument("--seq-max", type=int, default=0,
                    help="ignore frames after this index — use it to stay inside the "
                         "window where the independent check still confirms the count")
    ap.add_argument("--expect-per-image", type=int, default=None)
    ap.add_argument("--min-area-factor", type=float, default=0.22)
    ap.add_argument("--close-px", type=int, default=5)
    ap.add_argument("--no-tile-dishes", dest="tile_dishes", action="store_false", default=None)
    a = ap.parse_args()

    cfg = cropcfg.load(a.crop)
    prof = profiles.get(cfg["profile"])
    khsv = prof["kernel_hsv"]
    scale = cfg["scale"]
    frames_dir = Path(a.frames_dir) if a.frames_dir else cfg.path("frames_dir")
    tile_dishes = cfg["tile_dishes"] if a.tile_dishes is None else a.tile_dishes

    tiler = DishTiler.from_map(cfg.path("seed_map"), scale=scale, tile_px=cfg["tile_px"])
    expect = a.expect_per_image or (cfg["seeds_per_dish"] if tile_dishes else cfg.total_seeds)

    # Seed area must be PER DISH. The dishes have different radii (198-258 px here)
    # while the tile is a fixed 1024 px, so each dish is magnified differently
    # (1.74x - 2.27x). Using the mean magnification puts the seed-area threshold off
    # by up to +42%, which then drives the splitter: too-small a0 makes it over-split
    # (D5 averaged 6.4 forced splits/tile), too-large a0 makes it miss fused pairs.
    smap = json.loads(cfg.path("seed_map").read_text(encoding="utf-8"))
    map_scale = float(smap.get("scale", 0.5))

    def dish_r(i):
        """Seed radius in TILE pixels for dish i — from that dish's OWN seeds.
        The map stores a single global median, but per dish it ranges 12.7-15.0 px
        here, and after the differing tile magnifications the tile-space radius
        spans 25.6-32.8 px. One global value mis-sizes the splitter."""
        rs = [sd["eq_r"] for sd in smap["dishes"][i]["seeds"]] if i < len(smap["dishes"]) else []
        r = float(np.median(rs)) if rs else float(tiler.med_r_px / (scale / map_scale))
        r *= scale / map_scale
        return r * (tiler.factor(i) if tile_dishes else 1.0)

    def dish_a0(i):
        r = dish_r(i)
        return float(np.pi * r * r)

    print(f">> crop={a.crop} profile={cfg['profile']} kernel_hsv={khsv}")
    if tile_dishes:
        print(f">> dish tiles, expecting {expect} seeds each; per-dish seed radius: "
              + ", ".join(f"D{i+1}={dish_r(i):.0f}px" for i in range(len(tiler.dishes))))
    else:
        print(f">> whole frame: seed radius {tiler.med_r_px:.0f}px, "
              f"expecting {expect} seeds per image")

    frames = [p for p in frames_dir.glob("*.jpg") if seq_index(p.name) >= 0]
    if a.seq_min:
        frames = [p for p in frames if seq_index(p.name) >= a.seq_min]
    if a.seq_max:
        frames = [p for p in frames if seq_index(p.name) <= a.seq_max]
    if not frames:
        raise SystemExit(f"no frames in {frames_dir} within the requested seq range")
    picked = select_frames(frames, a.max_frames)
    lo, hi = seq_index(picked[0].name), seq_index(picked[-1].name)
    print(f">> {len(picked)} frames, seq {lo}-{hi} "
          f"(t = {cfg['dt_hours']*(lo-1):.1f}-{cfg['dt_hours']*(hi-1):.1f} h)")

    out = Path(a.out_dir)
    for sub in ("images", "labels", "annotations"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    coco = {"info": {"description": f"GermiCam {a.crop} colour pre-annotations (review in CVAT)"},
            "licenses": [], "categories": [{"id": 1, "name": "seed", "supercategory": "seed"}],
            "images": [], "annotations": []}
    ann_id = img_id = 0
    counts = []

    def emit(name, bgr, masks, forced, a0, cores):
        nonlocal ann_id, img_id
        img_id += 1
        H, W = bgr.shape[:2]
        cv2.imwrite(str(out / "images" / name), bgr)
        coco["images"].append({"id": img_id, "file_name": name, "width": W, "height": H})
        # the largest `forced` pieces are the invented ones; flag the smallest-likeness
        order = sorted(range(len(masks)), key=lambda i: seed_likeness(masks[i], a0))
        suspect = set(order[:forced]) if forced else set()
        yolo = []
        for i, m in enumerate(masks):
            seg = mask_to_poly(m)
            if seg is None:
                continue
            bbox, area = poly_meta(seg)
            ann_id += 1
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [seg], "bbox": bbox, "area": area, "iscrowd": 0,
                "attributes": {"auto": True, "split": bool(i in suspect)}})
            yolo.append("0 " + " ".join(f"{seg[j]/W:.6f} {seg[j+1]/H:.6f}"
                                        for j in range(0, len(seg), 2)))
        (out / "labels" / (Path(name).stem + ".txt")).write_text(
            "\n".join(yolo) + ("\n" if yolo else ""))
        counts.append((name, len(yolo), forced, cores))
        d = len(yolo) - expect
        agree = abs(cores - expect) <= 1
        print(f"  {name}: {len(yolo)} seeds"
              + ("  OK" if d == 0 else f"  {d:+d}")
              + (f"  [{forced} forced]" if forced else "")
              + (f"  independent={cores} {'CONFIRMED' if agree else 'CHECK'}"))

    for p in picked:
        bgr = load(p, scale, fast=True)
        if tile_dishes:
            for di, tile in tiler.tiles(bgr):
                a0 = dish_a0(di)
                masks, forced = find_seeds(tile, khsv, a0, expect,
                                           a.min_area_factor, a.close_px)
                cores = core_count(tile, khsv, dish_r(di), a.close_px)
                emit(tiler.tile_name(p.name, di), tile, masks, forced, a0, cores)
        else:
            a0 = dish_a0(0)
            masks, forced = find_seeds(bgr, khsv, a0, expect, a.min_area_factor, a.close_px)
            cores = core_count(bgr, khsv, dish_r(0), a.close_px)
            emit(p.name, bgr, masks, forced, a0, cores)

    tmp = out / "annotations" / "instances_default.json.tmp"
    tmp.write_text(json.dumps(coco))
    os.replace(tmp, out / "annotations" / "instances_default.json")
    (out / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: seed\n")
    (out / "classes.txt").write_text("seed\n")
    (out / "prelabel_counts.csv").write_text(
        "image,n_seeds,forced_splits,independent_count,confirmed\n"
        + "\n".join(f"{n},{c},{f},{k},{int(abs(k-expect)<=1)}" for n, c, f, k in counts) + "\n")

    ns = np.array([c for _, c, _, _ in counts])
    fs = np.array([f for _, _, f, _ in counts])
    ks = np.array([k for _, _, _, k in counts])
    exact = int((ns == expect).sum())
    print(f"\nimages={len(counts)}  seed_instances={ann_id}")
    print(f"per image: min={ns.min()} median={int(np.median(ns))} max={ns.max()} "
          f"mean={ns.mean():.1f}")
    print(f"exactly {expect}: {exact}/{len(ns)} images ({100*exact/len(ns):.0f}%)")
    print(f"forced splits: {int(fs.sum())} instances over {int((fs>0).sum())} images "
          f"— review these first, the geometry there is inferred, not measured")
    conf = int((np.abs(ks - expect) <= 1).sum())
    print(f"INDEPENDENT check (morphology only, ignores the sown count): "
          f"mean {ks.mean():.1f}, agrees with {expect} on {conf}/{len(ks)} images "
          f"({100*conf/len(ks):.0f}%)")
    print("  -> where it agrees, the count is MEASURED; where it does not, the tile's "
          "20 seeds are inferred from what you sowed and need real review")
    worst = sorted(counts, key=lambda t: (-abs(t[3] - expect), -t[2]))[:8]
    for n, c, f, k in worst:
        if abs(k - expect) > 1 or f:
            print(f"   {n}: {c} seeds, {f} forced, independent {k}")
    print(f"\nQC counts -> {out / 'prelabel_counts.csv'}")
    print(f"next: upload {out / 'images'} to CVAT (one polygon label named 'seed'), "
          f"import annotations/instances_default.json as COCO 1.0")


if __name__ == "__main__":
    main()
