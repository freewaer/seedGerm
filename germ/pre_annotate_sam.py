#!/usr/bin/env python3
"""Stage 0 - SAM/SAM2-assisted SINGLE-CLASS seed pre-annotation for CVAT.

Segment every seed with zero training so a human only REVIEWS in CVAT. Decouples
DETECTION (find seeds) from GERMINATION SCORING (separate stage).

Segmenter (auto-selected by --model name):
  * FastSAM-s.pt / FastSAM-x.pt -> ultralytics FastSAM (fast, lighter masks)
  * sam2_l.pt / sam2_b.pt / sam_l.pt / mobile_sam.pt -> ultralytics SAM (SAM2 = best
    mask edges; slower). Weights auto-download on first use.

Pipeline: "segment everything" -> filter to seeds by area and (if --map) petri-dish
circle -> --kernel-only colour filter (weak for maize) -> proximity_merge collapses
fragments -> COCO 1.0 (atomic write) + YOLO-seg labels + data.yaml, one class 'seed'.

  python pre_annotate_sam.py --frames-dir ../maizeCam --out-dir ../datasets/seed_maize_sam2 \
      --map seed_map_maize.json --profile maize --model sam2_l.pt --max-frames 30 --device xpu
  (if SAM2 hits an unsupported XPU op, add --device cpu)
"""
import argparse, json, os, shutil
from pathlib import Path
import numpy as np, cv2

try:
    from process import seq_index
    from calibrate import load, kernel_mask
    import profiles
except Exception:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from process import seq_index
    from calibrate import load, kernel_mask
    import profiles


def install_device(device):
    import torch
    dev = device.lower()
    have_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if dev == "auto":
        dev = "xpu" if have_xpu else "cpu"
    if dev == "xpu" and not have_xpu:
        print(">> [warn] XPU not available, falling back to CPU"); dev = "cpu"
    if dev == "xpu":
        import ultralytics.utils.torch_utils as tu
        _orig = tu.select_device
        def sel(device="", *a, **k):
            if "xpu" in str(device).lower():
                print(">> [patch] Intel XPU:", torch.xpu.get_device_name(0))
                return torch.device("xpu:0")
            return _orig(device, *a, **k)
        tu.select_device = sel
        try:
            import ultralytics.engine.predictor as pr
            pr.select_device = sel
        except Exception:
            pass
    return dev


def simplify_poly(xy):
    c = xy.astype(np.float32).reshape(-1, 1, 2)
    if len(c) < 3 or cv2.contourArea(c) < 8:
        return None
    eps = max(1.5, 0.01 * cv2.arcLength(c, True))
    ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    return [float(v) for p in ap for v in p]


def poly_bbox_area(seg):
    pts = np.array(seg, np.float32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    area = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return [float(x), float(y), float(w), float(h)], area, (cx, cy)


def bbox_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def dedup(cands, iou_thr=0.7):
    cands = sorted(cands, key=lambda c: c["area"], reverse=True)
    kept = []
    for c in cands:
        dup = False
        for k in kept:
            if bbox_iou(c["bbox"], k["bbox"]) > iou_thr:
                dup = True; break
            kx, ky, kw, kh = k["bbox"]; cx, cy = c["center"]
            if (kx <= cx <= kx + kw and ky <= cy <= ky + kh
                    and c["area"] < 0.5 * k["area"]):
                dup = True; break
        if not dup:
            kept.append(c)
    return kept


def elongation(seg):
    pts = np.array(seg, np.float32).reshape(-1, 1, 2)
    (_, (w, h), _) = cv2.minAreaRect(pts)
    return max(w, h) / max(1e-6, min(w, h))


def proximity_merge(cands, merge_dist):
    n = len(cands)
    if n < 2 or merge_dist <= 0:
        return cands
    par = list(range(n))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    md2 = merge_dist * merge_dist
    for i in range(n):
        cix, ciy = cands[i]["center"]
        for j in range(i + 1, n):
            cjx, cjy = cands[j]["center"]
            if (cix - cjx) ** 2 + (ciy - cjy) ** 2 < md2:
                par[f(i)] = f(j)
    groups = {}
    for i in range(n):
        groups.setdefault(f(i), []).append(i)
    keep = []
    for g in groups.values():
        best = min(g, key=lambda i: (elongation(cands[i]["seg"]), -cands[i]["area"]))
        keep.append(cands[best])
    return keep


def kernel_fraction(seg, kmask):
    H, W = kmask.shape[:2]
    poly = np.zeros((H, W), np.uint8)
    pts = np.array(seg, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(poly, [pts], 1)
    a = int(poly.sum())
    if a == 0:
        return 0.0
    return float(((kmask > 0) & (poly > 0)).sum()) / a


def load_dishes(map_path, out_scale):
    if not map_path:
        return None, None
    cfg = json.loads(Path(map_path).read_text())
    map_scale = cfg.get("scale", 0.5)
    sc = out_scale / map_scale
    med_r = cfg["median_kernel_eq_r_px"] * sc
    dishes = [(d["cx"] * sc, d["cy"] * sc, d["r"] * sc) for d in cfg["dishes"]]
    return dishes, float(np.pi * med_r * med_r)


def in_any_dish(center, dishes, pad=1.08):
    cx, cy = center
    for dx, dy, dr in dishes:
        if (cx - dx) ** 2 + (cy - dy) ** 2 <= (dr * pad) ** 2:
            return True
    return False


def select_frames(frames, max_frames):
    seqs = [seq_index(p.name) for p in frames]
    order = np.argsort(seqs)
    frames = [frames[i] for i in order]
    if len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
    return [frames[i] for i in sorted(set(idx))]


def segment_image(model, is_fast, bgr, dev, a, amin, amax, seed_area,
                  inside=None, kmask=None):
    """Segment-everything on one image -> filtered, merged seed candidates."""
    if is_fast:
        res = model(bgr, device=dev, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                    retina_masks=True, verbose=False)
    else:
        res = model(bgr, device=dev, verbose=False)   # SAM/SAM2 segment-everything
    cands = []
    if res and res[0].masks is not None:
        for xy in res[0].masks.xy:
            if xy is None or len(xy) < 3:
                continue
            seg = simplify_poly(np.asarray(xy))
            if seg is None:
                continue
            bbox, area, center = poly_bbox_area(seg)
            if area < amin or area > amax:
                continue
            if inside is not None and not inside(center):
                continue
            if kmask is not None and kernel_fraction(seg, kmask) < a.kernel_frac:
                continue
            cands.append({"seg": seg, "bbox": bbox, "area": area, "center": center})
    cands = dedup(cands)
    if seed_area and a.merge_dist_factor > 0:
        merge_dist = a.merge_dist_factor * float(np.sqrt(seed_area / np.pi))
        cands = proximity_merge(cands, merge_dist)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default=None,
                    help="crop key in models.yaml — fills frames-dir/map/scale/profile/tiling")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--map", default=None)
    ap.add_argument("--model", default="FastSAM-s.pt",
                    help="FastSAM-s.pt / FastSAM-x.pt / sam2_l.pt / sam2_b.pt / mobile_sam.pt")
    ap.add_argument("--device", default="xpu", choices=["xpu", "cpu", "auto"])
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--min-area-factor", type=float, default=0.20)
    ap.add_argument("--max-area-factor", type=float, default=25.0)
    ap.add_argument("--min-area", type=float, default=150.0)
    ap.add_argument("--max-area", type=float, default=60000.0)
    ap.add_argument("--kernel-only", action="store_true")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--kernel-frac", type=float, default=0.25)
    ap.add_argument("--merge-dist-factor", type=float, default=1.3)
    ap.add_argument("--tile-dishes", dest="tile_dishes", action="store_true", default=None,
                    help="pre-annotate one image PER DISH (small/dense seeds, e.g. wheat)")
    ap.add_argument("--no-tile-dishes", dest="tile_dishes", action="store_false")
    ap.add_argument("--tile-px", type=int, default=None)
    ap.add_argument("--expect-per-image", type=int, default=None,
                    help="expected seeds per output image; drives the QC report")
    a = ap.parse_args()

    # ---- registry defaults (explicit flags always win) ----
    cfg = None
    if a.crop:
        import cropcfg
        cfg = cropcfg.load(a.crop)
        a.frames_dir = a.frames_dir or str(cfg.path("frames_dir"))
        a.map = a.map or str(cfg.path("seed_map"))
        a.scale = a.scale if a.scale is not None else cfg["scale"]
        a.profile = a.profile or cfg["profile"]
        a.tile_dishes = cfg["tile_dishes"] if a.tile_dishes is None else a.tile_dishes
        a.tile_px = a.tile_px or cfg["tile_px"]
        if a.expect_per_image is None:
            a.expect_per_image = cfg["seeds_per_dish"] if a.tile_dishes else cfg.total_seeds
    a.scale = 0.5 if a.scale is None else a.scale
    a.profile = a.profile or "maize"
    a.tile_dishes = bool(a.tile_dishes)
    a.tile_px = a.tile_px or 1024
    if not a.frames_dir:
        raise SystemExit("need --frames-dir (or --crop to take it from models.yaml)")

    khsv = None
    if a.kernel_only:
        khsv = profiles.get(a.profile)["kernel_hsv"]
        print(f">> kernel-only: profile '{a.profile}', min kernel fraction {a.kernel_frac}")

    dev = install_device(a.device)
    is_fast = "fastsam" in a.model.lower()
    if is_fast:
        from ultralytics import FastSAM
        model = FastSAM(a.model)
    else:
        from ultralytics import SAM          # sam2_l.pt / sam2_b.pt / sam_l.pt / mobile_sam.pt
        model = SAM(a.model)
    print(f">> segmenter: {a.model}  ({'FastSAM' if is_fast else 'SAM'})  device={dev}")

    dishes, seed_area = load_dishes(a.map, a.scale)

    tiler = None
    if a.tile_dishes:
        if not a.map:
            raise SystemExit("--tile-dishes needs --map (dish circles come from the seed map)")
        from dish_tiles import DishTiler
        tiler = DishTiler.from_map(a.map, scale=a.scale, tile_px=a.tile_px)
        # seed area must be expressed in TILE pixels, which are magnified
        f = float(np.mean([tiler.factor(i) for i in range(len(tiler.dishes))]))
        seed_area = seed_area * f * f
        print(f">> tiling: {len(tiler.dishes)} dishes -> {a.tile_px}px tiles "
              f"(magnification x{f:.2f}); seed diameter in tile "
              f"~{2 * tiler.med_r_px * f:.0f}px")

    if seed_area:
        amin, amax = a.min_area_factor * seed_area, a.max_area_factor * seed_area
        print(f">> filter: area [{amin:.0f}, {amax:.0f}] px^2 (seed-disk~{seed_area:.0f})")
    else:
        amin, amax = a.min_area, a.max_area
        print(f">> filter: no map, area [{amin:.0f}, {amax:.0f}] px^2")

    frames = [p for p in sorted(Path(a.frames_dir).glob("*.jpg")) if seq_index(p.name) >= 0]
    picked = select_frames(frames, a.max_frames)

    out = Path(a.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    coco = {"info": {"description": "GermiCam seed pre-annotations (review in CVAT)"},
            "licenses": [], "categories": [{"id": 1, "name": "seed", "supercategory": "seed"}],
            "images": [], "annotations": []}
    ann_id = 1
    img_id = 0
    counts = []          # (image_name, n_seeds) for the QC report

    def emit(name, bgr, cands):
        """Write one output image + its COCO/YOLO annotations."""
        nonlocal ann_id, img_id
        img_id += 1
        H, W = bgr.shape[:2]
        cv2.imwrite(str(out / "images" / name), bgr)
        coco["images"].append({"id": img_id, "file_name": name, "width": W, "height": H})
        yolo = []
        for c in cands:
            coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [c["seg"]], "bbox": c["bbox"], "area": c["area"],
                "iscrowd": 0, "attributes": {"auto": True}})
            pts = c["seg"]
            norm = " ".join(f"{pts[i]/W:.6f} {pts[i+1]/H:.6f}" for i in range(0, len(pts), 2))
            yolo.append(f"0 {norm}")
            ann_id += 1
        (out / "labels" / (Path(name).stem + ".txt")).write_text(
            "\n".join(yolo) + ("\n" if yolo else ""))
        counts.append((name, len(cands)))
        flag = ""
        if a.expect_per_image:
            d = len(cands) - a.expect_per_image
            flag = "  OK" if d == 0 else f"  {d:+d} vs expected {a.expect_per_image}"
        print(f"  {name}: {len(cands)} seeds{flag}")

    for p in picked:
        bgr_full = load(p, a.scale)
        if tiler is None:
            kmask = kernel_mask(bgr_full, khsv) if a.kernel_only else None
            inside = (lambda c: in_any_dish(c, dishes)) if dishes is not None else None
            cands = segment_image(model, is_fast, bgr_full, dev, a, amin, amax,
                                  seed_area, inside, kmask)
            emit(p.name, bgr_full, cands)
        else:
            for di, tile in tiler.tiles(bgr_full):
                kmask = kernel_mask(tile, khsv) if a.kernel_only else None
                # inside the dish circle, which is centred in the tile by construction
                c0 = a.tile_px / 2.0
                rad = (a.tile_px / 2.0) / tiler.pad
                inside = (lambda c, c0=c0, rad=rad:
                          (c[0] - c0) ** 2 + (c[1] - c0) ** 2 <= (rad * 1.02) ** 2)
                cands = segment_image(model, is_fast, tile, dev, a, amin, amax,
                                      seed_area, inside, kmask)
                emit(tiler.tile_name(p.name, di), tile, cands)

    ann_path = out / "annotations" / "instances_default.json"
    tmp_path = out / "annotations" / "instances_default.json.tmp"
    tmp_path.write_text(json.dumps(coco))
    os.replace(tmp_path, ann_path)
    (out / "data.yaml").write_text(
        "# Ultralytics YOLO-seg dataset (SAM pre-labels, CVAT-corrected)\n"
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: seed\n")
    (out / "classes.txt").write_text("seed\n")
    (out / "README_CVAT_SEED.txt").write_text(
        "CVAT import - single class 'seed' (instance segmentation)\n"
        "1. New task; add ONE label named exactly 'seed' as POLYGON.\n"
        "2. Upload the images/ folder as the task data.\n"
        "3. Actions -> Upload annotations -> 'COCO 1.0' -> annotations/instances_default.json\n"
        "4. Review/correct auto masks (add missed, delete stray, fix edges).\n"
        "5. Export as 'COCO 1.0', then: python build_dataset.py --crop <crop> --coco <export.zip|json>\n")

    # ---- QC report: which images need the most review attention ----
    ns = np.array([c for _, c in counts], float)
    print(f"\nimages={len(counts)}  seed_instances={ann_id-1}  device={dev}  -> {out}")
    if len(ns):
        print(f"per image: min={int(ns.min())} median={int(np.median(ns))} "
              f"max={int(ns.max())} mean={ns.mean():.1f}")
    if a.expect_per_image and len(ns):
        exact = int((ns == a.expect_per_image).sum())
        print(f"exactly {a.expect_per_image}: {exact}/{len(ns)} images "
              f"({100*exact/len(ns):.0f}%)")
        worst = sorted(counts, key=lambda t: -abs(t[1] - a.expect_per_image))[:8]
        if worst and worst[0][1] != a.expect_per_image:
            print("review these first (largest deviation):")
            for n, c in worst:
                if c != a.expect_per_image:
                    print(f"   {n}: {c} ({c - a.expect_per_image:+d})")
    qc = out / "prelabel_counts.csv"
    qc.write_text("image,n_seeds\n" + "\n".join(f"{n},{c}" for n, c in counts) + "\n")
    print(f"QC counts -> {qc}")


if __name__ == "__main__":
    main()
