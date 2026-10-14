#!/usr/bin/env python3
"""Stage 1b — CVAT COCO export -> YOLO-seg training set (train/val split).

This was done by hand for maize; here it is as a reusable, checked step.

    python build_dataset.py --crop wheat \
        --coco ../datasets/wheat_seed_review.zip \
        --images ../datasets/seed_wheat_sam/images \
        --out ../datasets/wheat_seed_yolo --val-frac 0.2

Accepts either the CVAT export .zip or the extracted instances_default.json.
The split is TIME-SPREAD (every k-th frame goes to val) so validation covers the
whole germination run rather than only the late, sprouted frames — with dish
tiles, all tiles of a frame are kept on the same side of the split, otherwise
val leaks near-identical images from train and the metrics lie.

Prints a per-image instance count so a miscounted crop (the failure that killed
the first YOLO attempt) is caught BEFORE training.
"""
import argparse, json, random, shutil, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    from process import seq_index
    from dish_tiles import DishTiler
except Exception:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from process import seq_index
    from dish_tiles import DishTiler


def read_coco(src):
    src = Path(src)
    if src.is_dir():
        cand = list(src.rglob("instances_default.json")) or list(src.rglob("*.json"))
        if not cand:
            raise SystemExit(f"no COCO json under {src}")
        return json.loads(cand[0].read_text(encoding="utf-8")), cand[0]
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if n.endswith(".json")]
            pick = ([n for n in names if n.endswith("instances_default.json")] or names)
            if not pick:
                raise SystemExit(f"no .json inside {src}")
            return json.loads(z.read(pick[0]).decode("utf-8")), f"{src}!{pick[0]}"
    return json.loads(src.read_text(encoding="utf-8")), src


def poly_to_yolo(seg, W, H):
    """COCO polygon (flat xy list) -> YOLO-seg normalised line."""
    pts = np.asarray(seg, np.float64).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0] / W, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / H, 0.0, 1.0)
    if len(pts) < 3:
        return None
    return "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)


def group_key(file_name):
    """Images that must not be split across train/val (all tiles of one frame)."""
    base, dish = DishTiler.parse_tile_name(file_name)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default=None, help="key in models.yaml (for defaults + QC)")
    ap.add_argument("--coco", required=True, help="CVAT export .zip, folder, or json")
    ap.add_argument("--images", required=True, help="folder holding the images referenced by the COCO")
    ap.add_argument("--out", default=None, help="dataset root to create")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--expect-per-image", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = None
    if a.crop:
        import cropcfg
        cfg = cropcfg.load(a.crop)
        if a.expect_per_image is None:
            a.expect_per_image = (cfg["seeds_per_dish"] if cfg["tile_dishes"]
                                  else cfg.total_seeds)
        a.out = a.out or str(Path(__file__).resolve().parent.parent /
                             "datasets" / f"{a.crop}_seed_yolo")
    if not a.out:
        raise SystemExit("need --out (or --crop to derive it)")

    coco, where = read_coco(a.coco)
    print(f">> COCO: {where}")
    imgs = {im["id"]: im for im in coco["images"]}
    anns = defaultdict(list)
    for an in coco["annotations"]:
        segs = an.get("segmentation") or []
        if isinstance(segs, dict):
            print(f"   [skip] RLE mask on image {an['image_id']} (need polygons)")
            continue
        for s in segs:
            if len(s) >= 6:
                anns[an["image_id"]].append(s)

    src_images = Path(a.images)
    counts = {im["file_name"]: len(anns.get(i, [])) for i, im in imgs.items()}
    n = np.array(list(counts.values()), float)
    print(f">> images={len(imgs)}  instances={int(n.sum())}  "
          f"per image: min={int(n.min())} median={int(np.median(n))} max={int(n.max())}")

    if a.expect_per_image:
        bad = {k: v for k, v in counts.items() if v != a.expect_per_image}
        print(f">> exactly {a.expect_per_image}: {len(counts)-len(bad)}/{len(counts)}")
        if bad:
            print("   MISCOUNTED IMAGES — fix these in CVAT before training:")
            for k, v in sorted(bad.items(), key=lambda t: -abs(t[1] - a.expect_per_image))[:20]:
                print(f"     {k}: {v} ({v - a.expect_per_image:+d})")
            print("   (a wrong count here is exactly what poisoned the first YOLO run)")

    # ---- time-spread split, grouped so frame tiles stay together ----
    groups = defaultdict(list)
    for i, im in imgs.items():
        groups[group_key(im["file_name"])].append(i)
    keys = sorted(groups, key=lambda k: (seq_index(k + ".jpg"), k))
    k_val = max(1, int(round(1.0 / max(a.val_frac, 1e-6))))
    val_keys = set(keys[k_val // 2::k_val])
    if len(val_keys) >= len(keys):
        val_keys = {keys[-1]}
    print(f">> split: {len(keys)-len(val_keys)} train / {len(val_keys)} val frames "
          f"(every {k_val}th, time-spread; tiles kept with their frame)")

    out = Path(a.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            f.unlink()

    written = {"train": 0, "val": 0}
    inst = {"train": 0, "val": 0}
    missing = []
    for key in keys:
        split = "val" if key in val_keys else "train"
        for i in groups[key]:
            im = imgs[i]
            src = src_images / im["file_name"]
            if not src.exists():
                missing.append(im["file_name"])
                continue
            shutil.copy2(src, out / "images" / split / im["file_name"])
            W, H = im["width"], im["height"]
            lines = [ln for s in anns.get(i, []) if (ln := poly_to_yolo(s, W, H))]
            (out / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            written[split] += 1
            inst[split] += len(lines)
    if missing:
        print(f">> [warn] {len(missing)} images referenced by COCO not found in {src_images}"
              f" (first: {missing[:3]})")

    root = out.resolve().as_posix()
    (out / "data.yaml").write_text(
        f"# {a.crop or 'seed'} detector (YOLO-seg, single class) — CVAT-corrected labels\n"
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: seed\n",
        encoding="utf-8")

    print(f">> wrote {written['train']} train ({inst['train']} inst) / "
          f"{written['val']} val ({inst['val']} inst) -> {out}")
    if written["val"] == 0:
        raise SystemExit("val split is EMPTY — metrics would be meaningless. Lower --val-frac denominator.")
    imgsz = cfg["imgsz"] if cfg else 640
    print(f"\nnext:\n  python train_xpu.py --task segment --data {(out / 'data.yaml').as_posix()} \\\n"
          f"      --model yolo11n-seg.pt --epochs 80 --imgsz {imgsz} --batch 8")


if __name__ == "__main__":
    main()
