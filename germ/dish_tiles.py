#!/usr/bin/env python3
"""Per-dish tiling — the answer to small, dense seeds (wheat: 100 seeds/frame).

Why: a detector only sees a seed reliably when the seed is ~15-40 px at the
NETWORK input. Maize (med_r 26.5 @scale 0.5) hits ~17 px on the whole frame at
imgsz 640. Wheat seeds are half that size and 3x denser, so the same setup gives
~9 px seeds — under the detection floor. Instead of pushing imgsz to 1280+ (slow
on the Arc iGPU, 4-5 min/epoch), we crop each petri dish and upscale it to a
fixed tile. The seed then appears at ~33 px — the validated maize regime — while
each image holds only 20 seeds, and imgsz stays at a fast 640.

Bonus: the dish identity comes from the tile, so no kmeans dish clustering is
needed at inference, and CVAT review is done on readable 1024 px tiles instead of
one 2028 px frame crammed with 100 tiny polygons.

    from dish_tiles import DishTiler
    t = DishTiler.from_map("seed_map_wheat.json", scale=0.5, tile_px=1024)
    for d, tile in t.tiles(bgr):        # bgr already loaded at `scale`
        ...                              # detect on `tile`
    fx, fy = t.to_frame(d, tx, ty)       # tile coords -> frame coords
"""
import json
from pathlib import Path
import numpy as np
import cv2

TILE_PAD = 1.14          # margin so rim seeds survive dish drift + Hough jitter


class DishTiler:
    """Cuts a fixed-size tile per petri dish.

    If a per-frame dish track is supplied (dish_track.py), the tile FOLLOWS the
    dish: pass `seq=` to any geometry call and the box is taken from that frame's
    measured circle. Without a track it falls back to the calibrated position,
    which is fine only for a genuinely static rig.
    """

    def __init__(self, dishes, tile_px=1024, pad=TILE_PAD, med_r_px=None, track=None):
        # dishes: list of (cx, cy, r) in the coordinate system of the images fed in
        self.dishes = [(float(x), float(y), float(r)) for x, y, r in dishes]
        self.tile_px = int(tile_px)
        self.pad = float(pad)
        self.med_r_px = med_r_px
        self.track = track or None       # {seq: [(cx, cy, r), ...]}

    # ---------- construction ----------

    @classmethod
    def from_map(cls, map_path, scale=0.5, tile_px=1024, pad=TILE_PAD, track=None):
        cfg = json.loads(Path(map_path).read_text(encoding="utf-8"))
        map_scale = float(cfg.get("scale", 0.5))
        sc = float(scale) / map_scale
        dishes = [(d["cx"] * sc, d["cy"] * sc, d["r"] * sc) for d in cfg["dishes"]]
        return cls(dishes, tile_px=tile_px, pad=pad, track=track,
                   med_r_px=float(cfg["median_kernel_eq_r_px"]) * sc)

    @classmethod
    def from_cfg(cls, cfg, scale=None, use_track=True):
        """Build from a cropcfg entry, picking up <out_dir>/dish_track.json if present."""
        scale = scale or cfg["scale"]
        track = None
        if use_track:
            import dish_track
            track = dish_track.load_track(cfg.path("out_dir") / "dish_track.json", scale)
        return cls.from_map(cfg.path("seed_map"), scale=scale,
                            tile_px=cfg["tile_px"], track=track)

    # ---------- geometry ----------

    def circle(self, i, seq=None):
        """(cx, cy, r) of dish i, from the per-frame track when available."""
        if seq is not None and self.track:
            row = self.track.get(int(seq))
            if row and i < len(row):
                return row[i]
        return self.dishes[i]

    def box(self, i, seq=None):
        """Square source box (x0, y0, side) for dish i, in image coordinates."""
        cx, cy, r = self.circle(i, seq)
        side = 2.0 * r * self.pad
        return cx - side / 2.0, cy - side / 2.0, side

    def factor(self, i, seq=None):
        _, _, side = self.box(i, seq)
        return self.tile_px / side

    def apparent_seed_px(self, imgsz, seq=None):
        """Seed diameter at the network input, averaged over dishes."""
        if self.med_r_px is None:
            return None
        f = np.mean([self.factor(i, seq) for i in range(len(self.dishes))])
        return float(2 * self.med_r_px * f * (imgsz / self.tile_px))

    def in_dish(self, i, tx, ty, tol=1.02):
        """Is a TILE-space point inside the dish circle? The tile corners show
        neighbouring dishes, whose seeds must not be counted for this dish."""
        c = self.tile_px / 2.0
        rad = (self.tile_px / 2.0) / self.pad * tol
        return (tx - c) ** 2 + (ty - c) ** 2 <= rad ** 2

    # ---------- cropping ----------

    def tile(self, bgr, i, seq=None):
        """Crop dish i and resize to tile_px. Out-of-frame area is edge-padded."""
        x0, y0, side = self.box(i, seq)
        H, W = bgr.shape[:2]
        xa, ya = int(np.floor(x0)), int(np.floor(y0))
        xb, yb = int(np.ceil(x0 + side)), int(np.ceil(y0 + side))
        px0, py0 = max(0, -xa), max(0, -ya)
        px1, py1 = max(0, xb - W), max(0, yb - H)
        sub = bgr[max(0, ya):min(H, yb), max(0, xa):min(W, xb)]
        if px0 or py0 or px1 or py1:
            sub = cv2.copyMakeBorder(sub, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
        return cv2.resize(sub, (self.tile_px, self.tile_px), interpolation=cv2.INTER_CUBIC)

    def tiles(self, bgr, seq=None):
        for i in range(len(self.dishes)):
            yield i, self.tile(bgr, i, seq)

    # ---------- coordinate mapping ----------

    def to_frame(self, i, tx, ty, seq=None):
        """Tile pixel -> image pixel."""
        x0, y0, side = self.box(i, seq)
        s = side / self.tile_px
        return x0 + tx * s, y0 + ty * s

    def to_frame_poly(self, i, pts, seq=None):
        x0, y0, side = self.box(i, seq)
        s = side / self.tile_px
        p = np.asarray(pts, np.float32).reshape(-1, 2)
        return np.stack([x0 + p[:, 0] * s, y0 + p[:, 1] * s], axis=1)

    def to_tile(self, i, fx, fy, seq=None):
        """Image pixel -> tile pixel."""
        x0, y0, side = self.box(i, seq)
        f = self.tile_px / side
        return (fx - x0) * f, (fy - y0) * f

    def to_dish_frame(self, i, fx, fy, seq=None):
        """Image pixel -> coordinates RELATIVE TO THE DISH CENTRE.

        Slot tracking must run in this frame: otherwise a dish that shifts 100 px
        looks like all 20 of its seeds jumping at once, and every slot is lost.
        """
        cx, cy, _ = self.circle(i, seq)
        return fx - cx, fy - cy

    def dish_of(self, fx, fy, seq=None):
        """Nearest dish index for an image-space point."""
        d = [(fx - c[0]) ** 2 + (fy - c[1]) ** 2
             for c in (self.circle(i, seq) for i in range(len(self.dishes)))]
        return int(np.argmin(d))

    def scale_len(self, i, tile_len, seq=None):
        _, _, side = self.box(i, seq)
        return tile_len * side / self.tile_px

    def tile_name(self, frame_name, i):
        return f"{Path(frame_name).stem}__D{i + 1}.jpg"

    @staticmethod
    def parse_tile_name(name):
        """'GermiCam_2026.03.12_7__D3.jpg' -> ('GermiCam_2026.03.12_7', 2)"""
        stem = Path(name).stem
        if "__D" not in stem:
            return stem, None
        base, _, d = stem.rpartition("__D")
        try:
            return base, int(d) - 1
        except ValueError:
            return stem, None


def main():
    """Quick visual QC: write annotated tiles for one frame."""
    import argparse
    from calibrate import load
    ap = argparse.ArgumentParser(description="Dump dish tiles of one frame for inspection")
    ap.add_argument("--frame", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--tile-px", type=int, default=1024)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="../out_tiles")
    a = ap.parse_args()

    t = DishTiler.from_map(a.map, scale=a.scale, tile_px=a.tile_px)
    bgr = load(a.frame, a.scale)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, tile in t.tiles(bgr):
        r = t.med_r_px * t.factor(i)
        cv2.circle(tile, (a.tile_px // 2, a.tile_px // 2), int(r), (0, 0, 255), 2)
        cv2.putText(tile, f"D{i+1}  seed r~{r:.0f}px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(out / t.tile_name(Path(a.frame).name, i)), tile)
    print(f">> {len(t.dishes)} tiles -> {out}")
    print(f">> seed diameter at network input (imgsz {a.imgsz}): "
          f"{t.apparent_seed_px(a.imgsz):.1f} px   [target 15-40]")


if __name__ == "__main__":
    main()
