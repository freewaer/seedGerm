#!/usr/bin/env python3
"""Stage 2 - per-frame per-seed germination signal (profile-driven).

Sprout signal per crop is set in profiles.py. The kernel is re-located each frame
(maize kernels roll several radii as the radicle pushes) and the sprout connected
to the CURRENT kernel is measured. Outputs sprout extent and area.
"""
import argparse, csv, re, cv2, json, numpy as np
from pathlib import Path
from calibrate import load, kernel_mask, dish_mask
import profiles

CADENCE_H = 0.5

def pale_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); Hc, S, V = cv2.split(hsv)
    return ((S <= 65) & (V >= 105) & (V <= 240)).astype(np.uint8) * 255

def green_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); Hc, S, V = cv2.split(hsv)
    return ((Hc >= 32) & (Hc <= 95) & (S >= 35) & (V >= 40)).astype(np.uint8) * 255

def seq_index(name):
    m = re.search(r"_(\d+)\.jpg$", name)
    return int(m.group(1)) if m else -1

def measure_seed(seed, kmask, sprout_sig, med_r):
    """Track the (moving) kernel near its fixed position, then measure the sprout
    connected to it. Returns (kx, ky, extent_px, area_px, kernel_found)."""
    x0, y0 = seed["x"], seed["y"]
    win = int(max(9*med_r, 60))
    H, W = kmask.shape
    xa, xb = max(0, x0-win), min(W, x0+win)
    ya, yb = max(0, y0-win), min(H, y0+win)
    ksub = kmask[ya:yb, xa:xb]; ssub = sprout_sig[ya:yb, xa:xb]
    lx, ly = x0-xa, y0-ya
    sbin = ((ssub > 0) & (ksub == 0)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((ksub > 0).astype(np.uint8), 8)
    best, bd = -1, 1e18
    for i in range(1, n):
        cx, cy = cent[i]; dd = (cx-lx)**2 + (cy-ly)**2
        if stats[i, cv2.CC_STAT_AREA] >= (med_r*med_r*0.4) and dd < bd:
            bd, best = dd, i
    kernel_found = int(best > 0)
    if best > 0:
        kbin = (lab == best).astype(np.uint8); sx, sy = cent[best]
    else:
        kbin = np.zeros(ksub.shape, np.uint8); sx, sy = lx, ly
    seed_disk = np.zeros(ksub.shape, np.uint8)
    cv2.circle(seed_disk, (int(round(sx)), int(round(sy))), max(3, int(med_r*0.6)), 1, -1)
    base = ((kbin > 0) | (sbin > 0) | (seed_disk > 0)).astype(np.uint8)
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    nc, labc = cv2.connectedComponents(base, 8)
    lbl = labc[int(round(sy)), int(round(sx))]
    kx_out, ky_out = xa+int(round(sx)), ya+int(round(sy))
    if lbl == 0:
        return kx_out, ky_out, 0.0, 0, kernel_found
    sprout = (labc == lbl) & (sbin > 0)
    area = int(sprout.sum())
    if area == 0:
        return kx_out, ky_out, 0.0, 0, kernel_found
    ys, xs = np.where(sprout)
    d = np.sqrt((xs-sx)**2 + (ys-sy)**2)
    extent = float(max(0.0, d.max() - med_r))
    return kx_out, ky_out, extent, area, kernel_found

def build_signal(bgr, prof, ref_sub):
    pale = pale_mask(bgr)
    if ref_sub is not None:
        pale = cv2.bitwise_and(pale, pale, mask=cv2.bitwise_not(ref_sub))
    sig = pale
    if prof["use_green"]:
        sig = cv2.bitwise_or(sig, green_mask(bgr))
    return sig

def process_frame(path, cfg, med_r, scale, prof, ref_sub, khsv):
    bgr = load(path, scale)
    kmask = kernel_mask(bgr, khsv)
    sig = build_signal(bgr, prof, ref_sub)
    rows = []
    for dish in cfg["dishes"]:
        dm = dish_mask(bgr.shape, dish["cx"], dish["cy"], dish["r"])
        km = cv2.bitwise_and(kmask, kmask, mask=dm)
        ss = cv2.bitwise_and(sig, sig, mask=dm)
        for seed in dish["seeds"]:
            kx, ky, ext, area, found = measure_seed(seed, km, ss, med_r)
            rows.append(dict(seed_id=seed["seed_id"], dish_id=dish["dish_id"],
                             kx=kx, ky=ky, sprout_len_px=round(ext, 2),
                             sprout_area=area, kernel_found=int(found)))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--map", default="seed_map.json")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--profile", default="wheat")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--ref-seq", type=int, default=1)
    args = ap.parse_args()
    prof = profiles.get(args.profile); khsv = prof["kernel_hsv"]
    cfg = json.loads(Path(args.map).read_text())
    med_r = cfg["median_kernel_eq_r_px"]
    frames = sorted([p for p in Path(args.frames_dir).glob("*.jpg")], key=lambda p: seq_index(p.name))
    frames = [p for p in frames if seq_index(p.name) >= 0]
    ref_path = next(p for p in frames if seq_index(p.name) == args.ref_seq)
    ref = load(ref_path, args.scale)
    mode = prof.get("ref_mode", "none"); dil = max(3, int(prof["ref_dilate"]))
    if mode == "pale":
        rm = pale_mask(ref)
    elif mode == "bright":
        V = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)[:, :, 2]
        rm = ((V >= int(prof["ref_bright"])).astype(np.uint8)) * 255
    else:
        rm = np.zeros(ref.shape[:2], np.uint8)
    ref_sub = cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)))
    out = Path(args.out); done = set()
    if out.exists():
        for r in csv.DictReader(out.open()):
            done.add(r["image"])
    fields = ["image","seq","elapsed_h","seed_id","dish_id","kx","ky","sprout_len_px","sprout_area","kernel_found"]
    write_header = not out.exists()
    f = out.open("a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if write_header: w.writeheader()
    sel = frames[args.start:args.end]; n_proc = 0
    for p in sel:
        if p.name in done: continue
        seq = seq_index(p.name); elapsed = round(CADENCE_H*(seq-1), 3)
        for row in process_frame(p, cfg, med_r, args.scale, prof, ref_sub, khsv):
            row.update(image=p.name, seq=seq, elapsed_h=elapsed); w.writerow(row)
        n_proc += 1
    f.close()
    print(f"processed {n_proc} frames, profile={args.profile}, appended to {out}")

if __name__ == "__main__":
    main()
