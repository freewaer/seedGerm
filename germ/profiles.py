#!/usr/bin/env python3
"""Per-crop germination profiles. One place to tune each experiment.

Knobs that change per crop:
  kernel_hsv : (H_lo, H_hi, S_min, V_min) for the orange/yellow kernel colour
  use_green  : count green shoot tissue (wheat greens up; maize stays pale)
  ref_mode   : cancel static background so it is not read as sprout —
               'pale'   subtract the reference frame's whole pale mask (wheat),
               'bright' subtract only static near-white pixels V>=ref_bright (maize:
                        keeps white roots on grey paper, removes the blotter strip),
               'none'   no subtraction.
  ref_bright : V threshold for ref_mode='bright'
  ref_dilate : dilation (px) of the reference mask (drift tolerance)
  metric     : 'len' (sprout extent) or 'area' (sprout area)
  delta, k   : germination = growth >= delta (kernel-radii) for k consecutive frames
  abs_floor  : sprout must also reach this absolute length (kernel-radii) — rejects
               early glare that merely rises above baseline
"""
PROFILES = {
    "wheat": dict(kernel_hsv=(3, 30, 70, 45), use_green=True, ref_mode="pale",
                  ref_bright=235, ref_dilate=6, metric="len", delta=1.5, k=4,
                  abs_floor=1.5),
    "maize": dict(kernel_hsv=(3, 32, 60, 45), use_green=True, ref_mode="bright",
                  ref_bright=235, ref_dilate=8, metric="len", delta=1.0, k=6,
                  abs_floor=2.5),
}

def get(name):
    if name not in PROFILES:
        raise SystemExit(f"unknown profile '{name}'. options: {list(PROFILES)}")
    return PROFILES[name]
