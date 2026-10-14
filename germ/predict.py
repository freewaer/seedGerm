#!/usr/bin/env python3
"""Run a trained seed detector on image(s) and report the count. Intel XPU-aware.

  # single wheat frame (does the maize-trained detector find wheat seeds?)
  python predict.py --weights runs/segment/train-3/weights/best.pt \
      --source ../wheatCam/GermiCam_2026.03.20_200.jpg --imgsz 640 --conf 0.25

  # a folder, or your own mixed photo:
  python predict.py --weights runs/segment/train-3/weights/best.pt --source ../some_mixed.jpg
Annotated images are saved under runs/predict/, and the per-image count is printed.
"""
import argparse, torch
from pathlib import Path


def patch_xpu():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        import ultralytics.utils.torch_utils as tu
        _orig = tu.select_device
        def sel(device="", *a, **k):
            if "xpu" in str(device).lower():
                return torch.device("xpu:0")
            return _orig(device, *a, **k)
        tu.select_device = sel
        try:
            import ultralytics.engine.predictor as pr
            pr.select_device = sel
        except Exception:
            pass
        return "xpu"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU; lower = merge overlapping duplicate boxes")
    ap.add_argument("--agnostic", action="store_true", help="class-agnostic NMS (single class -> recommended)")
    ap.add_argument("--device", default="xpu", choices=["xpu", "cpu"])
    ap.add_argument("--out", default="runs/predict")
    a = ap.parse_args()
    dev = patch_xpu() if a.device == "xpu" else "cpu"
    print(f">> device: {dev}")
    from ultralytics import YOLO
    m = YOLO(a.weights)
    results = m.predict(source=a.source, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                        agnostic_nms=a.agnostic, device=dev,
                        save=True, project=a.out, name="test", exist_ok=True,
                        line_width=2, retina_masks=True, verbose=False)
    total = 0
    for r in results:
        n = 0 if r.boxes is None else len(r.boxes)
        total += n
        print(f"  {Path(r.path).name}: {n} seeds")
    print(f"\nTOTAL seeds across {len(results)} image(s): {total}")
    print(f"Annotated output -> {a.out}/test/")


if __name__ == "__main__":
    main()
