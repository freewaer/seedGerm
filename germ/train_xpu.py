#!/usr/bin/env python3
"""Train on an Intel Arc (XPU) iGPU by patching Ultralytics' device + memory checks.

Two Ultralytics assumptions break on Intel XPU, both patched here:
  1. select_device() rejects 'xpu' -> we return torch.device('xpu:0').
  2. _get_memory()/_clear_memory() call torch.cuda.* unconditionally (crash:
     "Torch not compiled with CUDA enabled") -> we make them XPU-safe.
AMP is off (Intel XPU AMP is flaky in Ultralytics).

  python train_xpu.py --task segment  --data ../datasets/maize_seed_yolo/data.yaml \
      --model yolo11n-seg.pt --epochs 80 --imgsz 640
  python train_xpu.py --task detect   --data ../datasets/seed_detect/data.yaml \
      --model yolo11n.pt --epochs 80 --imgsz 1280
"""
import argparse, torch


def _install_xpu_patch():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("No XPU available. Check: python -c \"import torch;print(torch.xpu.is_available())\"")
    import ultralytics.utils.torch_utils as tu
    _orig = tu.select_device
    def sel(device="", *a, **k):
        if "xpu" in str(device).lower():
            print(">> [patch] routing to Intel XPU:", torch.xpu.get_device_name(0))
            return torch.device("xpu:0")
        return _orig(device, *a, **k)
    tu.select_device = sel
    try:
        import ultralytics.engine.trainer as tr
        tr.select_device = sel
    except Exception:
        pass


def _install_mem_patch():
    """Make BaseTrainer memory helpers XPU-safe (they hardcode torch.cuda)."""
    import gc
    from ultralytics.engine.trainer import BaseTrainer
    _orig_get = BaseTrainer._get_memory
    _orig_clear = BaseTrainer._clear_memory

    def _get_memory(self, fraction=False):
        dev = getattr(self, "device", None)
        if dev is not None and getattr(dev, "type", "") == "xpu":
            try:
                reserved = torch.xpu.memory_reserved(dev)
                if fraction:
                    total = torch.xpu.get_device_properties(dev).total_memory
                    return reserved / total if total else 0.0
                return reserved / 1e9
            except Exception:
                return 0.0
        return _orig_get(self, fraction)

    def _clear_memory(self, threshold=None):
        dev = getattr(self, "device", None)
        if dev is not None and getattr(dev, "type", "") == "xpu":
            gc.collect()
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
            return
        return _orig_clear(self, threshold)

    BaseTrainer._get_memory = _get_memory
    BaseTrainer._clear_memory = _clear_memory
    print(">> [patch] XPU-safe memory helpers installed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["detect", "classify", "segment"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)       # fixed: XPU can't use auto-batch (-1)
    ap.add_argument("--cache", default="ram")             # ram/disk/'' - tiny dataset -> ram
    ap.add_argument("--workers", type=int, default=0)     # 0 avoids Windows dataloader issues
    a = ap.parse_args()
    if a.batch <= 0:
        a.batch = 8
        print(">> [patch] auto-batch (-1) unsupported on XPU; using --batch 8")
    _install_xpu_patch()
    _install_mem_patch()
    from ultralytics import YOLO
    m = YOLO(a.model)
    m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            cache=(a.cache or False), device="xpu", amp=False, workers=a.workers)


if __name__ == "__main__":
    main()
