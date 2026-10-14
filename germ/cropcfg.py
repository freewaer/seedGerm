#!/usr/bin/env python3
"""Loader for models.yaml — the per-crop registry.

Every stage of the chain asks this module for its constants instead of
hardcoding them, so adding a new crop means editing models.yaml only.

    import cropcfg
    cfg = cropcfg.load("wheat")          # dict + attribute access
    cfg.seeds_per_dish                   # 20
    cfg.total_seeds                      # 100
    cropcfg.set_weights("wheat", "runs/segment/train-5/weights/best.pt")
"""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "models.yaml"

REQUIRED = ["profile", "frames_dir", "seed_map", "out_dir", "n_dishes",
            "seeds_per_dish", "scale", "med_r_px", "imgsz", "dt_hours"]

DEFAULTS = dict(weights="", tile_dishes=False, tile_px=1024)


class CropCfg(dict):
    """dict with attribute access, so cfg.imgsz works as well as cfg['imgsz']."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    @property
    def total_seeds(self):
        return int(self["n_dishes"]) * int(self["seeds_per_dish"])

    def path(self, key):
        """Resolve a path-valued key relative to germ/."""
        v = self[key]
        if not v:
            return None
        p = Path(v)
        return p if p.is_absolute() else (HERE / p).resolve()

    def apparent_seed_px(self, image_w_at_scale=None):
        """Seed diameter in px at the network input — the number that must land
        in ~15-40 px for the detector to behave."""
        if self["tile_dishes"]:
            return None  # depends on dish radius; dish_tiles.py reports it
        w = image_w_at_scale or 2028
        return 2 * float(self["med_r_px"]) * (float(self["imgsz"]) / w)


def _read_registry():
    try:
        import yaml
        return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    except ImportError:
        return _mini_yaml(REGISTRY.read_text(encoding="utf-8"))


def _mini_yaml(text):
    """Tiny fallback parser for the 3-level, scalar-only structure of models.yaml,
    so the chain still runs in an environment without PyYAML."""
    root, stack = {}, [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _scalar(val)
    return root


def _scalar(v):
    if v in ("''", '""'):
        return ""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return json.loads(v)
    except Exception:
        return v.strip("'\"")


def crops():
    return sorted(_read_registry().get("crops", {}))


def load(name):
    reg = _read_registry().get("crops", {})
    if name not in reg:
        raise SystemExit(f"unknown crop '{name}' in {REGISTRY.name}. options: {sorted(reg)}")
    cfg = CropCfg({**DEFAULTS, **reg[name]})
    cfg["crop"] = name
    missing = [k for k in REQUIRED if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise SystemExit(f"crop '{name}': missing keys in models.yaml: {missing}")
    return cfg


def set_weights(name, weights):
    """Rewrite the `weights:` line for a crop in place, keeping comments intact."""
    lines = REGISTRY.read_text(encoding="utf-8").splitlines()
    in_crop, done = False, False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{name}:") and (len(line) - len(line.lstrip())) == 2:
            in_crop = True
            continue
        if in_crop:
            if stripped and (len(line) - len(line.lstrip())) <= 2:
                break
            if stripped.startswith("weights:"):
                pad = " " * (len(line) - len(line.lstrip()))
                lines[i] = f"{pad}weights: {weights}"
                done = True
                break
    if not done:
        raise SystemExit(f"could not find weights: for crop '{name}'")
    REGISTRY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">> models.yaml: {name}.weights = {weights}")


def require_weights(cfg):
    w = cfg.path("weights")
    if w is None or not w.exists():
        raise SystemExit(
            f"crop '{cfg['crop']}' has no trained detector yet "
            f"(models.yaml weights: '{cfg['weights']}').\n"
            f"Train one first, then: python -c \"import cropcfg;"
            f"cropcfg.set_weights('{cfg['crop']}','runs/segment/trainN/weights/best.pt')\"")
    return w


if __name__ == "__main__":
    for c in crops():
        cfg = load(c)
        ap = cfg.apparent_seed_px()
        print(f"{c:8s} seeds={cfg.total_seeds:4d} ({cfg.n_dishes}x{cfg.seeds_per_dish})  "
              f"med_r={cfg.med_r_px}  imgsz={cfg.imgsz}  tiles={cfg.tile_dishes}  "
              f"apparent_seed_px={'tile-dependent' if ap is None else round(ap, 1)}  "
              f"weights={cfg.weights or 'NOT TRAINED'}")
