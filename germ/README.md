# GermiCam germination pipeline (annotation-free, multi-crop)

Counts germinated seeds per petri dish across a fixed-camera timelapse and exports
germination statistics — **no manual CVAT tagging, no model training.** Same rig,
same scripts for different crops; only a small per-crop **profile** changes.

## One command per experiment
```bash
python run_all.py --frames-dir ../wheatCam --profile wheat --prefix wheat
python run_all.py --frames-dir ../maizeCam --profile maize --prefix maize
```
Outputs per prefix: `*_seed_map.json`, `*_features.csv`, `*_per_seed.csv`,
`*_per_frame.csv`, `germination_report_<prefix>.xlsx`, `germination_timelapse_<prefix>.mp4`.

## Results
**Wheat** (5×20 = 100 seeds): **95/100** detected, matches visual. Clean timing
(0 germinated < 4.5 h; T50 ≈ 45 h).

**Maize** (5×6 = 30 seeds): **26/30** detected (D3 6/6, D4 6/6, D5 6/6, D2 5/6,
D1 3/6). The 4 misses are the 3 **white-blotter-strip seeds in D1** (white roots on
white paper — genuine low contrast) plus 1 borderline seed. NOTE: maize germination
*timing* (MGT/T50) still has a few early false triggers from wet-paper glare — treat
per-dish timing as provisional; the germinated **count** is the reliable output.

Calibration fix (helps every crop): `split_kernels` used to seed the watershed from
a global distance-transform max, which dropped the smallest kernel in each dish and
let the centre ruler be captured as a giant phantom "seed". It now works per
connected-component and only splits genuinely oversized blobs — all 30 maize kernels
map correctly. Re-run wheat to pick this up.

## Why crops differ (captured in profiles.py)
- **Wheat** seedlings green up → the green cue carries detection; the white blotter
  strip is static and removed by whole-pale reference differencing.
- **Maize** seedlings stay **pale/white** (radicle + coleoptile, little green) and the
  big kernels **roll several radii** as the radicle pushes. So: rely on pale roots,
  track the kernel each frame, and subtract only the static *near-white* strip
  (not all pale) so grey-paper roots still register.

## Pipeline
1. `calibrate.py` — detect 5 dishes once + kernels (watershed splits touching) → seed map.
2. `process.py` — per frame: track each kernel, measure sprout (pale ± green) connected to it.
3. `analyze.py` — per-seed baseline + sustained-growth latch (+ absolute floor) → germination time.
4. `report.py` — Excel: Summary (GP, MGT, rate, T50), Per_Seed, Per_Frame, cumulative curve.
5. `make_video.py` — annotated QC timelapse (green=germinated, red=not).

Deps: `opencv-python numpy pandas scipy scikit-image openpyxl`.

## Tuning a new crop
Add an entry to `profiles.py` (kernel colour, green on/off, ref mode, thresholds).
Biggest accuracy lever remains imaging: avoid the white blotter strip (use grey
germination paper) and lock camera exposure/white-balance so thresholds stay stable.

## Known limitations / next steps
- White-strip dish (D1) is the hard case for white-rooted crops — fix at capture
  (grey paper) or add texture/edge-change detection there.
- Maize early-glare false triggers: raise `abs_floor`/`k`, or add reference-based
  glare masking, to clean up MGT/T50.
- Lock the 5 dish ROIs to a per-rig config to exclude the ruler/overlap phantom.

## Annotation mode — bootstrap a CNN training set (CVAT)

`export_cvat.py` turns the pipeline output into **instance-segmentation pre-labels**
(2 classes: `germinated` / `ungerminated`) so you *review/correct* in CVAT instead of
drawing from scratch — typically several × faster per image.

```bash
python export_cvat.py --frames-dir ../maizeCam --map seed_map_maize.json \
    --features maize_features.csv --per-seed maize_per_seed.csv \
    --profile maize --out-dir ../datasets/annotation_maize --max-frames 24 --scale 0.5
```
Produces `annotation_<crop>/`: `images/`, `annotations/instances_default.json` (COCO 1.0),
`review_priority.csv` (uncertain calls first), `README_CVAT.txt`. It selects informative
frames — every seed's germination-transition frame (the hardest cases) plus an early→late
spread — so both classes are well represented.

**In CVAT:** create a task with polygon labels `ungerminated` + `germinated`, upload
`images/`, then Actions → Upload annotations → **COCO 1.0** → the JSON. Correct the
polygons (start with `review_priority.csv`), then export **COCO 1.0** for training.

**Iterative loop (human-in-the-loop):** label one batch → train the CNN (e.g. YOLO-seg)
→ run the CNN to pre-label the *next* batch → correct → retrain. Each round the model
gets better, so corrections shrink. The classical pipeline seeds round 1 for free.

Notes: use `--scale 1.0` for full-resolution training images; compact seeds get clean
masks, long tangled roots need the most correction (expected for auto pre-labels).

### Fits your existing YOLO toolchain
`export_cvat.py` now writes **both** formats into `annotation_<crop>/`:
- `annotations/instances_default.json` (COCO 1.0) → for CVAT correction.
- `images/` + `labels/*.txt` + `data.yaml` + `classes.txt` (**Ultralytics YOLO-seg**) →
  trains directly, e.g. `yolo segment train data=annotation_maize/data.yaml model=yolo26s-seg.pt`.

Note vs the old `dataset/`: that used a single class `seed` (via `annotation_tool.py`, which
can only *export* — it can't load pre-labels, so correct in CVAT). This exporter uses the two
germination classes you chose — `0 ungerminated`, `1 germinated`. After CVAT correction, export
as COCO/YOLO-seg and point `data.yaml` at it (fix `val:` to a held-out split first).

## Fastest path — click-to-fix review (no drawing, no CVAT)

Since counting only needs germinated/ungerminated, skip polygons entirely:

```bash
python export_review.py --frames-dir ../maizeCam --map seed_map_maize.json \
    --features maize_features.csv --per-seed maize_per_seed.csv \
    --profile maize --out-dir ../datasets/review_maize --max-frames 24
# open review_maize/review.html in a browser, click the WRONG tiles, press Export
python apply_review.py --review-dir ../datasets/review_maize --out-dir ../datasets/cls_dataset_maize
```
- `export_review.py` crops every tracked seed, pre-labels it, and builds a single
  self-contained `review.html` grid (green=germinated, red=ungerminated; yellow caption =
  pipeline unsure, check first; a marker shows which seed to judge). You only click the
  few wrong ones, then **Export → corrected_labels.csv**. No per-seed drawing.
- `apply_review.py` turns that CSV + the saved clean crops into an ImageFolder dataset
  (`train/val` × `germinated/ungerminated`), ready for `yolo classify train ... model=yolo11n-cls.pt`.
- Training crops are saved WITHOUT the center marker so the CNN can't cheat on it.
- Inference = run the tiny classifier on each seed's crop (positions from `seed_map`), sum
  the germinated → the same count, now from a learned model. Retrain-on-corrections loop as above.

## Generalize the classifier — add loose dry-seed photos as `ungerminated`

`add_seed_photos.py` turns folders of loose dry-seed photos (any background/lighting,
e.g. `seedCam/wheat`, `seedCam/maize`) into extra `ungerminated` crops and merges them
with the reviewed rig dataset, so the classifier stops keying on the rig's exact look.

```bash
python add_seed_photos.py --photos ../seedCam/wheat ../seedCam/maize \
    --base ../datasets/cls_maize --out ../datasets/cls_maize_plus --max-per-image 3 --max-total 100
python train_classifier.py --data ../datasets/cls_maize_plus --model yolo11n-cls.pt --epochs 50
```
- Seeds auto-detected by background-difference; sampled per photo; split GROUP-AWARE by
  source photo (a photo's crops never straddle train/val).
- **Cap the count** (`--max-total`) so negatives don't swamp positives; the script warns
  if train ungerminated ≫ germinated. Keep Ultralytics augmentation on (the printed
  train command adds hsv/flip) to fight a "weird background ⇒ ungerminated" shortcut.

Honest limits: these are all one class (dry = ungerminated) from a different domain, so
they teach background-invariance only on the *negative* side. For true off-rig
robustness you'll eventually want *germinated* seedlings on varied backgrounds too.
These dry-seed photos are actually better suited to a separate wheat-vs-maize seed
**detector** (localize + species) — worth building if you want to drop the fixed seed_map.

## Two-stage, rig-independent counter (detect seed → has white strings?)

The deployable design: a learned **seed detector** finds every seed on any background
(no dish geometry / seed_map), then the **germination classifier** decides, per detected
seed, whether it has white strings attached. "White strings attached" is judged by the
learned model, NOT a hand threshold (a single-image pale-tissue rule scores only
AUC~0.5-0.68 because root-on-pale-paper is low contrast).

Stage A — seed detector dataset from loose-seed photos (auto-boxed, species from folder):
```bash
python export_seed_detect.py --wheat ../seedCam/wheat --maize ../seedCam/maize \
    --test "../seedCam/maize&wheat" --out ../datasets/seed_detect --imgsz 1280
yolo detect train data=../datasets/seed_detect/data.yaml model=yolo11n.pt epochs=80 imgsz=1280
```
- ~4000 auto-boxes over wheat+maize; `qc/` overlays to spot-check; mixed folder = held-out
  test (images only — per-seed species isn't known from a mixed folder).
- Weak labels (~85-90% recall) — YOLO tolerates it; drop obviously-bad images if needed.
- NOTE: trained on DRY seeds; on the germinating rig the kernel stays visible but gets
  partly covered — add a few rig frames to Stage A training if recall drops there.

Stage B — germination classifier: the reviewed-crop classifier from the sections above.

Run the full counter on any image:
```bash
python count_two_stage.py --det seed_detect/runs/detect/train/weights/best.pt \
    --cls runs/classify/train/weights/best.pt --image any_photo.jpg --out counted.jpg
```
Outputs total seeds, germinated count, germination %, and a per-species breakdown.

## Seed detector v2 — includes germinated rig seeds (fixes the two-stage rig failure)

The first seed detector only saw dry loose seeds, so on the rig it missed root-covered
kernels (found ~9/30). `export_seed_dataset.py` builds a ONE-CLASS `seed` detection set
that also boxes seeds on the RIG frames — using the fixed `seed_map`, which knows every
seed position whether germinated or not — so the detector finally learns germinated seeds.

```bash
python export_seed_dataset.py \
  --rig maize:../maizeCam:seed_map_maize.json:maize_features.csv \
  --rig wheat:../wheatCam:seed_map_wheat.json:wheat_features.csv \
  --photos ../seedCam/wheat ../seedCam/maize \
  --out ../datasets/seed_det_all --rig-frames 15 --imgsz 1280
yolo detect train data=../datasets/seed_det_all/data.yaml model=yolo11n.pt epochs=80 imgsz=1280 device=cpu
```
- Boxes the KERNEL (persistent object), incl. root-covered ones → detector recall on the
  rig jumps. `qc/` overlays let you spot-check; drop bad frames if needed.
- Then the two-stage counter works on the rig: seed detector finds all kernels →
  germination classifier (Stage B, `cls_maize_plus3`) judges each. `count_two_stage.py`
  now crops a wide context (`--context 3`) so the classifier sees the roots, not just the kernel.

Architecture (your Idea 1): **`seed` detector (one class, grow it with more crops/frames)
+ germination classifier (germinated/ungerminated).** Species can be a later add-on.
