# GermiCam — autonóm magcsírázás-mérő

Felülnézeti timelapse (RPi HQ cam) egy adott növény magjairól → **megtaláljuk a magokat → megszámoljuk → eldöntjük magonként csírázott-e → Excel** statisztikákkal (T50, MGT, csírázás %). Egy tálcán mindig **egyféle** mag van (a fajta ismert), nincs szükség maize/wheat megkülönböztetésre.

Minden Intel Arc **XPU**-n fut (Lunar Lake iGPU); a `.venv` a projekt gyökerében.

---

## Hogyan működik — két fázis

### 1. fázis: a magdetektor betanítása (crop-onként EGYSZER)

1. **Elő-annotáció** (FastSAM, tanítás nélkül minden magot körberajzol):
   ```
   cd germ
   python pre_annotate_sam.py --frames-dir ../maizeCam --out-dir ../datasets/seed_maize_sam  --map seed_map_maize.json --profile maize --max-frames 30 --device xpu
   ```
   → `datasets/seed_maize_sam/annotations/instances_default.json` (COCO)

2. **Review CVAT-ban** (lokális vagy app.cvat.ai): egy `seed` polygon-label, képek feltöltése,
   Actions → Upload annotations → **COCO 1.0** → a fenti json; javít, majd **Export → COCO 1.0**.
   (Interaktív finomításhoz a CVAT SAM 3 kattintós eszköze jó.) Lásd `CVAT.md`.

3. **Tanítás** (a review-olt adatból train/val bontás → `datasets/maize_seed_yolo`):
   ```
   python train_xpu.py --task segment --data ../datasets/maize_seed_yolo/data.yaml  --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 8
   ```
   → `runs/segment/train-3/weights/best.pt` (a jelenlegi maize detektor).
   Eredmény: box mAP50 0.994, recall 0.994 — minden magot megtalál.

### 2. fázis: egy kísérlet feldolgozása

4. **Számolás** (detektál + követ, csészénként ≤6, a gurulást követi):
   ```
   python run_track.py --weights runs/segment/train-3/weights/best.pt  --frames-dir ../maizeCam --n-dishes 5 --imgsz 640 --conf 0.3 --iou 0.5 --out ../out_maize
   ```
   → `out_maize/per_frame_counts.csv`, `detections.csv`

5. **Csírázás** (csíra-jel növekedés magonként a saját alapvonalhoz képest; torch nélkül):
   ```
   python germinate_from_detections.py     # a paths a fájl tetején állítható
   ```
   → `out_maize/germ_cue_per_frame.csv`, `germ_cue_per_seed.csv`

6. **Excel riport**:
   ```
   python build_report.py --out ../out_maize --crop maize
   ```
   → `out_maize/germination_report_maize_detector.xlsx` (Összesítő + Képkockánként + Magonként + görbe)

---

## Mappák

| Mappa | Tartalom |
|---|---|
| `maizeCam/`, `wheatCam/` | timelapse képek (`GermiCam_ÉÉÉÉ.HH.NN_N.jpg`; idő = 0,5 h × (N−1)) |
| `seedCam/maize&wheat/` | szórt vegyes-mag fotók (csak teszt) |
| `germ/` | minden kód + modellek |
| `germ/runs/segment/train-3/weights/best.pt` | **a maize magdetektor** (YOLO11n-seg) |
| `datasets/maize_seed_yolo/` | detektor tanítóadat (train/val) |
| `datasets/seed_maize_sam/` | a CVAT-ban review-olt maize forrás |
| `out_maize/` | kimenetek (számok, csírázás, Excel, görbék) |
| `cvat/` | lokális CVAT telepítés (Docker) |
| `archive/` | régi, félretett anyagok |

## Kód (`germ/`)

- **Új lánc:** `pre_annotate_sam.py`, `train_xpu.py`, `predict.py`, `run_track.py`, `germinate_from_detections.py`, `build_report.py`
- **Közös/klasszikus alap:** `calibrate.py`, `process.py` (csíra-jel), `analyze.py` (latch), `profiles.py` (crop-paraméterek), `seed_map_*.json`
- **Klasszikus pipeline (opcionális):** `run_all.py`, `report.py`, `make_video.py`
- **Modellek:** `yolo11n-seg.pt` (tanítás alap), `FastSAM-s.pt` (elő-annotáció)

## Környezet / XPU tudnivalók

- A `train_xpu.py` befoltozza az Ultralytics XPU-hibáit: `select_device` elfogadja az `xpu`-t,
  a memória-ellenőrzés XPU-biztos, és **fix `--batch`** kell (az auto-batch `-1` XPU-n elszáll).
- Inferenciánál a **felbontást a betanítotthoz kell illeszteni**: a modell ~0,5-skálás (~1000 px)
  képeken tanult, ezért a teljes felbontású (4056 px) képet is `--imgsz 640`-nel futtasd, hogy a
  magok mérete stimmeljen.
- Segmentálásra a **FastSAM** jobb nálunk, mint a SAM2 (a SAM2 „everything" a sok apró magot kihagyja).
- A **YOLO11** jobb a mi kis adatunkon, mint a YOLO26 (a YOLO26 recallja alacsonyabb).

## Ismert korlát — csírázás jel

A maize gyök halvány, és **világos/reflektív háttéren** (fehér blotter-csík) a jele elveszik →
ott a csírázás alulbecsül (a `train-3` kísérletben a dish3 csak 1/6). A tiszta hátterű csészék
pontosak (6/6, reális T50). **Javítás a felvételnél: szürke/sötét csíráztatópapír** — ekkor
minden csésze megbízható. (Ez képalkotási, nem algoritmus-korlát; három módszerrel is igazolva.)

## Jelenlegi állapot

Maize: **detektálás + számolás kész** (30/30), **csírázás end-to-end** működik
(23/30 = 76,7%, T50=52,5 h, MGT=43 h).

Wheat: a lánc **crop-függetlenre átépítve** (`germ/models.yaml` regiszter), a csírázás-
pontozás a búza képeken **végigmérve és jó** (94/95, T50=35 h, baseline 0,00–0,12 —
a maize világos-hátterű korlátja búzánál nem jelentkezik). Hátra van a búza
**magdetektor betanítása** (CVAT review + tanítás). Részletes recept: **[WHEAT.md](WHEAT.md)**.

Búza ≠ „ugyanaz más profillal": 100 mag/kép, feleakkora magok → a teljes képen a mag
csak ~9 px a hálózat bemenetén. Ezért a búza **csészénkénti tile-okon** detektál
(`tile_dishes: true`), így a mag ~35 px, és `imgsz` maradhat a gyors 640-en.

## Crop-regiszter

Minden crop-specifikus konstans egy helyen: `germ/models.yaml`.
Státusz: `python cropcfg.py`. Új növény = egy új blokk a yaml-ban.
