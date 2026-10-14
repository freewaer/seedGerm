# Búza detektor — lépésről lépésre

Cél: a maize-szal azonos pontosságú **magszámlálás + csírázás-pontozás búzára**.
Minden GPU-s lépést Te futtatsz az Arc XPU-s gépen, a `germ/` mappából, aktív `.venv`-vel.

> **PowerShell:** minden parancs itt egysoros. A `\` NEM sortörés PowerShellben
> (az a backtick `` ` ``), ezért ha tördelsz, backtick-et használj — vagy hagyd egy sorban.

---

## Miért nem elég a maize recept változtatás nélkül

| | maize (kész) | wheat |
|---|---|---|
| mag / képkocka | 30 (6 csészénként) | **100 (20 csészénként)** |
| mag sugár (0,5 skálán) | 26,5 px | **14,3 px** |
| mag a hálózat bemenetén, teljes képen `imgsz 640` | ~17 px | **~9 px → a detektálási küszöb alatt** |
| csírázási jel | halvány gyök → jelkorlátos | **zöldül → tiszta jel** |

A 9 px-es magot a YOLO nem találja meg megbízhatóan. Két út volt: `imgsz`-t 1280-ra
tolni (Arc iGPU-n 4–5 perc/epoch → 6+ óra tanítás), vagy **csészénkénti tile**.
A tile nyert:

- minden csészét külön 1024 px-es képre nagyítunk → a mag **35 px** a hálózat bemenetén,
  vagyis a bevált maize tartomány **kétszerese**, miközben `imgsz` marad 640 (gyors);
- egy képen csak 20 mag van, nem 100 → a CVAT-review olvasható, és a tracker
  20 magot különböztet meg egy csészén belül, nem 100-at az egész képen;
- a csésze identitása a tile-ból adódik → nincs szükség k-means csésze-klaszterezésre.

---

## Amit már megjavítottam a te adataidon

1. **`seed_map_wheat.json` 5 hibás bejegyzése** (`D1_03`, `D2_04`, `D3_03`, `D4_07`,
   `D5_09`) az egész képet írta le magként (eq_r ≈ 987 px). Törölve.
   Az eredeti `seed_map_wheat.raw.json` néven megmaradt.
2. **A csészekörök rosszak voltak.** A `calibrate.py` Hough-köre D2-nél ~50%-kal
   túl nagy sugarat, D5-nél a képből kilógó kört adott. Ezek a körök vágták le a
   magokat is — ezért lett csészénként 19 a 20-ból. Az új körök a **review-olt
   magpozíciók minimális befoglaló köréből** származnak (× 1,30 margó), ami itt
   sokkal megbízhatóbb, mint a peremkeresés. Ellenőrizve, minden mag benne van.
3. **A „csészék elcsúsznak" riadó téves volt.** A per-csésze Hough ~100 px
   sodródást mutatott D5-nél — de ez a Hough tévedése volt a rossz sugarakkal, nem
   valódi mozgás. Négy statikus képrészletre (állványrúd, tálca, oszlop) mért
   fáziskorreláció szerint a rig **0,5 px-en belül merev mind a 200 képkockán**.
   A `dish_track.py` most ezt a helyes mérést végzi, és ellenőrzésként érdemes futtatni.

---

## A lánc

```
dish_track.py          rig merevség-ellenőrzés  (1 perc, ajánlott)
pre_annotate_color.py  színalapú elő-annotáció csésze-tile-okra  (CPU)
   ↓  CVAT review (a te munkád)
build_dataset.py       COCO → YOLO-seg train/val, miscount-ellenőrzéssel
train_xpu.py           detektor tanítás                          [XPU]
run_track.py           detektál + követ + számol                 [XPU]
germinate_from_detections.py   csírázás-pontozás (torch nélkül)
build_report.py        Excel riport
```

Minden lépés a `germ/models.yaml`-ból veszi a paramétereket — nincs több beégetett
konstans. `python cropcfg.py` kiírja az aktuális regisztert.

---

## 1. Rig-ellenőrzés (1 perc)

```
cd germ
python dish_track.py --crop wheat --plot ../out_wheat/rig_drift.png
```

Elvárt: `RIG IS RIGID — max rigid shift < 3px`. Ha nagyobb, a mentett eltolásokat a
tile-ok automatikusan követik, de nézd meg a felvételt (megbökött kamera fókuszt is
változtathat, azt eltolás nem javítja).

## 2. Elő-annotáció — színalapú, GPU nélkül

```
python pre_annotate_color.py --crop wheat --out-dir ../datasets/seed_wheat_color --max-frames 30 --seq-max 104
```

**Nem FastSAM-mel.** Az első futás megmutatta, miért: a tile-oknak csak **5%-án**
adott pontosan 20 magot, átlag 10,2 mag/tile, a futás végére 1,4. Az ok **nem
takarás** — 44 h-nál a magok jól látszanak —, hanem hogy a FastSAM az egész
**csíranövényt** szegmentálja: a szem és a koleoptile egy összefüggő objektum.

Búzánál a **szín** élesen elválaszt (barna szem / zöld hajtás / szürke papír), és a
`profiles.py` kernel-HSV maszkja minden magot lefed minden időpontban. A megmaradt
feladat az érintkező szemek szétvágása.

| ugyanazon a 150 tile-on | FastSAM | színalapú |
|---|---|---|
| pontosan 20 mag | 7/150 (5%) | **143/150 (95%)** |
| átlag mag/tile | 10,2 | **19,9** |
| ismert magok megtalálva (seq 1) | — | **91/95 (96%)**, medián hiba 2,5 px (magsugár 29 px) |

### Miért `--seq-max 104`

A darabszám-vezérelt szétvágás az **elvetett** magszámot (20) használja. Ez jogos
előzetes tudás, de önmagában körkörös lenne. Ezért a script egy **független**
becslést is számol (`core_count`): a szem vastag, a koleoptile vékony, így egy
~0,55 magsugarú nyitás letörli a függelékeket, és magonként egy tömör „mag-mag"
marad — ez **nem tud** a 20-ról.

| ablak | független becslés átlaga | egyezik 20-szal |
|---|---|---|
| seq 1–104 (t ≤ 52 h) | **20,3** | **83%** |
| seq 1–200 (t ≤ 100 h) | 31,5 | 48% |

Vagyis **t ≤ 52 h-ig a 20-as szám mért, nem feltételezett**; utána a barna
gyökérszőnyeg szétesik darabokra, és a címke találgatás lenne. 52 h már jóval T50
(35 h) után van, tehát a detektor a lényeges szakaszt megkapja — a késői képkockákon
a csírázás úgyis már bereteszelt, a számlálást pedig ott amúgy is a takarás korlátozza.
**Kitalált címkékre tanítani pontosan az a hiba volt, ami az első YOLO-kísérletet megölte.**

A kiírás végén:

```
exactly 20: 94/100 images (94%)
forced splits: 256 instances over 75 images
INDEPENDENT check: mean 20.3, agrees with 20 on 83/100 images (83%)
```

A `CHECK`-kel jelölt tile-okkal kezdd a review-t — ott a geometria következtetett.
Hangolás: `--min-area-factor 0.18`, `--close-px 7`.

## 3. CVAT review — ~3000 poligon

Task létrehozás: **egy** label, neve pontosan `seed`, típusa **polygon**.
Képek: `datasets/seed_wheat_color/images/` (150 db 1024 px-es tile).
`Actions → Upload annotations → COCO 1.0 →`
`datasets/seed_wheat_color/annotations/instances_default.json`.

Review közben:

- **csészénként pontosan 20 magnak kell lennie** — ez a legfontosabb ellenőrzés.
  A régi YOLO-kísérletet pont a rosszul számolt címkék ölték meg.
- a tile sarkaiban látszó **szomszéd csésze magjait NE annotáld** — a kör azokat
  amúgy is kiszűri, de a felesleges címke rontja a tanítást;
- a késői képkockákon a hajtás **eltakarja** a magot: a magot annotáld, ne a hajtást;
- a maszk éle lehet laza, a **darabszám** és a **középpont** a fontos.

Export: **COCO 1.0** → `datasets/wheat_seed_review.zip`.

## 4. Tanítóadat építése (miscount-ellenőrzéssel)

```
python build_dataset.py --crop wheat --coco ../datasets/wheat_seed_review.zip --images ../datasets/seed_wheat_color/images --out ../datasets/wheat_seed_yolo --val-frac 0.2
```

Kiírja, hány tile-on nincs pontosan 20 mag — **ha ilyet jelez, menj vissza CVAT-ba**.
A train/val bontás idő-szórt, és egy képkocka 5 tile-ja mindig azonos oldalra kerül
(különben a val a train majdnem-azonos képeit kapja, és a metrika hazudik).

## 5. Tanítás

```
python train_xpu.py --task segment --data ../datasets/wheat_seed_yolo/data.yaml --model yolo11n-seg.pt --epochs 80 --imgsz 640 --batch 8
```

Kb. a maize-zal azonos idő (~30 perc), mert a tile miatt `imgsz` maradt 640, csak
5× több kép van — számolj inkább 1–1,5 órával. Elvárt: box mAP50 > 0,98, recall > 0,98.

Utána írd be a regiszterbe:

```
python -c "import cropcfg; cropcfg.set_weights('wheat','runs/segment/trainN/weights/best.pt')"
```

## 6. Számlálás

```
python run_track.py --crop wheat
```

Elvárt kiírás: `exactly 100 seeds in .../200 frames`. A késői képkockákon a hajtás
eltakarhat magokat (a maize-nál is így volt) — ha az alulszámlálás a **futás végén**
sűrűsödik, az normális; ha az **elején**, csökkentsd a `--conf`-ot.

## 7. Csírázás + riport

```
python germinate_from_detections.py --crop wheat
python build_report.py --crop wheat
```

→ `out_wheat/germination_report_wheat_detector.xlsx`

Gyors előnézet a teljes futás előtt: `--stride 4`.

---

## Amit a csírázási fokozatról már tudunk — mérve, a te képeiden

A 3. fázist a **review-olt magpozíciókkal** végigfuttattam mind a 200 képkockán
(vagyis mindent teszteltem a detektor kivételével, ami még nincs betanítva):

```
slots=95   germinated_final=94 (98,9%)   T50=35,0 h   MGT=37,3 h
  dish0: 19/19   baseline 0,00
  dish1: 19/19   baseline 0,05
  dish2: 18/19   baseline 0,11
  dish3: 19/19   baseline 0,12
  dish4: 19/19   baseline 0,03
```

A lényeg: a **baseline mindenhol 0,00–0,12**, szemben a maize dish3 értékével (10,7).
Vagyis a maize nagy korlátja — halvány gyök világos alátéten — **búzánál nem
jelentkezik**, mert a búza zöldül, és a `ref_mode='pale'` levonás után a zöld jel
tisztán marad. **A fehér blotter-csíkos D1 csészénél is 19/19.**

Tehát búzánál a csírázási időzítés (T50, MGT) is használható, nem csak a darabszám —
ez a maize-nál nem volt igaz. A 95 slot a 100-ból abból jön, hogy a régi klasszikus
magtérkép csészénként 1 magot kihagyott; a betanított detektor ezt megoldja, és akkor
100/100 lesz.

---

## Változott / új fájlok a `germ/`-ben

| fájl | mi |
|---|---|
| `models.yaml` | **új** — crop-regiszter: súlyok, profil, csészék, mag/csésze, tile, imgsz |
| `cropcfg.py` | **új** — regiszter-betöltő (`python cropcfg.py` = státusz) |
| `dish_tiles.py` | **új** — csészénkénti tile + koordináta-visszaképezés |
| `dish_track.py` | **új** — rig merevség-ellenőrzés statikus képrészletekből |
| `build_dataset.py` | **új** — CVAT COCO → YOLO-seg, miscount-ellenőrzéssel |
| `pre_annotate_sam.py` | tile mód, `--crop`, darabszám-QC riport |
| `run_track.py` | tile mód, `--crop`, csészén kívüli detekciók kiszűrése, count-QC |
| `germinate_from_detections.py` | teljesen paraméterezve (`--crop`), `--stride`, gyors dekódolás |
| `build_report.py` | `--crop` elég, az útvonal a regiszterből jön |
| `calibrate.py` | `load(..., fast=True)` libjpeg csökkentett dekódolás (2× gyorsabb); scipy import lustává téve |
| `seed_map_wheat.json` | hibás bejegyzések törölve, csészekörök újrailleszve |

A maize út változatlanul működik (`--crop maize`), a régi kézi kapcsolók továbbra is
felülírják a regisztert.
