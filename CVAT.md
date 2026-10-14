# CVAT — indítás, belépés, hibaelhárítás

Ez a CVAT annotáló futtatása ehhez a projekthez. A CVAT 18 Docker konténerben fut,
WSL2 (Ubuntu-24.04) alatt. Windows-on natívan nem tud futni — nincs Docker nélküli mód.

---

## Indítás

Kattints a projekt gyökerében:

```
start-cvat.cmd
```

A script kiírja, mikor áll készen (a szerver **60–90 másodperc** alatt indul el, ez normális).
Amikor ezt látod:

```
CVAT is READY -> http://localhost:8080
```

akkor nyisd meg a böngészőben: **http://localhost:8080**

## Belépés

| | |
|---|---|
| **Felhasználó** | `admin` |
| **Jelszó** | `SeedGerm2026!` |

> **Változtasd meg a jelszót!** Ez csak egy ideiglenes jelszó az induláshoz.
> Belépés után: jobb felül a felhasználónév → **Change password**.

## Leállítás

```
stop-cvat.cmd
```

Leállítja a konténereket és felszabadítja a ~4 GB RAM-ot.
**Az annotációk nem vesznek el** — Docker volume-okban maradnak, újraindításnál minden ott lesz.

---

## FONTOS: miért kell a "keepalive"?

A `start-cvat.cmd` indít egy rejtett folyamatot (`wsl … sleep infinity`). **Ez nem elhagyható.**

A WSL leállítja a virtuális gépet, amint egyetlen Windows-folyamat sem kapcsolódik hozzá.
A systemd és a Docker **nem számít** ilyennek. Ha a VM leáll:

- a Docker is leáll,
- a CVAT is leáll,
- és a `localhost:8080` **elérhetetlen** lesz.

Ezért **ne** indítsd a CVAT-ot kézzel `docker compose up`-pal keepalive nélkül, mert pár
másodperc múlva magától „eltűnik”. Mindig a `start-cvat.cmd`-t használd.

---

## Hibaelhárítás

### „A localhost nem érhető el”

Szinte mindig ez a kettő:

1. **Még indul.** A szervernek 60–90 mp kell. Várj, és frissíts.
2. **Leállt a WSL VM.** Ellenőrzés PowerShellben:

   ```powershell
   wsl --list --running
   ```

   Ha azt írja, hogy *„Nincsenek futó disztribúciók”*, akkor a VM leállt →
   futtasd újra a `start-cvat.cmd`-t.

### Állapot ellenőrzése

```powershell
wsl -d Ubuntu-24.04 -u root -e bash -lc "cd /mnt/d/Develope/seedGerm/cvat && docker compose ps"
```

Mind a 18 konténer `running` kell legyen.

### Naplók

```powershell
wsl -d Ubuntu-24.04 -u root -e bash -lc "cd /mnt/d/Develope/seedGerm/cvat && docker compose logs --tail=50 cvat_server"
```

### Ha `502 Bad Gateway` jön a böngészőben

A felület betölt, de a háttérszerver még nem válaszol → **még indul**. Várj 1–2 percet.

---

## Az előcímkék betöltése (annotation_maize)

20 kép, kész COCO poligon előcímkékkel: `datasets/annotation_maize/`

1. **Create new task**
2. Címkék — pontosan így, **polygon** típussal:
   - `ungerminated`
   - `germinated`
3. Töltsd fel a képeket: `datasets/annotation_maize/images/` (20 db)
4. A task megnyitása után: **Actions → Upload annotations → COCO 1.0**
   → válaszd: `datasets/annotation_maize/annotations/instances_default.json`
5. Javítsd az előcímkéket (automatikusan generáltak).
   Sorrendhez: `datasets/annotation_maize/review_priority.csv`
6. Export a tanításhoz: **Actions → Export dataset → COCO 1.0**

---

## Mi lett beállítva (háttér-infó)

| Beállítás | Hol | Miért |
|---|---|---|
| CVAT verzió `v2.49.0` | [cvat/.env](cvat/.env) | fix release, hogy ne a mozgó `:dev` image jöjjön |
| `DOCKER_MIN_API_VERSION=1.24` | WSL: `/etc/systemd/system/docker.service.d/min-api.conf` | a Docker 29 kidobta a régi API-t, amit a Traefik használ — enélkül minden kérés 404 |
| `memory=10GB`, `vmIdleTimeout=-1` | `C:\Users\zolta\.wslconfig` | a WSL alapból a RAM felét vinné |
| WSL lemez a `D:\WSL\` alatt | — | a C:-n csak ~29 GB volt szabad, a Docker image-ek (~5.8 GB) megtöltötték volna |

- **Adatok helye:** Docker volume-ok (a WSL lemezén, `D:\WSL\`), nem a projektmappában.
- **Erőforrás:** kb. 4 GB RAM, amíg fut. Ezért nincs automatikus indítás bootkor —
  csak akkor fut, amikor te elindítod.
