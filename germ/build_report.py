#!/usr/bin/env python3
"""Build the germination Excel report from germinate_from_detections.py output.

  python build_report.py --out ../out_maize --crop maize
Reads <out>/germ_cue_per_frame.csv + germ_cue_per_seed.csv ->
writes <out>/germination_report_<crop>.xlsx (Összesítő / Képkockánként / Magonként + curve).
Needs openpyxl (pip install openpyxl if missing).
"""
import argparse, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, Reference

ap = argparse.ArgumentParser()
ap.add_argument("--crop", default="maize", help="key in models.yaml")
ap.add_argument("--out", default=None, help="output dir (default: from models.yaml)")
a = ap.parse_args()
if not a.out:
    import cropcfg
    a.out = str(cropcfg.load(a.crop).path("out_dir"))
pf = pd.read_csv(f"{a.out}/germ_cue_per_frame.csv")
ps = pd.read_csv(f"{a.out}/germ_cue_per_seed.csv")
dish_cols = [c for c in pf.columns if c.startswith("dish")]
gt = sorted(float(x) for x in ps["germ_h"] if pd.notna(x) and str(x) != "")
gfin = int(ps["germinated"].sum()); N = len(ps)
t50 = gt[int(np.ceil(0.5*len(gt)))-1] if gt else None
mgt = round(float(np.mean(gt)), 2) if gt else None

wb = Workbook(); AR = Font(name="Arial", size=11); ARB = Font(name="Arial", size=11, bold=True)
TITLE = Font(name="Arial", size=14, bold=True); hdr = PatternFill("solid", fgColor="1F4E78")
HW = Font(name="Arial", size=11, bold=True, color="FFFFFF")
thin = Border(*[Side(style="thin", color="D9D9D9")]*4)
def style_header(ws, row, ncol):
    for c in range(1, ncol+1):
        cell = ws.cell(row=row, column=c); cell.fill = hdr; cell.font = HW
        cell.alignment = Alignment(horizontal="center")

s = wb.active; s.title = "Összesítő"
s["A1"] = f"{a.crop.capitalize()} csírázási jelentés — GermiCam (detektor + követés + csíra-jel)"; s["A1"].font = TITLE
rows = [("Magok száma összesen", N), ("Csírázott (végső)", gfin),
        ("Csírázási arány", f"{round(100*gfin/N,1)}%"),
        ("T50 (50%-hoz szükséges idő)", f"{t50} h"), ("MGT (átlagos csírázási idő)", f"{mgt} h")]
r = 3
for k, v in rows:
    s.cell(r, 1, k).font = ARB; s.cell(r, 2, v).font = AR; r += 1
r += 1
s.cell(r, 1, "Csészénkénti bontás").font = TITLE; r += 1
for i, h in enumerate(["Csésze", "Magok", "Csírázott", "Arány", "Baseline (jel/zaj)", "Megjegyzés"], 1):
    s.cell(r, i, h)
style_header(s, r, 6); r += 1
for d in sorted(ps["dish"].unique()):
    sub = ps[ps.dish == d]; ng = int(sub.germinated.sum()); nn = len(sub); base = sub.baseline_len_r.mean()
    note = "jel elveszik (világos háttér)" if base > 5 else "tiszta jel"
    for i, v in enumerate([f"dish{d}", nn, ng, f"{round(100*ng/nn)}%", round(base, 2), note], 1):
        c = s.cell(r, i, v); c.font = AR; c.border = thin
        if base > 5 and i == 6: c.font = Font(name="Arial", size=11, color="C00000", italic=True)
    r += 1
for col, w in {"A": 30, "B": 12, "C": 12, "D": 10, "E": 18, "F": 34}.items():
    s.column_dimensions[col].width = w

pfs = wb.create_sheet("Képkockánként")
cols = ["seq", "time_h"] + dish_cols + ["total_germinated", "total_seeds", "germ_pct"]
for i, h in enumerate(cols, 1): pfs.cell(1, i, h)
style_header(pfs, 1, len(cols))
for j, (_, row) in enumerate(pf.iterrows(), 2):
    for i, c in enumerate(cols, 1): pfs.cell(j, i, row[c]).font = AR
pfs.freeze_panes = "A2"
ch = LineChart(); ch.title = "Csírázás az idő függvényében"; ch.y_axis.title = "%"; ch.x_axis.title = "idő (h)"
ch.height = 9; ch.width = 18
ch.add_data(Reference(pfs, min_col=cols.index("germ_pct")+1, min_row=1, max_row=len(pf)+1), titles_from_data=True)
ch.set_categories(Reference(pfs, min_col=2, min_row=2, max_row=len(pf)+1))
s.add_chart(ch, "H3")

pss = wb.create_sheet("Magonként")
for i, h in enumerate(["Csésze", "Slot", "Csírázott", "Csírázás ideje (h)", "Baseline", "Max csíra (sugár)"], 1):
    pss.cell(1, i, h)
style_header(pss, 1, 6)
for j, (_, row) in enumerate(ps.sort_values(["dish", "slot"]).iterrows(), 2):
    vals = [f"dish{int(row.dish)}", int(row.slot), "igen" if row.germinated else "nem",
            row.germ_h if str(row.germ_h) != "nan" else "—", row.baseline_len_r, row.max_len_r]
    for i, v in enumerate(vals, 1): pss.cell(j, i, v).font = AR
pss.freeze_panes = "A2"
outp = f"{a.out}/germination_report_{a.crop}_detector.xlsx"
wb.save(outp); print("saved", outp)
