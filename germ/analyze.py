#!/usr/bin/env python3
"""
Stage 3 - temporal latching -> per-seed germination time.

Germination is scored as SUSTAINED GROWTH of the sprout above each seed's own
baseline (change detection), which cancels static pale background (e.g. the white
blotter strip) and rejects transient glare:

  signal(t) = smoothed(sprout_len / kernel_r) - baseline
  germinated when signal(t) >= DELTA for K consecutive frames; state latches on.

Outputs: per_seed.csv, per_frame.csv
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
import profiles

def latch(df, med_r, delta=1.8, k=4, baseline_n=3, smooth=5, metric='len', abs_floor=2.0):
    df = df.copy()
    if metric == "area":
        df["len_r"] = np.sqrt(df["sprout_area"].clip(lower=0) / np.pi) / med_r
    else:
        df["len_r"] = df["sprout_len_px"] / med_r
    rows_seed, per_seed_series = [], {}
    for sid, g in df.sort_values("seq").groupby("seed_id"):
        g = g.sort_values("seq")
        len_r = g["len_r"].to_numpy(dtype=float)
        seqs = g["seq"].to_numpy()
        elap = g["elapsed_h"].to_numpy(dtype=float)
        # smooth (rolling median) to kill single-frame spikes
        sm = pd.Series(len_r).rolling(smooth, center=True, min_periods=1).median().to_numpy()
        baseline = float(np.median(sm[:baseline_n]))
        signal = sm - baseline
        # find first index where signal>=delta for k consecutive frames
        germ_i = -1
        run = 0
        for i, v in enumerate(signal):
            ok = (v >= delta) and (sm[i] >= abs_floor)
            run = run + 1 if ok else 0
            if run >= k:
                germ_i = i - k + 1
                break
        germinated = germ_i >= 0
        rows_seed.append(dict(
            seed_id=sid, dish_id=g["dish_id"].iloc[0],
            germinated=int(germinated),
            germ_seq=int(seqs[germ_i]) if germinated else "",
            germ_elapsed_h=round(float(elap[germ_i]), 2) if germinated else "",
            baseline_len_r=round(baseline, 2),
            final_len_r=round(float(np.nanmedian(sm[-5:])), 2),
            max_len_r=round(float(np.nanmax(sm)), 2),
        ))
        # per-frame germinated state (latched)
        state = np.zeros(len(seqs), dtype=int)
        if germinated:
            state[germ_i:] = 1
        per_seed_series[sid] = dict(seqs=seqs, state=state, dish=g["dish_id"].iloc[0])
    per_seed = pd.DataFrame(rows_seed)
    return per_seed, per_seed_series

def per_frame_counts(series, seq_index, elapsed_map, dishes):
    rows = []
    for seq in seq_index:
        row = dict(seq=int(seq), elapsed_h=round(float(elapsed_map[seq]), 2))
        tot = 0
        for d in dishes:
            c = 0
            for sid, s in series.items():
                if s["dish"] != d:
                    continue
                idx = np.where(s["seqs"] == seq)[0]
                if len(idx):
                    c += int(s["state"][idx[0]])
            row[f"{d}_germinated"] = c
            tot += c
        row["total_germinated"] = tot
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--map", default="seed_map.json")
    ap.add_argument("--profile", default="")
    ap.add_argument("--metric", default="len", choices=["len", "area"])
    ap.add_argument("--delta", type=float, default=1.8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--abs-floor", type=float, default=2.0)
    ap.add_argument("--out-seed", default="per_seed.csv")
    ap.add_argument("--out-frame", default="per_frame.csv")
    args = ap.parse_args()

    cfg = json.loads(Path(args.map).read_text())
    med_r = cfg["median_kernel_eq_r_px"]
    delta, k, metric = args.delta, args.k, args.metric
    if args.profile:
        prof = profiles.get(args.profile); delta = prof["delta"]; k = prof["k"]; metric = prof["metric"]
        args.abs_floor = prof.get("abs_floor", args.abs_floor)
    dishes = [d["dish_id"] for d in cfg["dishes"]]
    n_per_dish = {d["dish_id"]: d["n_seeds"] for d in cfg["dishes"]}

    df = pd.read_csv(args.features)
    per_seed, series = latch(df, med_r, delta=delta, k=k, metric=metric, abs_floor=args.abs_floor)

    seq_index = sorted(df["seq"].unique())
    elapsed_map = df.groupby("seq")["elapsed_h"].first().to_dict()
    per_frame = per_frame_counts(series, seq_index, elapsed_map, dishes)
    # add percentages
    for d in dishes:
        per_frame[f"{d}_pct"] = (per_frame[f"{d}_germinated"] / n_per_dish[d] * 100).round(1)
    per_frame["total_pct"] = (per_frame["total_germinated"] / sum(n_per_dish.values()) * 100).round(1)

    per_seed.to_csv(args.out_seed, index=False)
    per_frame.to_csv(args.out_frame, index=False)

    g = per_seed["germinated"].sum()
    print(f"seeds={len(per_seed)} germinated={g} ({100*g/len(per_seed):.0f}%)")
    print("final per-dish germinated:",
          per_frame.iloc[-1][[f'{d}_germinated' for d in dishes]].to_dict())
    ok = per_seed[per_seed.germinated==1]
    if len(ok):
        print(f"germination time (h): min={ok.germ_elapsed_h.astype(float).min():.1f} "
              f"median={ok.germ_elapsed_h.astype(float).median():.1f} "
              f"max={ok.germ_elapsed_h.astype(float).max():.1f}")

if __name__ == "__main__":
    main()
