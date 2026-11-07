"""M5 sensitivity checks on the map x batch result (run after M5_map_batch.py).

  S-a. Round-bucket standardised per-map batch difference (direct standardisation to the pooled
       round-bucket mix of each map), to see whether round composition explains the de_cache shift.
  S-b. Leave-one-match-out jackknife of the per-map batch difference for de_cache and de_dust2.
  S-c. Batch x map interaction (cluster-robust Wald) with each flagged map dropped in turn.
  S-d. Source composition: FACEIT-hosted share and build number of each flagged map x batch cell.
  S-e. Round-level CT rate of each flagged map split by FACEIT source within batch.

Loads M5_map_batch.py by path (its main() is guarded). Reads only the scratchpad table and the header.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M5_sensitivity.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
_spec = importlib.util.spec_from_file_location("M5_map_batch", HERE / "M5_map_batch.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def sa_round_standardised(d):
    print("\n== S-a. Round-bucket standardised per-map batch difference ==")
    rows = []
    for m in M.CORE:
        sub = d.filter(pl.col("map_name") == m)
        tot = sub.group_by("rb").agg(pl.len().alias("n_all"))
        tot = tot.with_columns((pl.col("n_all") / pl.col("n_all").sum()).alias("w"))
        cells = (sub.group_by(["rb", "batch"]).agg(pl.len().alias("n"), pl.col("ct_win").mean().alias("p")))
        p1 = cells.filter(pl.col("batch") == "b1008").select(["rb", "p"]).rename({"p": "p1008"})
        p0 = cells.filter(pl.col("batch") == "b1001_1007").select(["rb", "p"]).rename({"p": "p1001"})
        j = tot.join(p1, on="rb", how="left").join(p0, on="rb", how="left")
        ok = j.drop_nulls(["p1008", "p1001"])
        cover = float(ok["w"].sum())
        std_diff = float((ok["w"] * (ok["p1008"] - ok["p1001"])).sum() / cover)
        raw = float(sub.filter(pl.col("batch") == "b1008")["ct_win"].mean() -
                    sub.filter(pl.col("batch") == "b1001_1007")["ct_win"].mean())
        rows.append({"map": m, "raw_diff_pp": 100 * raw, "round_std_diff_pp": 100 * std_diff,
                     "round_buckets_covered": round(cover, 3)})
    print(pl.DataFrame(rows).with_columns(pl.col(["raw_diff_pp", "round_std_diff_pp"]).round(2)))


def sa_jackknife(d):
    print("\n== S-b. Leave-one-match-out jackknife of per-map batch difference (pp) ==")
    for m in ["de_cache", "de_dust2", "de_mirage"]:
        sub = d.filter(pl.col("map_name") == m)
        mm = (sub.group_by(["match_key", "batch"]).agg(pl.len().alias("r"), pl.col("ct_win").sum().alias("w")))
        a = mm.filter(pl.col("batch") == "b1001_1007")
        b = mm.filter(pl.col("batch") == "b1008")
        def diff(aa, bb):
            return 100 * (bb["w"].sum() / bb["r"].sum() - aa["w"].sum() / aa["r"].sum())
        full = diff(a, b)
        jk = []
        for k in b["match_key"].to_list():
            jk.append(diff(a, b.filter(pl.col("match_key") != k)))
        for k in a["match_key"].to_list():
            jk.append(diff(a.filter(pl.col("match_key") != k), b))
        jk = np.array(jk)
        n = len(jk)
        se_jk = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
        print(f"{m}: full diff {full:+.2f} pp; leave-one-match-out range [{jk.min():+.2f}, {jk.max():+.2f}]; "
              f"jackknife SE {se_jk:.2f} pp; sign-stable: {bool(np.all(np.sign(jk) == np.sign(full)))}")


def sa_drop_map(d):
    print("\n== S-c. Batch x map interaction with one core map dropped at a time ==")
    print("Full core set first, then drop-one. Cluster-robust Wald on the interaction coefficients.")
    dc = d.filter(pl.col("map_pool") != "rare")
    full = M.run_model(dc, ["round", "map", "batch", "batch_x_map"], "full core set")
    W, df, p = M.wald(full["beta"], full["V"], full["idx"]["batch_x_map"])
    print(f"  full core set: interaction chi2={W:.2f} df={df} p={p:.4g}")
    for m in M.CORE:
        if m == "de_mirage":
            continue
        sub = dc.filter(pl.col("map_name") != m)
        # rebuild map_pool/batch_x_map on the reduced set; de_mirage stays the reference
        res = M.run_model(sub, ["round", "map", "batch", "batch_x_map"], f"drop {m}")
        W, df, p = M.wald(res["beta"], res["V"], res["idx"]["batch_x_map"])
        print(f"  drop {m:<11}: interaction chi2={W:.2f} df={df} p={p:.4g}")
    print("  (de_mirage is the reference map in the design, so it cannot be dropped in this parametrisation.)")
    sub2 = dc.filter(~pl.col("map_name").is_in(["de_cache", "de_dust2"]))
    res2 = M.run_model(sub2, ["round", "map", "batch", "batch_x_map"], "drop de_cache and de_dust2 (refit)")
    W, df, p = M.wald(res2["beta"], res2["V"], res2["idx"]["batch_x_map"])
    print(f"  drop de_cache and de_dust2: interaction chi2={W:.2f} df={df} p={p:.4g}")


def sd_source(d):
    print("\n== S-d. Source composition of flagged maps (matches; rounds) ==")
    flag = d.filter(pl.col("map_name").is_in(["de_cache", "de_dust2", "de_mirage", "de_nuke"]))
    print(flag.group_by(["map_name", "batch", "faceit", "build_num"])
          .agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rounds"),
               (pl.col("ct_win").mean() * 100).round(2).alias("ct_pct"))
          .sort(["map_name", "batch", "faceit", "build_num"]))


def se_source_split(d):
    print("\n== S-e. Map CT% by batch x FACEIT source (core maps) ==")
    dc = d.filter(pl.col("map_pool") != "rare")
    print(dc.group_by(["map_name", "batch", "faceit"])
          .agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rounds"),
               (pl.col("ct_win").mean() * 100).round(2).alias("ct_pct"))
          .sort(["map_name", "batch", "faceit"]))


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_fmt_str_lengths(30)
    d = M.load()
    print(f"loaded clean rounds (1001_1007 + 1008): {d.height}, matches {d['match_key'].n_unique()}")
    sa_round_standardised(d)
    sa_jackknife(d)
    sd_source(d)
    se_source_split(d)
    sa_drop_map(d)


if __name__ == "__main__":
    main()
