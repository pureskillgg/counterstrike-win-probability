"""Split the 429-match parity result by whether a match has tome rows; check the 874 admin-but-not-grid matches
against rounds.parquet use flags. Reads local report/grid/rounds files only. Writes nothing to S3.
"""
import glob
import json
import os

import polars as pl

HERE = os.path.expanduser("~/adam/status_scripts/parity")
COMB = os.path.expanduser("~/adam/status/data/combined")
ADMIN = os.path.expanduser("~/tomes/tome/csds/header")
LO, HI = "2026/10/09", "2026/10/25"

rep = json.load(open(os.path.join(HERE, "parity_report2.json")))
pm = rep["per_match"]
with_tome = [m for m in pm if m["V_rows"] > 0]
no_tome = [m for m in pm if m["V_rows"] == 0]
print(f"matches with tome rows {len(with_tome)}; without {len(no_tome)}", flush=True)
print("with tome: L_rows", sum(m["L_rows"] for m in with_tome), "L_only", sum(m["L_only"] for m in with_tome),
      "V_only", sum(m["V_only"] for m in with_tome), "mismatch", sum(m["mismatch"] for m in with_tome), flush=True)
print("without tome: L_rows", sum(m["L_rows"] for m in no_tome), flush=True)
bad = sorted([m for m in with_tome if m["L_only"] > 0], key=lambda m: -m["L_only"])
print("in-tome matches with L_only > 0:", len(bad), flush=True)
for m in bad[:8]:
    print("  ", m["chunk"], m["match_key"], "L", m["L_rows"], "V", m["V_rows"], "Lonly", m["L_only"], "Vonly", m["V_only"],
          flush=True)

# admin keys in window that the grid lacks: do they have usable rounds?
fs = sorted(glob.glob(os.path.join(ADMIN, "dataframe_*")))
admin = set(pl.concat([pl.read_parquet(f, columns=["key"]) for f in fs])["key"].to_list())
G = pl.read_parquet(os.path.join(COMB, "grid.parquet"), columns=["match_key", "upload_day", "era"])
grid_mk = set(G.filter((pl.col("era") == "2026") & (pl.col("upload_day") >= LO) & (pl.col("upload_day") <= HI))
              ["match_key"].to_list())


def day(k):
    p = k.split("/")
    return f"{p[1]}/{p[2]}/{p[3]}"


adm_win = {k for k in admin if LO <= day(k) <= HI}
gap = sorted(adm_win - grid_mk)
print(f"admin in window not in grid: {len(gap)}", flush=True)
R = pl.read_parquet(os.path.join(COMB, "rounds.parquet"), columns=["match_key", "round", "use", "round_status"])
Rg = R.filter(pl.col("match_key").is_in(gap))
per = Rg.group_by("match_key").agg(
    pl.len().alias("rounds"), (pl.col("use") == True).sum().alias("use_rounds")  # noqa: E712
)
print("gap matches with rounds in rounds.parquet:", per.height, "| with >=1 use round:",
      per.filter(pl.col("use_rounds") > 0).height, flush=True)
print("gap matches: use_rounds distribution:", per.group_by("use_rounds").agg(pl.len().alias("matches"))
      .sort("use_rounds").rows()[:10], flush=True)
print("gap matches not in rounds.parquet at all:", len(set(gap) - set(per["match_key"].to_list())), flush=True)
print("gap by day:", pl.Series([day(k) for k in gap]).value_counts().sort("count", descending=True).head(6).rows(),
      flush=True)
