"""D3-4 verifier: spot-check (8,)-group and (12,)-control matches against the E: raw 2024 tail copy.
READ-ONLY. Raw layout: E:\...\2024\02\<DD>\<match_uuid>\{round_end,round_state,header}  (single parquet files, not folders).
Compares raw distinct round_end rounds (count, max) and raw header final scores with the tome.
Also compares raw round_state freeze_end round coverage with the tome.
"""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
def folder(ch):
    return TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
def read(ch, cols=None):
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")
H = read("header", ["key", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day")).filter(pl.col("day") >= 12)
keys = H["key"].to_list()
hdr = {k: (int(t), int(c)) for k, t, c in H.select(["key", "t_starters_score_final", "ct_starters_score_final"]).iter_rows()}
RE = read("round_end", ["match_key", "round"]).filter(pl.col("match_key").is_in(keys)).unique(["match_key", "round"], keep="first")
tome = {m: (n, mx) for m, n, mx in RE.group_by("match_key").agg(pl.len().alias("n"), pl.col("round").max().alias("mx")).select(["match_key", "n", "mx"]).iter_rows()}
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(pl.col("match_key").is_in(keys))
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").group_by(["match_key", "round"])
      .agg(pl.col("t_score").first(), pl.col("ct_score").first()))
del RS
fe = {(m, r): (t, c) for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
S = collections.defaultdict(list)
for (m, r) in sorted(fe):
    if (m, r + 1) in fe:
        a, b = fe[(m, r)], fe[(m, r + 1)]
        dt, dc = b[0] - a[0], b[1] - a[1]
        if dt + dc == 1 and min(dt, dc) < 0: S[m].append(r)
keys8 = sorted(m for m in keys if tuple(S[m]) == (8,))
keys12 = sorted(m for m in keys if tuple(S[m]) == (12,))
print("(8,) keys:", len(keys8), "| (12,) keys:", len(keys12))
sample8 = [keys8[i] for i in range(0, len(keys8), max(1, len(keys8) // 8))][:8]
sample12 = [keys12[i] for i in range(0, len(keys12), max(1, len(keys12) // 4))][:4]
def raw_info(m):
    parts = m.split("/"); day, uuid = parts[3], parts[4]
    md = RAW / day / uuid
    out = {"exists": md.is_dir()}
    if not md.is_dir(): return out
    p = md / "round_end"
    if p.is_file():
        rr = pl.read_parquet(p, columns=["round"]).unique()
        out["re_n"], out["re_mx"] = rr.height, int(rr["round"].max())
    q = md / "header"
    if q.is_file():
        hh = pl.read_parquet(q, columns=["t_starters_score_final", "ct_starters_score_final"])
        out["hdr"] = (int(hh["t_starters_score_final"][0]), int(hh["ct_starters_score_final"][0]))
    s = md / "round_state"
    if s.is_file():
        ss = pl.read_parquet(s, columns=["round", "event_type"]).filter(pl.col("event_type") == "round_freeze_end").unique("round")
        out["fe_n"] = ss.height
    return out
for label, sample in [("(8,)", sample8), ("(12,) control", sample12)]:
    print(f"\n== {label} sample ({len(sample)}) ==")
    for m in sample:
        info = raw_info(m)
        print(f"  {m.split('/')[4][:8]} day {m.split('/')[3]} | tome re(n,max)={tome.get(m)} tome_hdr={hdr[m]} | raw={info}")
print("\nDONE")
