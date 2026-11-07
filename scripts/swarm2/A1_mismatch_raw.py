# A1: for mismatch matches (round_end short of header S), compare tome vs raw E: round-level channels and score_update.
import os, glob, json
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
m = pl.read_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_round_end_per_match.csv", infer_schema_length=0).with_columns(
    pl.col("S").cast(pl.Int64), pl.col("rows").cast(pl.Int64), pl.col("distinct_rounds").cast(pl.Int64), pl.col("max_round").cast(pl.Int64))
mis = m.filter(pl.col("distinct_rounds") != pl.col("S")).sort("key")
# pick 4 mismatch matches: first of 10/01, first of 10/02, first of 10/03, and last mismatch overall
sel = []
for day in ["01", "02", "03"]:
    cand = mis.filter(pl.col("key").str.slice(11, 2) == day)
    if cand.height:
        sel.append(cand.row(0, named=True))
sel.append(mis.row(-1, named=True))
print("selected:", [(r["key"][11:13] + "/" + r["key"].split("/")[4][:8], r["S"], r["rows"]) for r in sel])
def tome_rows(channel, key):
    pages = sorted(glob.glob(os.path.join(T, channel, "dataframe_*")))
    lf = pl.concat([pl.scan_parquet(p) for p in pages], how="diagonal_relaxed").filter(pl.col("match_key") == key)
    return lf.collect()
def raw(path):
    return pl.read_parquet(path)
out = []
for r in sel:
    key = r["key"]; dd, mid = key.split("/")[3], key.split("/")[4]
    base = os.path.join(E, dd, mid)
    rre = raw(os.path.join(base, "round_end"))
    tre = tome_rows("basic_round_end.2026-09-18,2026-10-08", key)
    rsu = raw(os.path.join(base, "score_update")) if os.path.exists(os.path.join(base, "score_update")) else None
    tsu = tome_rows("basic_score_update.2026-09-18,2026-10-08", key)
    rrs = raw(os.path.join(base, "round_start"))
    trs = tome_rows("basic_round_start.2026-09-18,2026-10-08", key)
    rh = pl.read_parquet(os.path.join(base, "header"))
    print("== key", key, "S", r["S"], "tome round_end rows", tre.height, "| raw round_end rows", rre.height,
          "| raw round max", rre.get_column("round").max(), "| tome max", tre.get_column("round").max())
    print("   raw round_start rows", rrs.height, "tome", trs.height, "| raw score_update rows", None if rsu is None else rsu.height, "tome", tsu.height)
    print("   raw round_end cols", rre.columns)
    print("   raw score_update cols", None if rsu is None else rsu.columns)
    if rsu is not None and rsu.height:
        cols = [c for c in rsu.columns if c in ("round", "tick", "t_score", "ct_score", "team_t_score", "team_ct_score", "score", "team_code", "team")]
        print("   raw score_update tail:", rsu.tail(3).select(cols if cols else rsu.columns[:6]).rows())
    if tsu.height:
        print("   tome score_update tail:", tsu.tail(3).drop("match_key").rows()[:3])
    print("   raw header T/CT final:", rh.select(["t_starters_score_final", "ct_starters_score_final"]).rows() if "t_starters_score_final" in rh.columns else "n/a")
    print("   raw round_end tail:", rre.tail(3).rows())
    print("   tome round_end tail:", tre.drop("match_key").tail(3).rows())
    out.append(dict(key=key, S=r["S"], tome_rows=tre.height, raw_rows=rre.height))
json.dump(out, open(r"G:\pureskill\csds-cs2\logs\swarm2\A1_mismatch_raw.json", "w"), indent=1)
