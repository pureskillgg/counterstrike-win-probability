# verify_A2_4_toplabel.py: where does the top freeze_end/round_end label sit relative to round_start rows?
import os, json
import polars as pl
TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
UUID = r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
def channel_files(ch):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + ch]
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    return [os.path.join(dp, p["dataframe"]["dataframe"]) for p in meta["pages"]]
def lazy(ch, cols):
    return pl.concat([pl.scan_parquet(f).select(cols) for f in channel_files(ch)], how="vertical_relaxed")
hdr = pl.scan_parquet(os.path.join(TOME, "header", "dataframe_00000")).select(["key", "build_num"]).collect()
hdr = hdr.with_columns(pl.col("key").str.extract(UUID, 1).alias("u"), pl.col("build_num").cast(pl.Utf8).alias("build"))
st = lazy("round_state", ["round", "tick", "event_type", "match_key"]).filter(pl.col("event_type") == "round_freeze_end").collect()
re_ = lazy("round_end", ["round", "tick", "winner_team_code", "match_key"]).collect()
rs = lazy("round_start", ["round", "tick", "match_key"]).collect()
for df in (st, re_, rs):
    df.insert_column(0, pl.col("match_key").str.extract(UUID, 1).alias("u").alias("u"))
# per (match, label) sets
S = {}
for nm, df in [("fe", st), ("re", re_), ("rs", rs)]:
    S[nm] = df.group_by("u").agg(pl.col("round").unique().sort().alias(nm + "_set"))
M = hdr.join(S["fe"], on="u", how="left").join(S["re"], on="u", how="left").join(S["rs"], on="u", how="left")
M = M.with_columns(
    pl.col("fe_set").list.max().alias("fe_top"), pl.col("re_set").list.max().alias("re_top"), pl.col("rs_set").list.max().alias("rs_top"),
    pl.col("fe_set").list.len().alias("fe_len"), pl.col("re_set").list.len().alias("re_len"), pl.col("rs_set").list.len().alias("rs_len"))
print("(build, fe_top - re_top, fe_top - rs_top) -> matches")
print(M.with_columns((pl.col("fe_top") - pl.col("re_top")).alias("a"), (pl.col("fe_top") - pl.col("rs_top")).alias("b"))
      .group_by(["build", "a", "b"]).len().sort(["build", "a", "b"]).rows(), flush=True)
# how many (match, round_end label) pairs have no round_start row with the same label
rk = rs.select(["u", "round"]).unique()
re_k = re_.select(["u", "round"]).unique()
miss = re_k.join(rk, on=["u", "round"], how="anti")
print("round_end (match,label) pairs with no round_start row, total:", miss.height, flush=True)
print("   by build:", miss.join(hdr.select(["u", "build"]), on="u", how="left").group_by("build").len().rows(), flush=True)
mm = miss.join(M.select(["u", "fe_top", "re_top", "rs_top"]), on="u", how="left").with_columns((pl.col("round") == pl.col("re_top")).alias("is_top"))
print("   missing-rs label is the match's top round_end label:", mm.group_by("is_top").len().rows(), flush=True)
print("   sample of missing (u, label):", mm.head(5).select(["u", "round", "fe_top", "re_top", "rs_top"]).rows(), flush=True)
print("DONE", flush=True)
