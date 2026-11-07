# verify_A2_4_r13build.py: bomb-outcome vs winner_team_code mismatch by build and round (round_state bomb events; round_end winner).
# exploded -> winner must be 2 (T); defused -> winner must be 3 (CT). Context for the open round-13 defect only.
import os, json
import polars as pl
pl.Config.set_tbl_rows(40)
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
st = lazy("round_state", ["round", "event_type", "match_key"]).filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect()
st = st.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u")).unique(["u", "round", "event_type"], keep="first")
re_ = lazy("round_end", ["round", "winner_team_code", "match_key"]).collect()
re_ = re_.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u")).unique(["u", "round"], keep="first")
j = st.join(re_.select(["u", "round", "winner_team_code"]), on=["u", "round"], how="left").join(hdr.select(["u", "build"]), on="u", how="left")
j = j.with_columns(pl.when(pl.col("event_type") == "bomb_exploded").then(pl.col("winner_team_code") != 2).otherwise(pl.col("winner_team_code") != 3).alias("mismatch"))
print("bomb-outcome rows with a round_end winner, by build (rows, mismatch):", j.group_by("build").agg(pl.len().alias("rows"), pl.col("mismatch").sum().alias("mismatch")).sort("build").rows(), flush=True)
r = j.filter(pl.col("round").is_in([12, 13, 14]))
print("rounds 12-14 by build and round (rows, mismatch):", r.group_by(["build", "round"]).agg(pl.len().alias("rows"), pl.col("mismatch").sum().alias("mismatch")).sort(["build", "round"]).rows(), flush=True)
print("round 13 by build x event (rows, mismatch):", r.filter(pl.col("round") == 13).group_by(["build", "event_type"]).agg(pl.len().alias("rows"), pl.col("mismatch").sum().alias("mismatch")).sort(["build", "event_type"]).rows(), flush=True)
print("round 13 by folder day within build 9842 / 9961 (day, rows, mismatch) using header match_key day:", flush=True)
hk = hdr.with_columns(pl.col("key").str.split("/").list.get(3).alias("day")).select(["u", "day"])
rd = r.filter(pl.col("round") == 13).join(hk, on="u", how="left")
print(rd.group_by(["build", "day"]).agg(pl.len().alias("rows"), pl.col("mismatch").sum().alias("mismatch")).sort(["build", "day"]).rows(), flush=True)
print("DONE", flush=True)
