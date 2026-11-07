# verify_A2_4_rst_gap.py: rounds with a round_end row but no round_start row, split by the bake-off's earlier filters
# (player_info 5 T and 5 CT by unique player id; player_disconnect in round). Build from header.
import os, json, glob
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
def U(df):
    return df.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u"))
RE = U(lazy("round_end", ["round", "match_key", "winner_team_code", "player_count"]).collect()).unique(["u", "round"], keep="first")
RST = U(lazy("round_start", ["round", "match_key"]).collect()).select(["u", "round"]).unique()
PI = U(lazy("player_info", ["round", "match_key", "player_id", "team_code"]).collect())
print("player_info rows:", PI.height, flush=True)
pi = PI.unique(["u", "round", "player_id"], keep="first").group_by(["u", "round"]).agg(
    (pl.col("team_code") == 2).sum().alias("nT"), (pl.col("team_code") == 3).sum().alias("nCT"), pl.len().alias("nP"))
DC = U(lazy("player_disconnect", ["round", "match_key"]).collect()).select(["u", "round"]).unique()
print("player_disconnect rounds:", DC.height, flush=True)
R = RE.join(hdr.select(["u", "build"]), on="u", how="left")
R = R.with_columns(pl.lit(True).alias("re"))
R = R.join(RST.with_columns(pl.lit(True).alias("has_rs")), on=["u", "round"], how="left")
R = R.join(pi, on=["u", "round"], how="left").join(DC.with_columns(pl.lit(True).alias("has_disc")), on=["u", "round"], how="left")
R = R.with_columns(pl.col("has_rs").fill_null(False), pl.col("has_disc").fill_null(False),
                   ((pl.col("nT") == 5) & (pl.col("nCT") == 5) & (pl.col("nP") == 10)).fill_null(False).alias("pi_ok"))
print("rounds with round_end (dedup):", R.height, flush=True)
print("(build, has_rs, pi_ok, has_disc) -> rounds:", R.group_by(["build", "has_rs", "pi_ok", "has_disc"]).len().sort(["build", "has_rs", "pi_ok", "has_disc"]).rows(), flush=True)
miss = R.filter(~pl.col("has_rs"))
print("missing-rs rounds by build x round_end label==top of match:", miss.group_by("build").len().rows(), flush=True)
top = R.group_by("u").agg(pl.col("round").max().alias("top"))
miss = miss.join(top, on="u").with_columns((pl.col("round") == pl.col("top")).alias("is_top"))
print("missing-rs rounds: is_top x pi_ok x has_disc:", miss.group_by(["is_top", "pi_ok", "has_disc"]).len().sort(["is_top", "pi_ok", "has_disc"]).rows(), flush=True)
print("missing-rs and pi_ok and no disc (cross_period-style drop count), by build:",
      miss.filter(pl.col("pi_ok") & ~pl.col("has_disc")).group_by("build").len().rows(), flush=True)
print("missing-rs, pi_ok false: nT/nCT sample:", miss.filter(~pl.col("pi_ok")).select(["nT", "nCT", "nP"]).head(8).rows(), flush=True)
print("DONE", flush=True)
