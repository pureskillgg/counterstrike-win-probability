# verify_A2_4_extra.py: (i) score at the round-1 end bundle vs freeze_end(2); (ii) any round-1 rows in player-level channels by build.
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
st = lazy("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"]).collect()
st = st.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u")).join(hdr.select(["u", "build"]), on="u", how="left")
print("(i) round-1 end bundle score vs freeze_end(2) score", flush=True)
b1 = (st.filter((pl.col("round") == 1) & (pl.col("event_type") == "round_officially_ended"))
      .sort("tick").group_by(["u", "build"]).agg(pl.col("t_score").first().alias("bt"), pl.col("ct_score").first().alias("bc"), pl.len().alias("n_bundle")))
print("   round-1 round_officially_ended rows per match:", b1.group_by("n_bundle").len().rows(), flush=True)
f2 = st.filter((pl.col("round") == 2) & (pl.col("event_type") == "round_freeze_end")).sort("tick").group_by(["u", "build"]).agg(
    pl.col("t_score").first().alias("ft"), pl.col("ct_score").first().alias("fc"))
j = b1.join(f2, on=["u", "build"], how="inner")
j = j.with_columns(
    pl.when((pl.col("bt") == 1) & (pl.col("bc") == 0)).then(2).when((pl.col("bc") == 1) & (pl.col("bt") == 0)).then(3).otherwise(0).alias("side_b1"),
    pl.when((pl.col("ft") == 1) & (pl.col("fc") == 0)).then(2).when((pl.col("fc") == 1) & (pl.col("ft") == 0)).then(3).otherwise(0).alias("side_f2"))
print("   (build, bundle score == fe(2) score) ->", j.group_by(["build", (pl.col("bt") == pl.col("ft")) & (pl.col("bc") == pl.col("fc"))]).len().sort("build").rows(), flush=True)
print("   (build, side_b1 == side_f2) ->", j.group_by(["build", (pl.col("side_b1") == pl.col("side_f2"))]).len().sort("build").rows(), flush=True)
print("   (build, side at round-1 end bundle) ->", j.group_by(["build", "side_b1"]).len().sort(["build", "side_b1"]).rows(), flush=True)
print("(ii) rows labelled round 1 in player-level channels, by build (matches with >=1 row)", flush=True)
for ch in ["player_info", "player_death", "player_spawn", "weapon_fire", "player_hurt", "bomb_state", "round_mvp", "player_personal"]:
    try:
        d = lazy(ch, ["round", "match_key"]).collect()
    except Exception as e:
        print("   ", ch, "-> no round column or read error:", type(e).__name__, flush=True)
        continue
    d = d.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u")).join(hdr.select(["u", "build"]), on="u", how="left")
    r1 = d.filter(pl.col("round") == 1)
    print(f"   {ch:16s} total rows {d.height:>8}  round1 rows by build (rows, matches):",
          r1.group_by("build").agg(pl.len().alias("rows"), pl.col("u").n_unique().alias("matches")).sort("build").rows(), flush=True)
print("DONE", flush=True)
