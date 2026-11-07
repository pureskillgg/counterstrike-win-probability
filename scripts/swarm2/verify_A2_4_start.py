# verify_A2_4_start.py: earliest tick present per match per channel vs the round-1 anchors, by build (seconds, tick_rate 64).
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
st = lazy("round_state", ["round", "tick", "event_type", "match_key"]).collect()
st = st.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u"))
ob1 = st.filter((pl.col("round") == 1) & (pl.col("event_type") == "round_officially_ended")).group_by("u").agg(pl.col("tick").min().alias("ob1"))
fe1 = st.filter((pl.col("round") == 1) & (pl.col("event_type") == "round_freeze_end")).group_by("u").agg(pl.col("tick").min().alias("fe1"))
A = hdr.select(["u", "build"]).join(ob1, on="u", how="left").join(fe1, on="u", how="left")
print("anchors present by build (ob1, fe1, n):",
      A.group_by("build").agg(pl.col("ob1").is_not_null().sum().alias("ob1"), pl.col("fe1").is_not_null().sum().alias("fe1"), pl.len().alias("n")).sort("build").rows(), flush=True)
chans = {"round_state": ["round", "tick", "match_key"], "round_start": ["round", "tick", "match_key"],
         "round_end": ["round", "tick", "match_key"], "player_info": ["round", "match_key"],
         "player_spawn": ["round", "tick", "match_key"], "player_death": ["round", "tick", "match_key"],
         "bomb_state": ["round", "tick", "match_key"], "player_personal": ["round", "match_key"]}
for ch, cols in chans.items():
    try:
        d = lazy(ch, cols).collect()
    except Exception as e:
        print("  ", ch, "skip:", type(e).__name__, flush=True)
        continue
    if "tick" not in d.columns:
        print("  ", ch, "has no tick column; skipped", flush=True)
        continue
    d = d.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u")).group_by("u").agg(pl.col("tick").min().alias("tmin"))
    d = d.join(A, on="u", how="right")
    for b in ["9842", "9961"]:
        x = d.filter(pl.col("build") == b)
        rel = ((x["tmin"] - x["ob1"]) / 64).drop_nulls()
        rel_f = ((x["tmin"] - x["fe1"]) / 64).drop_nulls()
        if rel.len() == 0:
            print(f"   {ch:16s} {b}: no rows", flush=True)
            continue
        f_med = rel_f.median() if rel_f.len() else float("nan")
        print(f"   {ch:16s} {b}: matches={rel.len()}  (min_tick - ob1)/64 s: min={rel.min():.1f} p05={rel.quantile(0.05):.1f} "
              f"med={rel.median():.1f} p95={rel.quantile(0.95):.1f}  | (min_tick - fe1)/64 s med={f_med:.1f}", flush=True)
print("DONE", flush=True)
