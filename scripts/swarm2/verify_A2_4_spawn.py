# verify_A2_4_spawn.py: timing of round-labelled player_spawn rows relative to the round_state anchors (seconds, tick_rate 64).
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
anch = st.filter(pl.col("event_type").is_in(["round_freeze_end", "round_officially_ended"]) & pl.col("round").is_in([1, 2, 3]))
anch = anch.unique(["u", "round", "event_type"], keep="first").select(["u", "round", "event_type", "tick"])
A = {}
for (r, ev) in [(1, "round_freeze_end"), (1, "round_officially_ended"), (2, "round_freeze_end"), (2, "round_officially_ended")]:
    A[f"{ev[6:12] if False else ev}_{r}"] = anch.filter((pl.col("round") == r) & (pl.col("event_type") == ev)).select(["u", "tick"]).rename({"tick": f"{'fe' if ev=='round_freeze_end' else 'ob'}{r}"})
W = None
for k, v in A.items():
    W = v if W is None else W.join(v, on="u", how="full", coalesce=True)
sp = lazy("player_spawn", ["round", "tick", "match_key"]).collect()
sp = sp.with_columns(pl.col("match_key").str.extract(UUID, 1).alias("u")).join(hdr.select(["u", "build"]), on="u", how="left")
s1 = sp.filter(pl.col("round") == 1).join(W, on="u", how="left")
print("round-1 spawn rows by build:", s1.group_by("build").len().rows(), flush=True)
for b in ["9842", "9961"]:
    x = s1.filter(pl.col("build") == b)
    out = {}
    for c in ["fe1", "ob1", "fe2", "ob2"]:
        if c in x.columns:
            out[c] = ((x["tick"] - x[c]) / 64).drop_nulls().describe() if x[c].null_count() < x.height else None
    print(f"build {b}: (spawn_tick - anchor)/64 seconds, round-1 spawn rows, by anchor:", flush=True)
    for c in ["fe1", "ob1", "fe2", "ob2"]:
        if c in x.columns:
            d = (x["tick"] - x[c]) / 64
            d = d.drop_nulls()
            if d.len():
                print(f"   {c}: n={d.len()} min={d.min():.1f} p05={d.quantile(0.05):.1f} med={d.median():.1f} p95={d.quantile(0.95):.1f} max={d.max():.1f}", flush=True)
            else:
                print(f"   {c}: no anchor for these matches", flush=True)
    print(f"   spawn tick vs ob1 (sign): before={int((x['tick'] < x['ob1']).sum()) if 'ob1' in x.columns else 'NA'} after={int((x['tick'] >= x['ob1']).sum()) if 'ob1' in x.columns else 'NA'}", flush=True)
    print("   spawn rows per match (n, matches):", x.group_by("u").agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("matches")).sort("n").rows(), flush=True)
# control: round-2 spawns in 9842 relative to fe2 and ob2
s2 = sp.filter((pl.col("round") == 2) & (pl.col("build") == "9842")).join(W, on="u", how="left")
d2 = ((s2["tick"] - s2["fe2"]) / 64).drop_nulls()
print("control 9842 round-2 spawn minus fe2 (s): n", d2.len(), "min", round(d2.min(), 1), "med", round(d2.median(), 1), "max", round(d2.max(), 1), flush=True)
d2b = ((s2["tick"] - s2["ob2"]) / 64).drop_nulls()
print("control 9842 round-2 spawn minus ob2 (s): n", d2b.len(), "min", round(d2b.min(), 1), "med", round(d2b.median(), 1), "max", round(d2b.max(), 1), flush=True)
print("DONE", flush=True)
