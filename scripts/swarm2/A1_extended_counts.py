# A1: row counts tome vs raw E: for 80 shortfall matches + 20 seeded random matches (all 35 channels).
# Tome counts: one pass per channel over its pages (match_key column only). Raw counts: parquet footer num_rows.
# Also raw tick span: max tick in raw 'tick' file minus last round_end tick, for shortfall matches and 40 control matches.
import os, glob, json, random, time
import polars as pl
import pyarrow.parquet as pq
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
E = r"E:\csds_final\csds\2026\10"
t0 = time.time()
m = pl.read_csv(r"G:\pureskill\csds-cs2\logs\swarm2\A1_round_end_per_match.csv", infer_schema_length=0).with_columns(
    pl.col("S").cast(pl.Int64), pl.col("distinct_rounds").cast(pl.Int64))
short = m.filter(pl.col("distinct_rounds") != pl.col("S")).get_column("key").to_list()
ok_keys = m.filter(pl.col("distinct_rounds") == pl.col("S")).get_column("key").to_list()
rng = random.Random(7)
sample_ok = rng.sample(ok_keys, 20)
control = rng.sample(ok_keys, 40)
sel = set(short) | set(sample_ok)
print("selected matches:", len(sel), "shortfall:", len(short), flush=True)
chans = sorted(d for d in os.listdir(T) if d != "header" and not d.startswith("."))
tome_counts = {k: {} for k in sel}
for c in chans:
    pages = sorted(glob.glob(os.path.join(T, c, "dataframe_*")))
    lf = pl.concat([pl.scan_parquet(p).select("match_key") for p in pages], how="vertical")
    g = lf.filter(pl.col("match_key").is_in(list(sel))).group_by("match_key").agg(pl.len().alias("n")).collect()
    d = dict(zip(g.get_column("match_key").to_list(), g.get_column("n").to_list()))
    raw_name = c.split(".")[0].replace("basic_", "", 1)
    for k in sel:
        n_tome = d.get(k, 0)
        dd, mid = k.split("/")[3], k.split("/")[4]
        rp = os.path.join(E, dd, mid, raw_name)
        n_raw = pq.ParquetFile(rp).metadata.num_rows if os.path.exists(rp) else None
        tome_counts[k][raw_name] = (n_tome, n_raw)
print("counts pass done", round(time.time() - t0, 1), flush=True)
mism = {}
for k in sel:
    bad = {ch: v for ch, v in tome_counts[k].items() if v[1] is None or v[0] != v[1]}
    if bad:
        mism[k] = bad
print("matches with any tome/raw count mismatch:", len(mism), "of", len(sel))
for k, v in list(mism.items())[:10]:
    print("  ", k, v)
# raw round_end vs tome for shortfall set
rr = {}
for k in short:
    dd, mid = k.split("/")[3], k.split("/")[4]
    base = os.path.join(E, dd, mid)
    rre = pl.read_parquet(os.path.join(base, "round_end"), columns=["round", "tick"])
    tk = pl.read_parquet(os.path.join(base, "tick"), columns=["tick"]).get_column("tick").max()
    rr[k] = dict(raw_round_end_rows=rre.height, raw_round_end_max_round=int(rre.get_column("round").max()),
                 last_round_end_tick=int(rre.get_column("tick").max()), raw_tick_max=int(tk))
gaps_short = [v["raw_tick_max"] - v["last_round_end_tick"] for v in rr.values()]
ctrl = {}
for k in control:
    dd, mid = k.split("/")[3], k.split("/")[4]
    base = os.path.join(E, dd, mid)
    rre = pl.read_parquet(os.path.join(base, "round_end"), columns=["round", "tick"])
    tk = pl.read_parquet(os.path.join(base, "tick"), columns=["tick"]).get_column("tick").max()
    ctrl[k] = dict(raw_round_end_rows=rre.height, last_round_end_tick=int(rre.get_column("tick").max()), raw_tick_max=int(tk))
gaps_ctrl = [v["raw_tick_max"] - v["last_round_end_tick"] for v in ctrl.values()]
def stat(xs):
    xs = sorted(xs); n = len(xs)
    return dict(n=n, min=xs[0], median=xs[n // 2], max=xs[-1])
print("shortfall: raw round_end rows == tome round_end rows:", sum(1 for k in short if tome_counts[k]["round_end"][0] == rr[k]["raw_round_end_rows"]), "of", len(short))
print("shortfall: raw round_end rows == header S - gap? (raw rows vs tome rows) ok")
print("tick gap (raw tick max - last round_end tick), shortfall:", stat(gaps_short))
print("tick gap, control (matched):", stat(gaps_ctrl))
print("elapsed", round(time.time() - t0, 1))
json.dump(dict(mismatch=mism, shortfall_raw=rr, control_raw=ctrl), open(r"G:\pureskill\csds-cs2\logs\swarm2\A1_extended_counts.json", "w"), indent=1, default=str)
