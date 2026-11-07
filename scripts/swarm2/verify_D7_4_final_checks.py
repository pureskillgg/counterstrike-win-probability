"""verify_D7_4 final checks (independent of D7_rounds_all.parquet). Raw 2024 tome, player_count==20 outcome rounds.
(a) negative-lag family by build (match level); (b) the 9961 matches with no -429 (what they contain);
(c) host of the two FACEIT-hosted matches; (d) post-onset round-13 cross-tab, lag -429 vs other lag, mismatch;
(e) lag family by round position in build 9961 (where is -429 absent, where do other negatives sit);
(f) positive-lag rounds in build 9961 (round positions).
ASSUMPTION: player_count == 20; tick rate 64; onset = first -429 row (2024-02-07 01:01:18), found from data.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def channel(name):
    d = sorted(glob.glob(str(T24 / f"basic_{name}.*")))[0]
    return pl.scan_parquet(files(Path(d)))


H = (pl.read_parquet(files(T24 / "header")[0], columns=["key", "build_num", "match_date", "server_name"]).rename({"key": "match_key"})
       .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt")))
RE = channel("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
BS = (channel("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
        .select(["match_key", "round", "tick", "event_type"]).collect())
OUT = BS.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick"), (pl.col("event_type") == "bomb_defused").any().alias("d"))
O = (RE.join(OUT, on=["match_key", "round"], how="inner").filter(pl.col("player_count") == 20)
       .join(H.select(["match_key", "build_num", "mdt", "server_name"]), on="match_key", how="left")
       .with_columns(pl.when(pl.col("d")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
                     (pl.col("out_tick") - pl.col("tick")).alias("lag"))
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm")))
ONSET = O.filter(pl.col("lag") == -429).sort("mdt").head(1).select("mdt").item()
O = O.with_columns(pl.when(pl.col("mdt") < ONSET).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"))
print("onset:", ONSET, "| outcome rows (p20):", O.height, "| mm:", int(O["mm"].sum()))

# (a) negative-lag family, match level
M = (O.group_by("match_key").agg(pl.col("build_num").first(), pl.col("mdt").first(), pl.col("server_name").first(),
                                  pl.len().alias("n_out"), (pl.col("lag") == -429).any().alias("has429"),
                                  (pl.col("lag") < 0).any().alias("has_neg"), (pl.col("lag") > 0).any().alias("has_pos"),
                                  pl.col("mm").any().alias("has_mm")))
print("\n(a) match level: build x has_neg (any negative lag):", M.group_by(["build_num", "has_neg"]).len().sort(["build_num", "has_neg"]).rows())
print("(a) build x has_429 x has_neg:", M.group_by(["build_num", "has429", "has_neg"]).len().sort(["build_num", "has429", "has_neg"]).rows())
print("(a) rows: negative-lag share by period (p20 clean rows): pre neg =",
      O.filter((pl.col("period") == "pre") & (pl.col("lag") < 0)).height, "| post neg clean =",
      O.filter((pl.col("period") == "post") & (pl.col("lag") < 0) & ~pl.col("mm")).height, "of post clean =",
      O.filter((pl.col("period") == "post") & ~pl.col("mm")).height)

# (b) build 9961 matches without -429
m9961_no429 = M.filter((pl.col("build_num") == 9961) & ~pl.col("has429"))
print("\n(b) 9961 matches without -429:", m9961_no429.height, "| with mm:", int(m9961_no429["has_mm"].sum()),
      "| all have negative lag:", bool(m9961_no429["has_neg"].all()))
ids = m9961_no429["match_key"].to_list()
rows_b = O.filter(pl.col("match_key").is_in(ids)).select(["round", "lag", "mm", "bomb_side"]).sort(["round"])
print("(b) round positions of their outcome rows (round, lag):", [(r, l) for r, l in rows_b.select(["round", "lag"]).rows()][:60])

# (c) host of FACEIT-hosted matches
fa = H.filter(~pl.col("server_name").str.starts_with("Valve")).select(["match_key", "build_num", "mdt", "server_name"])
print("\n(c) non-Valve host matches:", fa.height)
for mk, b, t, sn in fa.rows():
    sub = O.filter(pl.col("match_key") == mk).select(["round", "lag", "mm"]).sort("round")
    print("   build", b, t, "host", sn[:30], "| outcome rows (round,lag,mm):", sub.rows())

# (d) post-onset round 13 cross-tab, correct comparison (within swap round, post period)
r13 = O.filter((pl.col("round") == 13) & (pl.col("period") == "post"))
print("\n(d) post-onset round-13 rows:", r13.height, "| mm:", int(r13["mm"].sum()))
print("(d) lag == -429 vs other:", r13.with_columns((pl.col("lag") == -429).alias("is429")).group_by(["is429", "mm"]).len().sort(["is429", "mm"]).rows())
print("(d) lag < 0 vs other:", r13.with_columns((pl.col("lag") < 0).alias("neg")).group_by(["neg", "mm"]).len().sort(["neg", "mm"]).rows())
p_other = O.filter((pl.col("period") == "post") & ~pl.col("round").is_in([9, 13, 25, 28]))
print("(d) non-swap post rows by lag==-429:", p_other.with_columns((pl.col("lag") == -429).alias("is429")).group_by(["is429", "mm"]).len().sort(["is429", "mm"]).rows())

# (e) build 9961 lag family by round position
p9961 = O.filter(pl.col("build_num") == 9961)
prof = (p9961.with_columns(pl.when(pl.col("lag") == -429).then(pl.lit("429")).when(pl.col("lag") < 0).then(pl.lit("neg_other"))
                              .when(pl.col("lag") == 0).then(pl.lit("zero")).otherwise(pl.lit("pos")).alias("fam"))
        .group_by(["round", "fam"]).len().sort(["round", "fam"]))
rounds_9961 = sorted(p9961["round"].unique().to_list())
print("\n(e) build 9961: per round, family counts (round: {fam: n}) for rounds 1-30:")
out_e = []
for r in range(1, 31):
    sub = prof.filter(pl.col("round") == r)
    if sub.height:
        out_e.append((r, {f: n for f, n in sub.select(["fam", "len"]).rows()}))
print(out_e)

# (f) positive-lag rounds in 9961 (round positions and counts)
pos = p9961.filter(pl.col("lag") > 0)
print("\n(f) build 9961 positive-lag rows:", pos.height, "| by round:", pos.group_by("round").len().sort("round").rows())
print("(f) build 9961 positive-lag rows values (top):", pos.group_by("lag").len().sort("len", descending=True).head(6).rows())
