"""D6-4 verifier part E2: read-only raw E: parquet files for 4 affected build-9961 round-13 matches (same picks as part E).
Also checks the tome key day segment against the header match_date (date label used in the claim).
Only reads. Nothing is written under E:.
"""
import sys, glob
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")
T, CT = 2, 3


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


H = load(B24, "header", ["key", "build_num", "match_date"])
bm = {k: (b, d) for k, b, d in H.iter_rows()}
# claim's day label vs header match_date
H2 = H.with_columns(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("key_day"),
                    pl.col("match_date").str.slice(0, 10).alias("md"))
agree = H2.filter(pl.col("md").str.slice(8, 2) == pl.col("key_day")).height
print(f"tome key day folder vs header match_date day: agree={agree} of {H2.height}")
print(f"match_date day distribution for build 9961 matches (top): ")
print(str(H2.filter(pl.col("build_num") == 9961).group_by("md").len().sort("md").head(12)))

RE = load(B24, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = load(B24, "bomb_state", ["match_key", "round", "event_type", "tick"])
outs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("btick")))
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
J = RE.filter((pl.col("round") == 13) & (pl.col("player_count") == 20)).join(outs, on=["match_key", "round"], how="inner")
J = J.filter(pl.col("bomb_side") != pl.col("winner_team_code"))
J = J.filter(pl.col("match_key").map_elements(lambda m: bm.get(m, (0, ""))[0] == 9961, return_dtype=pl.Boolean))
J = J.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")).sort("day")
picks = []
for d in ["12", "15", "20", "25"]:
    sub = J.filter(pl.col("day") == d)
    if sub.height:
        picks.append(sub.row(0, named=True))

for p in picks:
    m = p["match_key"]
    mid = m.split("/")[4]
    folder = RAW / p["day"] / mid
    print(f"\n== match {m} | build {bm[m][0]} | header match_date {bm[m][1]} ==")
    print(f"tome round 13: round_end winner={p['winner_team_code']} tick={p['tick']} | bomb outcome side={p['bomb_side']} tick={p['btick']}")
    if not folder.exists():
        print(f"  raw folder missing: {folder}")
        continue
    re_raw = pl.read_parquet(folder / "round_end")
    print(f"  raw round_end columns: {re_raw.columns}")
    cols = [c for c in ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"] if c in re_raw.columns]
    print(str(re_raw.filter(pl.col("round").is_in([12, 13, 14])).select(cols).sort(["round", "tick"])))
    bs_raw = pl.read_parquet(folder / "bomb_state")
    bcols = [c for c in ["round", "tick", "event_type", "player_team_code"] if c in bs_raw.columns]
    print(f"  raw bomb_state rows rounds 12-14 with planted/defused/exploded:")
    print(str(bs_raw.filter(pl.col("round").is_in([12, 13, 14]) & pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).select(bcols).sort(["round", "tick"])))
