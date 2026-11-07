"""D6-4 independent verifier, part I: does map name explain the round-13 mismatch beyond build_num?
Counts round-13 bomb rows (player_count == 20) by map_name and build_num, with mismatch counts. Read-only.
"""
import sys, glob, collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


H = load(B24, "header", ["key", "build_num", "map_name"])
hm = {k: (b, mp) for k, b, mp in H.iter_rows()}
RE = load(B24, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = load(B24, "bomb_state", ["match_key", "round", "event_type"])
outs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
J = RE.filter((pl.col("round") == 13) & (pl.col("player_count") == 20)).join(outs, on=["match_key", "round"], how="inner")
c = collections.defaultdict(lambda: [0, 0])
for m, w, bs in J.select(["match_key", "winner_team_code", "bomb_side"]).iter_rows():
    b, mp = hm.get(m, (None, None))
    c[(b, mp)][0] += 1
    c[(b, mp)][1] += int(bs != w)
print("round-13 bomb rows (player_count==20) by build_num and map_name: (build, map): n, mismatch")
for k in sorted(c, key=lambda x: (x[0] or 0, str(x[1]))):
    print(f"  {k}: n={c[k][0]}, mismatch={c[k][1]}")
maps_9961 = {mp for (b, mp), v in c.items() if b == 9961}
print(f"maps with build 9961 round-13 bomb rows: {sorted(maps_9961)}")
print(f"maps with build 9842 round-13 bomb rows: {sorted({mp for (b, mp), v in c.items() if b == 9842})}")
