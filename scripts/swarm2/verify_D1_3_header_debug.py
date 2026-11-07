"""verify D1-3: debug of the team-A mapping for the header arbitration (read-only, 2024 tome).

Prints, for two matches (one where S2 holds with the header, one where neither holds), the per-round
sideA (team A side from player_info), winner, S2, kill side, bomb side, and running totals.
Also reports how many d01-07 matches pass the team-A construction (coverage check).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


HD = pl.concat([pl.read_parquet(f, columns=["key", "t_starters_score_final", "ct_starters_score_final"])
                for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))], how="vertical_relaxed")
HD = HD.unique("key", keep="first").rename({"key": "match_key", "t_starters_score_final": "hA", "ct_starters_score_final": "hB"})
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).unique(["match_key", "round", "player_id_fixed"], keep="first")
RE = rd("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")

r1 = PI.filter(pl.col("round") == 1)
print("round-1 rows per match: distribution of (nT, nCT)")
print(r1.group_by(["match_key", "team_code"]).len().group_by("team_code").agg(pl.col("len").value_counts()).head(3))
cnt = r1.group_by(["match_key", "team_code"]).len().pivot(on="team_code", index="match_key", values="len").fill_null(0)
print("matches with round-1 player_info:", cnt.height)
print(cnt.group_by(["2", "3"]).len().sort("len", descending=True).head(6))
dd = pl.concat([pl.DataFrame({"match_key": []})])
days = cnt.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print("matches by day group (round-1 present):", days.with_columns(pl.col("day").str.slice(0, 2)).group_by("day").len().sort("day").head(30))
print("header keys matched to round-1 match keys:", cnt.join(HD, on="match_key", how="inner").height, "of", cnt.height)

# pick matches
ok_pick = "csds/2024/02/12/m_partial"  # S2 holds, WIN fails (from header arbitration)
bad_pick_rows = PI.filter(pl.col("match_key").str.contains("csds/2024/02/12/") ).select("match_key").unique().head(0)
mA = [m for m in HD["match_key"].to_list() if m.startswith(ok_pick)]
print("\nmatch with S2 holds, WIN fails:", mA[:1])
mB = HD.filter(pl.col("match_key").str.contains("csds/2024/02/21/m_partial")).select("match_key").row(0)[0]
print("second match:", mB)


def show(mk):
    print("\n" + "=" * 110 + f"\n{mk}  header hA={HD.filter(pl.col('match_key') == mk)['hA'][0]} hB={HD.filter(pl.col('match_key') == mk)['hB'][0]}\n" + "=" * 110)
    pi = PI.filter(pl.col("match_key") == mk)
    A = pi.filter((pl.col("round") == 1) & (pl.col("team_code") == 2)).select("player_id_fixed").to_series().to_list()
    print("team A (round-1 T players):", sorted(A))
    sa = (pi.filter(pl.col("player_id_fixed").is_in(A)).group_by("round")
          .agg((pl.col("team_code") == 2).sum().alias("aT"), (pl.col("team_code") == 3).sum().alias("aCT"),
               pl.len().alias("nA_rows")))
    sa = sa.with_columns(pl.when(pl.col("aT") > pl.col("aCT")).then(2).when(pl.col("aCT") > pl.col("aT")).then(3).otherwise(None).alias("sideA"))
    re_ = RE.filter(pl.col("match_key") == mk).select(["round", "winner_team_code"])
    out = sa.join(re_, on="round", how="full", coalesce=True).sort("round")
    print(out)


show(mA[0])
show(mB)
