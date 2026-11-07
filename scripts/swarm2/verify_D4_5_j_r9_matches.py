r"""Verifier for D4-5, step J: inspect complete 2024 matches whose ONLY score-vs-round_end disagreement is round 9
(and, as a control, round 13), to see why neither labelling fits the header total.

For each such match prints: header tA/tB (starters final), per-round round_end winner (RE), score side (SC, '.' if
ambiguous), and the observed side-flip rate at boundaries 7..13 (from player side codes; pairs >= 1).
Also tests the alternative that the header starter columns are swapped (tA<->tB) and reports fit counts.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_j_r9_matches.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(18)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"]).with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("fsum"))
re_ = (read("round_end", ["match_key", "round", "tick", "winner_team_code"])
       .sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first"))
fe = (read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first"))
nxt = fe.select(["match_key", "round", "t_score", "ct_score"]).with_columns((pl.col("round") - 1).alias("round")).rename(
    {"t_score": "t2", "ct_score": "c2"})
sd = fe.select(["match_key", "round", "t_score", "ct_score"]).join(nxt, on=["match_key", "round"], how="inner").with_columns(
    (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
    pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
    .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("sc"))
L = re_.join(sd.select(["match_key", "round", "sc"]), on=["match_key", "round"], how="left").with_columns(
    pl.col("sc").fill_null(-1))
pm = re_.group_by("match_key").agg(pl.len().alias("n_rows"), pl.col("round").max().alias("mx"), pl.col("round").min().alias("mn"))
pm = pm.join(H.select(["key", "fsum", "t_starters_score_final", "ct_starters_score_final"]), left_on="match_key",
             right_on="key", how="left")
pm = pm.filter((pl.col("mx") == pl.col("n_rows")) & (pl.col("mn") == 1) & (pl.col("fsum") == pl.col("mx")) & (pl.col("mx") <= 24))
L = L.join(pm.select(["match_key", "n_rows", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="inner")
L = L.with_columns(pl.when((pl.col("sc") != -1) & (pl.col("sc") != 0) & (pl.col("sc") != pl.col("winner_team_code")))
                   .then(pl.lit(1)).otherwise(pl.lit(0)).alias("dis"))
per = L.group_by("match_key").agg(pl.col("dis").sum().alias("ndis"), pl.col("dis").arg_max().alias("_"),
                                  pl.col("t_starters_score_final").first().alias("tA"),
                                  pl.col("ct_starters_score_final").first().alias("tB"), pl.col("n_rows").first().alias("n"))
one9 = per.filter(pl.col("ndis") == 1)
dis_round = L.filter(pl.col("dis") == 1).select(["match_key", "round"]).unique("match_key", keep="first")
one9 = one9.join(L.filter(pl.col("dis") == 1).group_by("match_key").agg(pl.col("round").first().alias("dr")),
                 on="match_key", how="left")
print("one-disagreement matches by round:", one9.group_by("dr").len().sort("dr").rows())

# side-flip rates from player side codes, boundaries 7..13
de = read("player_death", ["match_key", "round", "player_id", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
pr = de.group_by(["match_key", "round", "player_id"]).agg(pl.col("player_team_code").first().alias("side"))
b = pr.with_columns((pl.col("round") - 1).alias("round")).rename({"side": "side_next"})
pairs = pr.join(b, on=["match_key", "round", "player_id"], how="inner").with_columns(
    (pl.col("side") != pl.col("side_next")).alias("flip"))
fr = pairs.group_by(["match_key", "round"]).agg(pl.len().alias("np"), pl.col("flip").mean().alias("fr"))

target = one9.filter(pl.col("dr") == 9)["match_key"].to_list()
ctrl = one9.filter(pl.col("dr") == 13)["match_key"].to_list()[:40]
for label, keys in [("DR9", target[:12]), ("DR13 control", ctrl[:6])]:
    print(f"\n==================== {label} ====================")
    for k in keys:
        rows = L.filter(pl.col("match_key") == k).sort("round").select(["round", "winner_team_code", "sc"]).to_dicts()
        t = per.filter(pl.col("match_key") == k).row(0, named=True)
        f = fr.filter(pl.col("match_key") == k).sort("round").select(["round", "np", "fr"]).to_dicts()
        fl = {x["round"]: (x["np"], round(x["fr"], 2)) for x in f if 6 <= x["round"] <= 14}
        print(k[-40:], f"tA={t['tA']} tB={t['tB']} n={t['n']}")
        print("  RE:", " ".join(str(r["winner_team_code"]) for r in rows))
        print("  SC:", " ".join(("." if r["sc"] in (0, -1) else str(r["sc"])) for r in rows))
        print("  flip(boundary r->r+1) for r=6..14:", fl)

# fit counts with the header columns possibly swapped
def fits(keys, label_col):
    res = []
    for k in keys:
        g = L.filter(pl.col("match_key") == k).sort("round")
        t = per.filter(pl.col("match_key") == k).row(0, named=True)
        xs = [1 if (r["round"] <= 12 and r[label_col] == T) or (r["round"] > 12 and r[label_col] == CT) else 0
              for r in g.iter_rows(named=True) if True]
        xr = sum(1 for r in g.iter_rows(named=True) if ((r["round"] <= 12 and r["winner_team_code"] == T) or (r["round"] > 12 and r["winner_team_code"] == CT)))
        res.append(((xr == t["tA"]) and (g.height - xr == t["tB"]), (xr == t["tB"]) and (g.height - xr == t["tA"])))
    return res


r9 = fits(target, "winner_team_code")
print("\nDR9 matches: RE fit (as header), RE fit (header swapped):", sum(a for a, _ in r9), sum(b for _, b in r9), "of", len(r9))
