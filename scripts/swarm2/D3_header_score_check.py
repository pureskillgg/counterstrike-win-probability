"""D3 secondary: check which round-winner labels agree with the header final scores (2024 G: tome).

Idea: header t_starters_score_final / ct_starters_score_final are the final scores of the team that
started on T and the team that started on CT. Team A = the team that started on T. Under MR12 with
side swaps at halftime and at OT start, A's side per round is:
  r <= 12: T;  13 <= r <= 24: CT;  OT block k = (r-25)//3: T if k even, CT if k odd  (ASSUMPTION)
If the round labels are right, the number of rounds A wins equals t_starters_score_final, and the
rounds the other team wins equal ct_starters_score_final.

Three label sets per round, each a winner code (2 = T, 3 = CT):
  L_RE    : round_end.winner_team_code (as used by the bake-off)
  L_BOMB  : bomb side (defused -> 3, exploded -> 2) where a bomb outcome exists, else L_RE
  L_SCORE : score-change side (round_state freeze_end of r+1 minus r) where available, else L_RE
Only matches whose round count equals the header total (complete matches) are scored.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_header_score_check.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def g_read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


H = g_read("header", ["key", "match_date", "t_starters_score_final", "ct_starters_score_final",
                      "number_of_points", "map_name"])
RE = g_read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
print("header", H.height, "round_end", RE.height)

rounds_dup = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
print("duplicate (match, round) in round_end (all rows):", rounds_dup)
print("player_count distribution over all round_end rows (top):",
      RE.group_by("player_count").len().sort("len", descending=True).head(6).rows())
RE = RE.unique(["match_key", "round"], keep="first")  # ASSUMPTION: first row kept for duplicates

outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
nxt = fe.with_columns((pl.col("round") - 1).alias("round_prev")).select(
    ["match_key", "round_prev", pl.col("t_score").alias("t_after"), pl.col("ct_score").alias("ct_after")])
cur = fe.select(["match_key", "round", pl.col("t_score").alias("t_before"), pl.col("ct_score").alias("ct_before")])
sc = cur.join(nxt, left_on=["match_key", "round"], right_on=["match_key", "round_prev"], how="left")
sc = sc.with_columns(
    pl.when(((pl.col("ct_after") - pl.col("ct_before")) == 1) & ((pl.col("t_after") - pl.col("t_before")) == 0))
    .then(pl.lit(CT))
    .when(((pl.col("t_after") - pl.col("t_before")) == 1) & ((pl.col("ct_after") - pl.col("ct_before")) == 0))
    .then(pl.lit(T)).otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))

R = (RE.join(outc, on=["match_key", "round"], how="left")
     .join(sc.select(["match_key", "round", "score_side"]), on=["match_key", "round"], how="left")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT))
         .when(pl.col("evs").list.contains("bomb_exploded")).then(pl.lit(T))
         .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
     .with_columns(
         pl.col("bomb_side").fill_null(pl.col("winner_team_code")).alias("L_BOMB"),
         pl.col("score_side").fill_null(pl.col("winner_team_code")).alias("L_SCORE"),
         pl.col("winner_team_code").alias("L_RE"),
         pl.col("round").alias("r")))
R = R.with_columns(pl.col("L_BOMB").alias("_b"), pl.col("bomb_side").is_not_null().alias("has_bomb"),
                   pl.col("score_side").is_not_null().alias("has_score"))

# per-match: is it affected (round-13 bomb mismatch)?
r13m = R.filter((pl.col("r") == 13) & pl.col("has_bomb") & (pl.col("bomb_side") != pl.col("winner_team_code")))
affected = set(r13m["match_key"].to_list())


def a_side(r):
    if r <= 12:
        return T
    if r <= 24:
        return CT
    k = (r - 25) // 3
    return T if k % 2 == 0 else CT


def a_wins(rows, label):
    n = 0
    for r, w in rows:
        if w == a_side(r):
            n += 1
    return n


per = collections.defaultdict(list)
for m, r, lre, lb, ls in R.select(["match_key", "r", "L_RE", "L_BOMB", "L_SCORE"]).iter_rows():
    per[m].append((r, lre, lb, ls))

hdr = {k: (t, c) for k, t, c in H.select(["key", "t_starters_score_final", "ct_starters_score_final"]).iter_rows()}

res = collections.Counter()
examples = collections.defaultdict(list)
for m, rows in per.items():
    if m not in hdr:
        res["no_header"] += 1
        continue
    t_fin, ct_fin = hdr[m]
    if t_fin is None or ct_fin is None:
        res["header_score_null"] += 1
        continue
    n = len(rows)
    if n != t_fin + ct_fin:
        res["incomplete_or_ot_total_mismatch"] += 1
        continue
    grp = "affected" if m in affected else "unaffected"
    res[f"complete_{grp}"] += 1
    for name, idx in (("L_RE", 1), ("L_BOMB", 2), ("L_SCORE", 3)):
        a = sum(1 for row in rows if row[idx] == a_side(row[0]))
        b = n - a
        ok = (a == t_fin and b == ct_fin)
        ok_swap = (a == ct_fin and b == t_fin)
        res[f"{name}_{grp}_ok"] += int(ok)
        res[f"{name}_{grp}_swapped"] += int(ok_swap and not ok)
        res[f"{name}_{grp}_total"] += 1
        if not ok and len(examples[f"{name}_{grp}"]) < 3:
            examples[f"{name}_{grp}"].append((m, n, t_fin, ct_fin, a, b))

print("\nmatch counts:")
for k, v in sorted(res.items()):
    print(f"  {k:<40} {v}")
print("\nexamples of non-matching complete matches (match, n_rounds, hdrT, hdrCT, A_wins, B_wins):")
for k, v in examples.items():
    print(" ", k, v)

# calibration: how many complete unaffected matches have A-side labelled wins equal to header?
print("\ncalibration check: unaffected complete matches with L_RE match (should be ~all):",
      res["L_RE_unaffected_ok"], "of", res["L_RE_unaffected_total"])
print("affected complete matches: L_RE ok", res["L_RE_affected_ok"], "/", res["L_RE_affected_total"],
      "| L_BOMB ok", res["L_BOMB_affected_ok"], "/", res["L_BOMB_affected_total"],
      "| L_SCORE ok", res["L_SCORE_affected_ok"], "/", res["L_SCORE_affected_total"])
print("affected matches total in tome (any round-13 mismatch):", len(affected))
