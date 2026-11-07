"""verify D5-9 2026: does the round-9 pattern appear in the 2026 tome's 16-round (Wingman) matches?

2026 header carries is_wingman and max_rounds (1,752 matches: 149 with max_rounds 16 and is_wingman True,
1,603 with max_rounds 24). This script tests the same round_end-vs-truth agreement, per label, on the
2026 tome: score step (no-swap mapping, for labels whose next boundary is not a halftime), bomb outcome,
and last victim (no-bomb rounds). Reads projected channels only; writes only to scratchpad (C:).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_2026.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

B = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
S = ".2026-09-18,2026-10-08"


def scan(ch):
    fs = sorted(glob.glob(str(B / f"basic_{ch}{S}" / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


hf = sorted(glob.glob(str(B / "header" / "dataframe_*")))
H = pl.read_parquet(hf[0], columns=["key", "is_wingman", "max_rounds", "match_date", "map_name", "tick_rate",
                                    "unique_steamids", "max_unique_players"]).rename({"key": "match_key"})
wing = {r["match_key"]: r["is_wingman"] for r in H.iter_rows(named=True)}
mrounds = {r["match_key"]: r["max_rounds"] for r in H.iter_rows(named=True)}
print("2026 headers:", H.height, " is_wingman x max_rounds:", H.group_by(["is_wingman", "max_rounds"]).len().to_dicts())

RS = scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type").is_in(["round_freeze_end", "round_announce_last_round_half", "round_announce_match_start"])).collect()
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"], maintain_order=True).agg(
          pl.col("t_score").first().alias("fe_t"), pl.col("ct_score").first().alias("fe_ct"),
          pl.col("tick").first().alias("fe_tick")))
LH = (RS.filter(pl.col("event_type") == "round_announce_last_round_half").group_by("match_key")
      .agg(pl.col("round").min().alias("lh_lab")))
lh = {r["match_key"]: r["lh_lab"] for r in LH.iter_rows(named=True)}
MS = RS.filter(pl.col("event_type") == "round_announce_match_start").group_by("match_key").agg(pl.len().alias("n_ms"))
regA_like = set(MS["match_key"].to_list())
print("2026 matches with round_announce_match_start:", MS.height, " with last-round-half announcement:", len(lh))

RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
      .sort(["match_key", "round", "tick"]).group_by(["match_key", "round"], maintain_order=True).agg(
          pl.col("winner_team_code").first().alias("w"), pl.col("tick").first().alias("end_tick"),
          pl.col("player_count").first().alias("pc"), pl.len().alias("end_n")))
print("2026 round_end pairs:", RE.height, " with >1 row:", int((RE["end_n"] > 1).sum()))
print("2026 round_end player_count:", RE.group_by("pc").len().sort("len", descending=True).head(6).to_dicts())

BS = scan("bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"])).collect()
BO = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"])
      .agg(pl.col("event_type").unique().alias("evs")))
BO = BO.with_columns(
    bo_side=pl.when(pl.col("evs").list.contains("bomb_exploded") & ~pl.col("evs").list.contains("bomb_defused")).then(2)
    .when(pl.col("evs").list.contains("bomb_defused") & ~pl.col("evs").list.contains("bomb_exploded")).then(3)
    .otherwise(None)).select(["match_key", "round", "bo_side"])
PL = BS.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(pl.len().alias("n_pl"))
print("2026 bomb planted-team codes:", BS.filter(pl.col("event_type") == "bomb_planted").group_by("player_team_code").len().to_dicts())

DP = scan("player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
DPs = DP.sort(["match_key", "round", "tick"])
LVt = DPs.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("lv_tick"))
LV = (DPs.join(LVt, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("lv_tick"))
      .group_by(["match_key", "round"]).agg(pl.col("player_team_code").n_unique().alias("lv_nteam"),
                                            pl.col("player_team_code").max().alias("lv_victim")))
ROST = DP.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"))
rost = {r["match_key"]: r["n_players"] for r in ROST.iter_rows(named=True)}

R = (FE.join(RE, on=["match_key", "round"], how="full", coalesce=True)
     .join(BO, on=["match_key", "round"], how="left")
     .join(PL, on=["match_key", "round"], how="left")
     .join(LV, on=["match_key", "round"], how="left"))
rounds = defaultdict(dict)
for row in R.iter_rows(named=True):
    rounds[row["match_key"]][row["round"]] = row


def g(mk, r):
    return rounds.get(mk, {}).get(r)


def stats(mks, r):
    bn = bd = ln = ld = sn = sd = 0
    for mk in mks:
        row = g(mk, r)
        nxt = g(mk, r + 1)
        if not row or row.get("w") not in (2, 3):
            continue
        if nxt and None not in (row.get("fe_t"), row.get("fe_ct"), nxt.get("fe_t"), nxt.get("fe_ct")):
            dt, dc = nxt["fe_t"] - row["fe_t"], nxt["fe_ct"] - row["fe_ct"]
            truth = 2 if (dt, dc) == (1, 0) else 3 if (dt, dc) == (0, 1) else None
            if truth is not None:
                sn += 1
                sd += int(truth != row["w"])
        if row.get("bo_side") is not None:
            bn += 1
            bd += int(row["bo_side"] != row["w"])
        elif row.get("n_pl") is None and row.get("lv_nteam") == 1 and row.get("lv_victim") in (2, 3):
            ln += 1
            ld += int((5 - row["lv_victim"]) != row["w"])
    p = lambda a, b: f"{100.0 * a / b:5.1f}%" if b else "  -   "
    return f"step {sd}/{sn} ({p(sd, sn)})  bomb {bd}/{bn} ({p(bd, bn)})  LV {ld}/{ln} ({p(ld, ln)})"


all_mk = list(rounds.keys())
W = [m for m in all_mk if wing.get(m)]
F = [m for m in all_mk if wing.get(m) is False]
print("\nmatches in tome with rounds:", len(all_mk), " wingman:", len(W), " 5v5-format:", len(F))

print("\n=== distinct players per match (kill roster), by is_wingman")
print("  wingman:", Counter(rost.get(m) for m in W).most_common(6))
print("  5v5:    ", Counter(rost.get(m) for m in F).most_common(6))

print("\n=== halftime label (last-round-half announcement) by format")
print("  wingman:", Counter(lh.get(m) for m in W).most_common(6))
print("  5v5:    ", Counter(lh.get(m) for m in F).most_common(6))

print("\n=== round_end winner vs truths, WINGMAN (16-round) matches, per label")
for r in range(1, 17):
    print(f"  r={r:2d}: {stats(W, r)}")

print("\n=== round_end winner vs truths, 5v5 (24-round) matches, per label (subset)")
for r in list(range(1, 25)) + [25, 26, 27, 28]:
    print(f"  r={r:2d}: {stats(F, r)}")

print("\n=== wingman max final score and last label")
fin = Counter()
for m in W:
    rr = rounds[m]
    labs = [r for r, v in rr.items() if v.get("w") in (2, 3) or v.get("fe_t") is not None]
    if not labs:
        continue
    N = max(labs)
    row = rr.get(N)
    if row and row.get("fe_t") is not None and row.get("w") in (2, 3):
        t = row["fe_t"] + (row["w"] == 2)
        ct = row["fe_ct"] + (row["w"] == 3)
        fin[("max", max(t, ct))] += 1
        fin[("N", N)] += 1
print("  ", sorted(((k, v) for k, v in fin.items()), key=lambda kv: (kv[0][0], kv[0][1])))
