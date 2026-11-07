"""verify D7-5 step 1: independent round-level table for the 2024 tome (raw parquet, projection only).

Reads the raw tome channels: header, round_end, round_state (round_freeze_end), bomb_state, player_death.
Builds one row per (match_key, round) with:
  - build_num and match_date from header
  - round_end winner_team_code, win_reason_code, win_reason_message, tick
  - round_freeze_end t_score/ct_score (and _raw), tick
  - bomb outcome (exploded -> side 2, defused -> side 3) from bomb_state event_type only (not team codes)
  - death-based elimination side: side whose 5 players are all dead inside [freeze_end tick, round_end tick]
  - ledger side A: free min-swap orientation DP (re-implemented here, independent of D7_8)
  - ledger side B: fixed structural schedule (swap before rounds 13, 25, and 28, 31, ... every 3 rounds)

ASSUMPTION: 5v5 rounds are player_count == 20 (used later for the main analysis).
ASSUMPTION: bomb_planted rows repeat per tick, so the first tick per (match, round, event) is used.
ASSUMPTION: deaths are counted as distinct player_id per side inside the round window.
Intermediate parquet goes to the scratchpad only (not the repo, not E:).
"""
import glob
import sys
import collections
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(200)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")
SCR.mkdir(parents=True, exist_ok=True)


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(ch, cols):
    return pl.scan_parquet(files(BASE / f"basic_{ch}{SUF}")).select(cols)


# ---------- header: match -> build, date ----------
H = (pl.scan_parquet(files(BASE / "header"))
     .select(["key", "match_id", "match_date", "build_num", "map_name"]).collect()
     .rename({"key": "match_key"}))
print("header rows:", H.height, "distinct match_key:", H["match_key"].n_unique())
H = H.with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime(strict=False).alias("mdt"))
print("header mdt nulls:", H["mdt"].null_count(), "| build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
print("header mdt min/max:", H["mdt"].min(), H["mdt"].max())

# ---------- round_end (engine record) ----------
RE_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                            "win_reason_message", "player_count"]).collect()
print("round_end rows:", RE_raw.height, "| duplicate (match,round) rows:",
      RE_raw.height - RE_raw.unique(["match_key", "round"]).height)
RE = RE_raw.unique(["match_key", "round"], keep="first").rename({"tick": "re_tick"})

# ---------- round_state: round_freeze_end (score ledger source) ----------
FE_raw = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                              "t_score_raw", "ct_score_raw", "is_warmup"]).filter(
    pl.col("event_type") == "round_freeze_end").collect()
print("freeze_end rows:", FE_raw.height, "| duplicate (match,round) rows:",
      FE_raw.height - FE_raw.unique(["match_key", "round"]).height)
FE = (FE_raw.sort("tick").unique(["match_key", "round"], keep="first")
      .select(["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "is_warmup"])
      .rename({"tick": "fe_tick"}))

# ---------- bomb_state: outcome and planter (first tick per event) ----------
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"]).filter(
    pl.col("event_type").is_in(["bomb_planted", "bomb_exploded", "bomb_defused"])).collect()
print("bomb_state rows:", BS.height, BS.group_by("event_type").len().sort("event_type").rows())


def first_tick(ev, name):
    return (BS.filter(pl.col("event_type") == ev)
            .group_by(["match_key", "round"])
            .agg(pl.col("tick").min().alias(name)))


plant = (BS.filter(pl.col("event_type") == "bomb_planted").sort("tick")
         .group_by(["match_key", "round"])
         .agg(pl.col("tick").min().alias("plant_tick"), pl.col("player_team_code").first().alias("plant_team")))
expl = first_tick("bomb_exploded", "exp_tick")
defu = first_tick("bomb_defused", "def_tick")
n_exp_def = (BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
             .group_by(["match_key", "round"]).agg(pl.col("event_type").n_unique().alias("n_outcome_types")))

# ---------- player_death: distinct victims per side inside the round window (computed after the join) ----------
PD = scan("player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
print("player_death rows:", PD.height, "| null player_id:", PD["player_id"].null_count(),
      "| team codes:", PD.group_by("player_team_code").len().sort("player_team_code").rows())

# ---------- base round table ----------
R = FE.join(RE, on=["match_key", "round"], how="full", coalesce=True)
print("rounds: freeze_end only:", R.filter(pl.col("re_tick").is_null()).height,
      "| round_end only:", R.filter(pl.col("fe_tick").is_null()).height,
      "| both:", R.filter(pl.col("fe_tick").is_not_null() & pl.col("re_tick").is_not_null()).height)
R = (R.join(H.select(["match_key", "build_num", "mdt"]), on="match_key", how="left")
     .join(plant, on=["match_key", "round"], how="left")
     .join(expl, on=["match_key", "round"], how="left")
     .join(defu, on=["match_key", "round"], how="left")
     .join(n_exp_def, on=["match_key", "round"], how="left"))
print("rounds with no header match:", R.filter(pl.col("build_num").is_null()).height)

# ---------- deaths inside [fe_tick, re_tick], distinct victims per side ----------
PDj = (PD.join(R.select(["match_key", "round", "fe_tick", "re_tick"]), on=["match_key", "round"], how="inner")
       .filter(pl.col("fe_tick").is_not_null() & pl.col("re_tick").is_not_null())
       .filter((pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
       .filter(pl.col("player_id").is_not_null())
       .unique(["match_key", "round", "player_id"], keep="first"))
dc = (PDj.group_by(["match_key", "round", "player_team_code"]).len()
      .pivot(on="player_team_code", index=["match_key", "round"], values="len")
      .rename({"2": "nT", "3": "nCT"}))
for c in ["nT", "nCT"]:
    if c not in dc.columns:
        dc = dc.with_columns(pl.lit(0).alias(c))
dc = dc.select(["match_key", "round", "nT", "nCT"]).with_columns(pl.col("nT").fill_null(0), pl.col("nCT").fill_null(0))
R = R.join(dc, on=["match_key", "round"], how="left").with_columns(
    pl.col("nT").fill_null(0), pl.col("nCT").fill_null(0))
R = R.with_columns(
    pl.when((pl.col("nT") == 5) & (pl.col("nCT") < 5)).then(pl.lit(3))
      .when((pl.col("nCT") == 5) & (pl.col("nT") < 5)).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side"))

# ---------- ledger from freeze_end scores, per match ----------
def swap_before(r):
    return r == 13 or r == 25 or (r >= 28 and (r - 25) % 3 == 0)


def orient_schedule(r):
    o = 0
    for k in range(2, r + 1):
        if swap_before(k):
            o ^= 1
    return o


FEp = FE.sort(["match_key", "round"])
per_match = collections.defaultdict(dict)
for mk, rd, t, c in FEp.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    per_match[mk][rd] = (t, c)


def A_B(t, c, o):
    # team A (on T at round 1) and team B; o=0 means A is on T
    return (t, c) if o == 0 else (c, t)


def pair_sides_schedule(t0, c0, r0, t1, c1, r1):
    o0, o1 = orient_schedule(r0), orient_schedule(r1)
    A0, B0 = A_B(t0, c0, o0)
    A1, B1 = A_B(t1, c1, o1)
    dA, dB = A1 - A0, B1 - B0
    if (dA, dB) == (1, 0):
        return True, (2 if o0 == 0 else 3)
    if (dA, dB) == (0, 1):
        return True, (2 if o0 == 1 else 3)
    return False, None


def valid_free(t0, c0, o0, t1, c1, o1):
    A0, B0 = A_B(t0, c0, o0)
    A1, B1 = A_B(t1, c1, o1)
    dA, dB = A1 - A0, B1 - B0
    if (dA, dB) == (1, 0):
        return True, (2 if o0 == 0 else 3)
    if (dA, dB) == (0, 1):
        return True, (2 if o0 == 1 else 3)
    return False, None


INF = 10 ** 9
dp_rows = []     # (match_key, round, dp_side, dp_status)
sch_rows = []    # (match_key, round, sch_side, sch_valid)
amb_rows = []    # (match_key, round, noswap_valid, swap_valid)
stat = collections.Counter()
for mk, d in per_match.items():
    rs = sorted(d)
    runs, cur = [], [rs[0]]
    for r in rs[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    for run in runs:
        L = len(run)
        if L == 1:
            stat["single_round_run"] += 1
            continue
        # structural schedule and structural ambiguity per pair
        for i in range(L - 1):
            r0, r1 = run[i], run[i + 1]
            t0, c0 = d[r0]
            t1, c1 = d[r1]
            ns = (t1 - t0, c1 - c0) in ((1, 0), (0, 1))
            sw = (c1 - t0, t1 - c0) in ((1, 0), (0, 1))
            amb_rows.append((mk, r0, ns, sw))
            ok, side = pair_sides_schedule(t0, c0, r0, t1, c1, r1)
            sch_rows.append((mk, r0, side, ok))
        # free min-swap DP over orientation (cost = number of orientation changes)
        V = []
        for i in range(L - 1):
            r0, r1 = run[i], run[i + 1]
            t0, c0 = d[r0]
            t1, c1 = d[r1]
            m = {}
            for o0 in (0, 1):
                for o1 in (0, 1):
                    m[(o0, o1)] = valid_free(t0, c0, o0, t1, c1, o1)
            V.append(m)
        f = [[0, 0]]
        for i in range(L - 1):
            nf = [INF, INF]
            for o1 in (0, 1):
                for o0 in (0, 1):
                    ok, _ = V[i][(o0, o1)]
                    if ok and f[i][o0] < INF:
                        nf[o1] = min(nf[o1], f[i][o0] + (1 if o0 != o1 else 0))
            f.append(nf)
        total = min(f[L - 1])
        if total >= INF:
            stat["infeasible_run"] += 1
            for i in range(L - 1):
                dp_rows.append((mk, run[i], None, "infeasible"))
            continue
        g = [None] * L
        g[L - 1] = [0, 0]
        for i in range(L - 2, -1, -1):
            gi = [INF, INF]
            for o0 in (0, 1):
                for o1 in (0, 1):
                    ok, _ = V[i][(o0, o1)]
                    if ok and g[i + 1][o1] < INF:
                        gi[o0] = min(gi[o0], (1 if o0 != o1 else 0) + g[i + 1][o1])
            g[i] = gi
        stat["runs_feasible"] += 1
        for i in range(L - 1):
            sides = set()
            for o0 in (0, 1):
                for o1 in (0, 1):
                    ok, side = V[i][(o0, o1)]
                    if not ok or f[i][o0] >= INF or g[i + 1][o1] >= INF:
                        continue
                    if f[i][o0] + (1 if o0 != o1 else 0) + g[i + 1][o1] == total:
                        sides.add(side)
            if len(sides) == 1:
                dp_rows.append((mk, run[i], sides.pop(), "unique"))
                stat["pairs_unique"] += 1
            else:
                dp_rows.append((mk, run[i], None, "ambiguous" if sides else "no_path"))
                stat["pairs_" + ("ambiguous" if sides else "no_path")] += 1
print("DP run stats:", dict(stat))

DP = pl.DataFrame(dp_rows, schema={"match_key": pl.Utf8, "round": pl.Int64, "dp_side": pl.Int64,
                                   "dp_status": pl.Utf8}, orient="row")
SCH = pl.DataFrame(sch_rows, schema={"match_key": pl.Utf8, "round": pl.Int64, "sch_side": pl.Int64,
                                     "sch_valid": pl.Boolean}, orient="row")
AMB = pl.DataFrame(amb_rows, schema={"match_key": pl.Utf8, "round": pl.Int64, "noswap_valid": pl.Boolean,
                                     "swap_valid": pl.Boolean}, orient="row")
print("DP status:", DP.group_by("dp_status").len().sort("dp_status").rows())
print("schedule pair validity:", SCH.group_by("sch_valid").len().sort("sch_valid").rows())

R = (R.join(DP, on=["match_key", "round"], how="left")
     .join(SCH, on=["match_key", "round"], how="left")
     .join(AMB, on=["match_key", "round"], how="left"))
R = R.with_columns(
    pl.when(pl.col("exp_tick").is_not_null()).then(pl.lit(2))
      .when(pl.col("def_tick").is_not_null()).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
R = R.with_columns(
    pl.when(pl.col("fe_tick").is_null() | pl.col("re_tick").is_null()).then(pl.lit(None, dtype=pl.Boolean))
      .otherwise((pl.col("re_tick") >= pl.col("fe_tick"))).alias("re_after_fe"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
)
R = R.with_columns(
    pl.when(pl.col("plant_tick").is_not_null() & pl.col("fe_tick").is_not_null() & pl.col("re_tick").is_not_null())
      .then(((pl.col("plant_tick") >= pl.col("fe_tick")) & (pl.col("plant_tick") <= pl.col("re_tick"))))
      .otherwise(None).alias("plant_in_window"),
    pl.when(pl.col("exp_tick").is_not_null() & pl.col("fe_tick").is_not_null() & pl.col("re_tick").is_not_null())
      .then(((pl.col("exp_tick") >= pl.col("fe_tick")) & (pl.col("exp_tick") <= pl.col("re_tick"))))
      .otherwise(None).alias("exp_in_window"),
    pl.when(pl.col("def_tick").is_not_null() & pl.col("fe_tick").is_not_null() & pl.col("re_tick").is_not_null())
      .then(((pl.col("def_tick") >= pl.col("fe_tick")) & (pl.col("def_tick") <= pl.col("re_tick"))))
      .otherwise(None).alias("def_in_window"),
)
R = R.sort(["match_key", "round"])
R.write_parquet(str(SCR / "R_all_rounds.parquet"))
print("\nR rows:", R.height, "| wrote", SCR / "R_all_rounds.parquet")
print("R null-check build_num:", R["build_num"].null_count(), "| winner null:", R["winner_team_code"].null_count(),
      "| dp_side null:", R["dp_side"].null_count(), "| sch_side null:", R["sch_side"].null_count())
