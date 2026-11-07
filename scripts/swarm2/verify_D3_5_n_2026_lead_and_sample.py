"""D3-5 verifier, step N (2026 bake-off tome + 1 Hz sample): direct derived-lead vs live-lead check and label check.

(a) derived lead (winner-based, bakeoff swap rule with the match's max_rounds, team A starts on T) vs live
    round_state ct_score - t_score at round_freeze_end, for every round with a live start, all 2026 matches.
(b) data/state_sample/rounds.parquet (1 Hz sample, keyed csds/2026/10/..): ct_win label vs round_end winner and vs
    the freeze-delta side; sample ct_score/t_score vs tome freeze fields.
Read only.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def rd(ch, cols):
    if ch == "header":
        folders = [str(TOME / "header")]
    else:
        folders = sorted(glob.glob(str(TOME / f"basic_{ch}.*")))
    files = []
    for fd in folders:
        files += sorted(glob.glob(str(Path(fd) / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


H = rd("header", ["key", "max_rounds"])
MR = {k: mr for k, mr in H.iter_rows()}
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
W = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    W[m][r] = w
LIVE = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    LIVE[m][r] = (t, c)

print("== 2026 derived lead vs live lead, and 1 Hz sample labels ==")
n = collections.Counter()
ex = []
for m in W:
    if m not in MR:
        continue
    mr = MR[m]
    side_a, a, b = T, 0, 0
    for r in sorted(W[m]):
        if r in LIVE[m] and LIVE[m][r][0] is not None and LIVE[m][r][1] is not None:
            t, c = LIVE[m][r]
            derived = (b - a) if side_a == T else (a - b)
            n["rounds_compared"] += 1
            if derived == c - t:
                n["agree"] += 1
            else:
                n["disagree"] += 1
                if len(ex) < 3:
                    ex.append((m[-45:], r, derived, c - t))
        w = W[m][r]
        if w == side_a:
            a += 1
        elif w == (CT if side_a == T else T):
            b += 1
        if swap_expected(r, mr):
            side_a = CT if side_a == T else T
print("(a) 2026 derived vs live lead:", dict(n), "examples:", ex)

# (b) 1 Hz sample labels
p = ROOT / "data" / "state_sample" / "rounds.parquet"
S = pl.read_parquet(str(p), columns=["match_key", "round", "ct_win", "ct_score", "t_score", "live_start_tick"])
print("(b) sample rows:", S.height, "| matches:", S["match_key"].n_unique(),
      "| round range:", S["round"].min(), S["round"].max(), "| ct_win values:", S["ct_win"].unique().to_list())
cnt = collections.Counter()
for m, r, cw, cs, ts in S.select(["match_key", "round", "ct_win", "ct_score", "t_score"]).iter_rows():
    if m not in W:
        cnt["match_not_in_tome"] += 1
        continue
    if r not in W[m]:
        cnt["round_not_in_round_end"] += 1
        continue
    cnt["rows_matched"] += 1
    cnt["label_eq_winner"] += int((cw == 1) == (W[m][r] == CT))
    if r in LIVE[m]:
        t, c = LIVE[m][r]
        cnt["score_rows_with_freeze"] += 1
        cnt["sample_scores_eq_freeze"] += int((cs, ts) == (c, t))
        if (r + 1) in LIVE[m]:
            t2, c2 = LIVE[m][r + 1]
            dd = (t2 - t, c2 - c)
            if dd in [(1, 0), (0, 1)]:
                D = T if dd == (1, 0) else CT
                cnt["delta_valid"] += 1
                cnt["label_eq_delta"] += int((cw == 1) == (D == CT))
print("    counts:", dict(cnt))
print("== end ==")
