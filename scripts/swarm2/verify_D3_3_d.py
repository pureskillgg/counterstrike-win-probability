"""verify_D3_3 part D: third independent oracle, player deaths and bomb timing, on the disputed rounds.

Per (match, round) in the 2024 tome, using player_death (victim player_team_code = side at death) and bomb_state:
  - distinct victims per side between freeze(k) and round_end(k)
  - elimination winner: the side whose distinct victims reached 5 (ASSUMPTION: 5 per side, 5v5)
  - bomb outcome and its tick relative to round_end(k) tick
Compares round_end winner, score-delta winner and elimination winner on
  (a) disputed round-13 rows (MR24, day 12-25) and (b) non-disputed MR24 d12-25 round-13 rows and round-12/14 controls.
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


def files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def day(m):
    return int(m.split("/")[3])


ann = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann")).collect())
MR = {m: int(r) * 2 + 2 for m, r in ann.iter_rows()}
fe_df = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
         .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect().sort("tick")
         .group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("t_score").first(),
                                               pl.col("ct_score").first()))
F, FT = collections.defaultdict(dict), collections.defaultdict(dict)
for m, r, tk, t, c in fe_df.select(["match_key", "round", "tick", "t_score", "ct_score"]).iter_rows():
    F[m][int(r)] = (int(t), int(c))
    FT[m][int(r)] = int(tk)
re_df = (pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code",
                                                    "win_reason_code"]).collect()
         .sort("tick").group_by(["match_key", "round"]).agg(pl.col("tick").first(),
                                                            pl.col("winner_team_code").first(),
                                                            pl.col("win_reason_code").first()))
RE = {(m, int(r)): (int(tk), int(w), int(rc)) for m, r, tk, w, rc in
      re_df.select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).iter_rows()}

dth = (pl.scan_parquet(files("player_death")).select(["match_key", "round", "tick", "player_id", "player_team_code"])
       .collect())
dth_g = collections.defaultdict(list)
for m, r, tk, pid, pt in dth.iter_rows():
    dth_g[(m, int(r))].append((int(tk), int(pid), int(pt) if pt is not None else -1))
bs = (pl.scan_parquet(files("bomb_state")).filter(pl.col("event_type").is_in(
      ["bomb_planted", "bomb_exploded", "bomb_defused"])).select(["match_key", "round", "tick", "event_type"]).collect())
bomb = collections.defaultdict(dict)
for m, r, tk, ev in bs.iter_rows():
    bomb[(m, int(r))].setdefault(ev, int(tk))

pi = (pl.scan_parquet(files("player_info")).select(["match_key", "round", "player_id", "team_code"]).collect())
roster = collections.defaultdict(lambda: {2: set(), 3: set()})
for m, r, pid, tc in pi.iter_rows():
    if tc in (2, 3):
        roster[(m, int(r))][int(tc)].add(pid)
print("player_death rows", dth.height, "| player_info rows", pi.height)


def elim(m, k):
    """Distinct victims per side between freeze(k) and round_end(k); side code of victim is player_team_code."""
    if (m, k) not in RE or k not in F[m]:
        return None
    t_end = RE[(m, k)][0]
    t_frz = FT[m][k]
    victims = {T: set(), CT: set()}
    for tk, pid, pt in dth_g.get((m, k), []):
        if t_frz <= tk <= t_end and pt in (T, CT):
            victims[pt].add(pid)
    n = {s: len(victims[s]) for s in (T, CT)}
    if n[T] >= 5 and n[CT] < 5:
        return CT, n
    if n[CT] >= 5 and n[T] < 5:
        return T, n
    return ("undetermined", n)


def delta_winner(m, k, mr):
    if k + 1 not in F[m] or k not in F[m] or swap_expected(k, mr):
        return None
    t, c = F[m][k]
    t2, c2 = F[m][k + 1]
    dT, dC = t2 - t, c2 - c
    if dC == 1 and dT == 0:
        return CT
    if dT == 1 and dC == 0:
        return T
    return None


# disputed set: MR24, d12_25, round 13 with delta != round_end winner (same definition as the claim)
disputed = set()
for m in F:
    if day(m) < 12 or MR.get(m, 24) != 24:
        continue
    d = delta_winner(m, 13, 24)
    if d is not None and (m, 13) in RE and d != RE[(m, 13)][1]:
        disputed.add(m)
print("disputed round-13 MR24 d12_25 matches:", len(disputed))

for label, sel in (("disputed r13", lambda m: m in disputed),
                   ("non-disputed r13 (MR24 d12_25)", lambda m: m not in disputed and day(m) >= 12 and MR.get(m, 24) == 24),
                   ("control r12 (MR24 d12_25, not swap-scheduled as delta)", None),
                   ("control r14 (MR24 d12_25)", None)):
    cnt = collections.Counter()
    k = 13 if "r13" in label else (12 if "r12" in label else 14)
    for m in F:
        if sel is not None:
            if not sel(m):
                continue
        elif day(m) < 12 or MR.get(m, 24) != 24:
            continue
        if (m, k) not in RE:
            continue
        e = elim(m, k)
        if e is None:
            cnt["no_freeze"] += 1
            continue
        ew, n = e
        rew = RE[(m, k)][1]
        dw = delta_winner(m, k, 24)
        cnt[("elim", ew if isinstance(ew, str) else ("T" if ew == T else "CT"))] += 1
        if not isinstance(ew, str):
            cnt["rend_agrees_elim" if ew == rew else "rend_disagrees_elim"] += 1
            if dw is not None:
                cnt["delta_agrees_elim" if dw == ew else "delta_disagrees_elim"] += 1
        bm = bomb.get((m, k), {})
        if bm:
            cnt[("bomb_row", "exploded" if "bomb_exploded" in bm else ("defused" if "bomb_defused" in bm else "planted_only"))] += 1
            if "bomb_exploded" in bm or "bomb_defused" in bm:
                ev = "bomb_exploded" if "bomb_exploded" in bm else "bomb_defused"
                cnt[("bomb_tick_vs_rend_tick", ev, "before" if bm[ev] <= RE[(m, k)][0] else "after")] += 1
        cnt[("rend_reason", RE[(m, k)][2], "winner", rew)] += 1
    print(f"\n== {label}, round {k}, matches considered: {sum(v for kk, v in cnt.items() if isinstance(kk, tuple) and kk[0]=='elim')} ==")
    for kk, v in sorted(cnt.items(), key=lambda kv: str(kv[0])):
        print("   ", kk, v)
